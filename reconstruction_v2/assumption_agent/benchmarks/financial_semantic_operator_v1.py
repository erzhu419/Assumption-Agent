from __future__ import annotations

import argparse
import difflib
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence


FINANCIAL_SEMANTIC_OPERATOR_VERSION = (
    "consumed_train_minilm_qa_typed_13f_query_v1"
)
FINANCIAL_SEMANTIC_ASSET_VERSION = "financial_semantic_operator_asset_v1"
FINANCIAL_SEMANTIC_PLAN_VERSION = "financial_semantic_typed_plan_v1"
FINANCIAL_QUERY_RECEIPT_VERSION = "financial_semantic_query_receipt_v1"
FINANCIAL_QA_RUNTIME_ASSET_VERSION = "financial_distilbert_qa_runtime_asset_v1"

EMBEDDING_DIMENSION = 384
MAXIMUM_INSTRUCTION_BYTES = 64 * 1024
MAXIMUM_QUESTION_BLOCKS = 4
MAXIMUM_ENTITY_CHARACTERS = 160
QUERY_CHUNK_ROWS = 250_000

OPERATION_ORDER = (
    "q3_aum",
    "q3_stock_count",
    "quarter_increase_rank",
    "q3_manager_rank",
)
OPERATIONS_BY_BLOCK_COUNT: Mapping[int, tuple[str, ...]] = {
    3: ("q3_aum", "quarter_increase_rank", "q3_manager_rank"),
    4: OPERATION_ORDER,
}
OPERATION_PROTOTYPES: Mapping[str, str] = {
    "q3_aum": (
        "Find the current-quarter assets under management of a named fund."
    ),
    "q3_stock_count": (
        "Count the stock positions held by a named fund in the current quarter."
    ),
    "quarter_increase_rank": (
        "Rank securities by dollar-value investment increase between two "
        "quarters for a named fund."
    ),
    "q3_manager_rank": (
        "Rank fund managers holding a named company by current-quarter "
        "position value."
    ),
}
QA_PROMPTS: Mapping[str, str] = {
    "q3_aum": "What full name follows 'AUM of'?",
    "q3_stock_count": "Which fund holds the stocks?",
    "quarter_increase_rank": "Which fund increased investment?",
    "q3_manager_rank": "What company did the requested fund managers invest in?",
}

# This intentionally reproduces the stock-class ontology present in the three
# consumed TRAIN solution programs.  Their adjacent string literals collapse
# the final terms into one value.  It is frozen here as benchmark metadata
# semantics, not inferred from prospective instructions or verifier output.
TRAIN_DEFINED_STOCK_CLASSES = (
    "com",
    "common stock",
    "cl a",
    "com new",
    "class a",
    "stock",
    "common",
    "com cl a",
    "com shs",
    (
        "sponsored adrsponsored adsadrequitycmncl bord shscl a com"
        "class a comcap stk cl acomm stkcl b newcap stk cl ccl a new"
        "foreign stockshs cl a"
    ),
)

FORMATION_ITEM_LABELS: Mapping[str, tuple[str, ...]] = {
    "financial-analysis-1": OPERATION_ORDER,
    "financial-analysis-3": (
        "q3_aum",
        "quarter_increase_rank",
        "q3_manager_rank",
    ),
    "financial-analysis-5": (
        "q3_aum",
        "quarter_increase_rank",
        "q3_manager_rank",
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMBERED_QUESTION = re.compile(r"(?m)^\s*(\d+)\.\s+")
_TOP_K = re.compile(
    r"\btop\s*[-:\s]?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS: Mapping[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


class FinancialSemanticError(RuntimeError):
    pass


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FinancialSemanticError(f"{label} is not a sha256 digest")
    return value


def _read_json(
    path: str | Path,
    *,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    if source.stat().st_size > maximum_bytes:
        raise FinancialSemanticError("JSON input exceeds its byte bound")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinancialSemanticError("JSON input is unreadable") from error
    if not isinstance(value, dict):
        raise FinancialSemanticError("JSON input must be an object")
    return value


def _verify_self_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    declared = _require_sha256(payload.get(field), f"{label} {field}")
    body = dict(payload)
    del body[field]
    if _payload_hash(body) != declared:
        raise FinancialSemanticError(f"{label} self hash mismatch")
    return declared


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def split_numbered_question_blocks(instruction: str) -> tuple[str, ...]:
    raw = instruction.encode("utf-8")
    if not raw or len(raw) > MAXIMUM_INSTRUCTION_BYTES:
        raise FinancialSemanticError("instruction byte length is invalid")
    matches = list(_NUMBERED_QUESTION.finditer(instruction))
    if len(matches) not in OPERATIONS_BY_BLOCK_COUNT:
        raise FinancialSemanticError("instruction needs three or four questions")
    blocks: list[str] = []
    for index, match in enumerate(matches):
        if int(match.group(1)) != index + 1:
            raise FinancialSemanticError("question numbering is not contiguous")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(
            instruction
        )
        if index + 1 == len(matches):
            format_index = instruction.find("\n\nFormat your answer", match.end())
            if format_index >= 0:
                end = format_index
        block = re.sub(r"\s+", " ", instruction[match.end() : end]).strip()
        if not block:
            raise FinancialSemanticError("question block is empty")
        blocks.append(block)
    return tuple(blocks)


def _parse_top_k(block: str) -> int:
    match = _TOP_K.search(block)
    if match is None:
        raise FinancialSemanticError("rank operation lacks a bounded top-k scalar")
    raw = match.group(1).lower()
    value = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
    if value < 1 or value > 10:
        raise FinancialSemanticError("top-k scalar is outside the frozen range")
    return value


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _name_score(query: object, candidate: object) -> tuple[float, ...]:
    left = _normalize_name(query)
    right = _normalize_name(candidate)
    if not left or not right:
        return (0.0, 0.0, 0.0, -math.inf)
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    substring = 1.0 if left in right or right in left else 0.0
    jaccard = len(left_tokens & right_tokens) / max(
        1, len(left_tokens | right_tokens)
    )
    sequence = difflib.SequenceMatcher(None, left, right).ratio()
    return (substring, jaccard, sequence, -float(abs(len(left) - len(right))))


def build_qa_runtime_asset(
    *,
    snapshot_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    snapshot = Path(snapshot_root).expanduser().resolve(strict=True)
    required_names = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    )
    rows: list[dict[str, Any]] = []
    for relative in required_names:
        source = snapshot / relative
        if not source.is_file():
            raise FinancialSemanticError("QA snapshot is incomplete")
        rows.append(
            {
                "relative_path": relative,
                "sha256": _sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import torch
        import transformers
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    except ImportError as error:
        raise FinancialSemanticError("QA build runtime is missing") from error
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        str(snapshot), local_files_only=True
    )
    model.eval()
    cases = (
        {
            "question": "What full name follows 'AUM of'?",
            "context": "In Q3, what is the AUM of Renaissance Technologies?",
            "expected_answer": "Renaissance Technologies",
        },
        {
            "question": (
                "What company did the requested fund managers invest in?"
            ),
            "context": "List managers which invested in Palantir in Q3.",
            "expected_answer": "Palantir",
        },
    )
    canary_rows: list[dict[str, str]] = []
    logit_bytes = bytearray()
    for case in cases:
        answer, _, start, end = _qa_answer(
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
            question=case["question"],
            context=case["context"],
        )
        if answer != case["expected_answer"]:
            raise FinancialSemanticError("QA snapshot failed its build canary")
        logit_bytes.extend(start)
        logit_bytes.extend(end)
        canary_rows.append(dict(case))
    asset: dict[str, Any] = {
        "asset_version": FINANCIAL_QA_RUNTIME_ASSET_VERSION,
        "model_id": "distilbert-base-cased-distilled-squad",
        "snapshot_revision": snapshot.name,
        "runtime_required_files": rows,
        "runtime_required_file_count": len(rows),
        "runtime_required_file_set_hash": _payload_hash(rows),
        "runtime_required_size_bytes": sum(row["size_bytes"] for row in rows),
        "weights_sha256": next(
            row["sha256"]
            for row in rows
            if row["relative_path"] == "model.safetensors"
        ),
        "runtime_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "execution": {
            "device": "cpu",
            "local_files_only": True,
            "torch_num_threads": 1,
            "torch_deterministic_algorithms": True,
            "network_calls": 0,
        },
        "deterministic_canary": {
            "cases": canary_rows,
            "little_endian_float32_logits_sha256": hashlib.sha256(
                logit_bytes
            ).hexdigest(),
        },
    }
    asset["manifest_hash"] = _payload_hash(asset)
    _atomic_write_json(output_path, asset)
    return asset


def verify_qa_runtime_asset(
    runtime_asset: Mapping[str, Any],
    *,
    snapshot_root: str | Path,
) -> dict[str, Any]:
    if runtime_asset.get("asset_version") != FINANCIAL_QA_RUNTIME_ASSET_VERSION:
        raise FinancialSemanticError("QA runtime asset version mismatch")
    manifest_hash = _verify_self_hash(
        runtime_asset, field="manifest_hash", label="QA runtime asset"
    )
    snapshot = Path(snapshot_root).expanduser().resolve(strict=True)
    if snapshot.name != runtime_asset.get("snapshot_revision"):
        raise FinancialSemanticError("QA snapshot revision mismatch")
    rows = runtime_asset.get("runtime_required_files")
    if (
        not isinstance(rows, list)
        or len(rows) != runtime_asset.get("runtime_required_file_count")
        or _payload_hash(rows) != runtime_asset.get("runtime_required_file_set_hash")
    ):
        raise FinancialSemanticError("QA runtime file manifest is malformed")
    total = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise FinancialSemanticError("QA runtime file row is malformed")
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise FinancialSemanticError("QA runtime file path is unsafe")
        source = snapshot / relative
        size = row.get("size_bytes")
        if (
            not source.is_file()
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or source.stat().st_size != size
            or _sha256_file(source) != row.get("sha256")
        ):
            raise FinancialSemanticError("QA runtime file content drifted")
        total += size
    if total != runtime_asset.get("runtime_required_size_bytes"):
        raise FinancialSemanticError("QA runtime byte total mismatch")
    weights = [
        row for row in rows if row["relative_path"] == "model.safetensors"
    ]
    if len(weights) != 1 or weights[0]["sha256"] != runtime_asset.get(
        "weights_sha256"
    ):
        raise FinancialSemanticError("QA runtime weights binding mismatch")
    return {
        "qa_runtime_asset_manifest_hash": manifest_hash,
        "qa_runtime_required_file_set_hash": runtime_asset[
            "runtime_required_file_set_hash"
        ],
        "qa_snapshot_revision": runtime_asset["snapshot_revision"],
        "qa_weights_sha256": runtime_asset["weights_sha256"],
    }


def _qa_answer(
    *,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    question: str,
    context: str,
) -> tuple[str, float, bytes, bytes]:
    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
    )
    with torch_module.inference_mode():
        output = model(**encoded)
    sequence_ids = encoded.sequence_ids(0)
    context_indices = [
        index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1
    ]
    if not context_indices:
        raise FinancialSemanticError("QA tokenizer produced no context tokens")
    start_logits = output.start_logits[0].detach().cpu()
    end_logits = output.end_logits[0].detach().cpu()
    best: tuple[float, int, int] | None = None
    for start in context_indices:
        for end in range(start, min(start + 24, len(sequence_ids))):
            if sequence_ids[end] != 1:
                break
            score = float(start_logits[start] + end_logits[end])
            candidate = (score, -start, -end)
            if best is None or candidate > (best[0], -best[1], -best[2]):
                best = (score, start, end)
    if best is None:
        raise FinancialSemanticError("QA runtime produced no bounded span")
    answer = tokenizer.decode(
        encoded["input_ids"][0, best[1] : best[2] + 1],
        skip_special_tokens=True,
    ).strip(" \t\r\n.,;:!?\"'")
    if not answer or len(answer) > MAXIMUM_ENTITY_CHARACTERS:
        raise FinancialSemanticError("QA entity span is invalid")
    import numpy as np

    start_bytes = np.asarray(start_logits.numpy(), dtype="<f4").tobytes(order="C")
    end_bytes = np.asarray(end_logits.numpy(), dtype="<f4").tobytes(order="C")
    return answer, best[0], start_bytes, end_bytes


class OfflineFinancialQA:
    def __init__(
        self,
        *,
        runtime_asset_path: str | Path,
        snapshot_root: str | Path,
    ) -> None:
        runtime_asset = _read_json(runtime_asset_path)
        self.runtime_receipt = verify_qa_runtime_asset(
            runtime_asset, snapshot_root=snapshot_root
        )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            import torch
            import transformers
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        except ImportError as error:
            raise FinancialSemanticError("offline QA runtime is missing") from error
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        declared_versions = runtime_asset.get("runtime_versions")
        if declared_versions != {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }:
            raise FinancialSemanticError("QA runtime dependency version drifted")
        snapshot = Path(snapshot_root).expanduser().resolve(strict=True)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True
        )
        self._model = AutoModelForQuestionAnswering.from_pretrained(
            str(snapshot), local_files_only=True
        )
        self._model.eval()
        canary = runtime_asset.get("deterministic_canary")
        if not isinstance(canary, dict) or not isinstance(canary.get("cases"), list):
            raise FinancialSemanticError("QA deterministic canary is missing")
        logit_bytes = bytearray()
        for case in canary["cases"]:
            if not isinstance(case, dict):
                raise FinancialSemanticError("QA canary row is malformed")
            answer, _, start, end = self._answer(
                str(case.get("question") or ""),
                str(case.get("context") or ""),
            )
            if answer != case.get("expected_answer"):
                raise FinancialSemanticError("QA deterministic answer drifted")
            logit_bytes.extend(start)
            logit_bytes.extend(end)
        if hashlib.sha256(logit_bytes).hexdigest() != canary.get(
            "little_endian_float32_logits_sha256"
        ):
            raise FinancialSemanticError("QA deterministic logits drifted")

    def _answer(self, question: str, context: str) -> tuple[str, float, bytes, bytes]:
        return _qa_answer(
            tokenizer=self._tokenizer,
            model=self._model,
            torch_module=self._torch,
            question=question,
            context=context,
        )

    def __call__(self, question: str, context: str) -> tuple[str, float]:
        answer, score, _, _ = self._answer(question, context)
        return answer, score


def build_financial_semantic_asset(
    *,
    benchmark_root: str | Path,
    minilm_runtime_asset_path: str | Path,
    qa_runtime_asset_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    benchmark = Path(benchmark_root).expanduser().resolve(strict=True)
    minilm_runtime = _read_json(minilm_runtime_asset_path)
    minilm_hash = _verify_self_hash(
        minilm_runtime, field="manifest_hash", label="MiniLM runtime asset"
    )
    qa_runtime = _read_json(qa_runtime_asset_path)
    qa_hash = _verify_self_hash(
        qa_runtime, field="manifest_hash", label="QA runtime asset"
    )
    examples: list[dict[str, Any]] = []
    formation_sources: list[dict[str, Any]] = []
    for item_id, labels in FORMATION_ITEM_LABELS.items():
        item = benchmark / "tasks" / "financial-analysis" / item_id
        instruction = item / "instruction.md"
        solution = item / "solution" / "solve.sh"
        expected = item / "tests" / "expected_output.json"
        if not all(path.is_file() for path in (instruction, solution, expected)):
            raise FinancialSemanticError("formation item is incomplete")
        instruction_text = instruction.read_text(encoding="utf-8")
        blocks = split_numbered_question_blocks(instruction_text)
        if len(blocks) != len(labels):
            raise FinancialSemanticError("formation label count mismatch")
        formation_sources.append(
            {
                "item_id": item_id,
                "instruction_sha256": _sha256_file(instruction),
                "solution_sha256": _sha256_file(solution),
                "expected_output_sha256": _sha256_file(expected),
                "question_block_count": len(blocks),
            }
        )
        for question_index, (block, operation) in enumerate(
            zip(blocks, labels), start=1
        ):
            examples.append(
                {
                    "item_id": item_id,
                    "question_index": question_index,
                    "operation": operation,
                    "text": block,
                    "text_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
                }
            )
    source_path = Path(__file__).resolve(strict=True)
    configuration = {
        "operation_order": list(OPERATION_ORDER),
        "operations_by_block_count": {
            str(key): list(value)
            for key, value in sorted(OPERATIONS_BY_BLOCK_COUNT.items())
        },
        "operation_prototypes": dict(OPERATION_PROTOTYPES),
        "semantic_score": "0.85_max_consumed_train_cosine_plus_0.15_prototype_cosine",
        "structured_decoder": "maximum_weight_one_to_one_assignment",
        "qa_prompts": dict(QA_PROMPTS),
        "top_k_parser": "typed_numeric_or_one_to_ten_word_after_top",
        "manager_resolution": "latest_non_notice_fuzzy_name",
        "issuer_resolution": "fuzzy_name_then_q3_aggregate_value",
        "stock_classes": list(TRAIN_DEFINED_STOCK_CLASSES),
        "query_chunk_rows": QUERY_CHUNK_ROWS,
    }
    asset: dict[str, Any] = {
        "asset_version": FINANCIAL_SEMANTIC_ASSET_VERSION,
        "operator_version": FINANCIAL_SEMANTIC_OPERATOR_VERSION,
        "formation_policy": "consumed_train_supervised_semantic_structure_only",
        "formation_item_ids": list(FORMATION_ITEM_LABELS),
        "formation_sources": formation_sources,
        "formation_source_set_hash": _payload_hash(formation_sources),
        "train_examples": examples,
        "train_example_count": len(examples),
        "train_example_set_hash": _payload_hash(examples),
        "configuration": configuration,
        "configuration_hash": _payload_hash(configuration),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "minilm_runtime_asset_manifest_hash": minilm_hash,
        "qa_runtime_asset_manifest_hash": qa_hash,
        "operator_source_sha256": _sha256_file(source_path),
        "excluded_split_access": {
            "prior_validation_content": False,
            "fresh_validation_content": False,
            "residual_sealed_content": False,
        },
        "online_calls": 0,
        "prospective_measurement_performed": False,
        "raw_instruction_logged_by_operator": False,
    }
    asset["candidate_id"] = _payload_hash(
        {
            "operator_version": FINANCIAL_SEMANTIC_OPERATOR_VERSION,
            "formation_source_set_hash": asset["formation_source_set_hash"],
            "train_example_set_hash": asset["train_example_set_hash"],
            "configuration_hash": asset["configuration_hash"],
            "minilm_runtime_asset_manifest_hash": minilm_hash,
            "qa_runtime_asset_manifest_hash": qa_hash,
            "operator_source_sha256": asset["operator_source_sha256"],
        }
    )
    asset["manifest_hash"] = _payload_hash(asset)
    _atomic_write_json(output_path, asset)
    return asset


def load_financial_semantic_asset(
    path: str | Path,
    *,
    minilm_runtime_asset_path: str | Path | None = None,
    qa_runtime_asset_path: str | Path | None = None,
) -> dict[str, Any]:
    asset = _read_json(path)
    if asset.get("asset_version") != FINANCIAL_SEMANTIC_ASSET_VERSION:
        raise FinancialSemanticError("financial asset version mismatch")
    _verify_self_hash(asset, field="manifest_hash", label="financial asset")
    if asset.get("operator_version") != FINANCIAL_SEMANTIC_OPERATOR_VERSION:
        raise FinancialSemanticError("financial operator version mismatch")
    if asset.get("operator_source_sha256") != _sha256_file(Path(__file__).resolve()):
        raise FinancialSemanticError("financial operator source drifted")
    examples = asset.get("train_examples")
    if (
        not isinstance(examples, list)
        or len(examples) != asset.get("train_example_count")
        or _payload_hash(examples) != asset.get("train_example_set_hash")
    ):
        raise FinancialSemanticError("financial TRAIN examples are malformed")
    for row in examples:
        if (
            not isinstance(row, dict)
            or row.get("operation") not in OPERATION_ORDER
            or hashlib.sha256(str(row.get("text") or "").encode("utf-8")).hexdigest()
            != row.get("text_sha256")
        ):
            raise FinancialSemanticError("financial TRAIN example drifted")
    configuration = asset.get("configuration")
    if not isinstance(configuration, dict) or _payload_hash(
        configuration
    ) != asset.get("configuration_hash"):
        raise FinancialSemanticError("financial configuration drifted")
    expected_candidate = _payload_hash(
        {
            "operator_version": FINANCIAL_SEMANTIC_OPERATOR_VERSION,
            "formation_source_set_hash": asset.get("formation_source_set_hash"),
            "train_example_set_hash": asset.get("train_example_set_hash"),
            "configuration_hash": asset.get("configuration_hash"),
            "minilm_runtime_asset_manifest_hash": asset.get(
                "minilm_runtime_asset_manifest_hash"
            ),
            "qa_runtime_asset_manifest_hash": asset.get(
                "qa_runtime_asset_manifest_hash"
            ),
            "operator_source_sha256": asset.get("operator_source_sha256"),
        }
    )
    if asset.get("candidate_id") != expected_candidate:
        raise FinancialSemanticError("financial candidate identity mismatch")
    if minilm_runtime_asset_path is not None:
        minilm_runtime = _read_json(minilm_runtime_asset_path)
        minilm_hash = _verify_self_hash(
            minilm_runtime, field="manifest_hash", label="MiniLM runtime asset"
        )
        if minilm_hash != asset.get("minilm_runtime_asset_manifest_hash"):
            raise FinancialSemanticError("MiniLM runtime binding mismatch")
    if qa_runtime_asset_path is not None:
        qa_runtime = _read_json(qa_runtime_asset_path)
        qa_hash = _verify_self_hash(
            qa_runtime, field="manifest_hash", label="QA runtime asset"
        )
        if qa_hash != asset.get("qa_runtime_asset_manifest_hash"):
            raise FinancialSemanticError("QA runtime binding mismatch")
    return asset


def _semantic_assignment(
    *,
    block_embeddings: Any,
    example_embeddings: Any,
    prototype_embeddings: Any,
    examples: Sequence[Mapping[str, Any]],
    allowed_operations: Sequence[str],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    import numpy as np

    block_matrix = np.asarray(block_embeddings, dtype=np.float32)
    example_matrix = np.asarray(example_embeddings, dtype=np.float32)
    prototype_matrix = np.asarray(prototype_embeddings, dtype=np.float32)
    if block_matrix.shape != (len(allowed_operations), EMBEDDING_DIMENSION):
        raise FinancialSemanticError("question embedding matrix shape mismatch")
    if example_matrix.shape != (len(examples), EMBEDDING_DIMENSION):
        raise FinancialSemanticError("TRAIN embedding matrix shape mismatch")
    if prototype_matrix.shape != (len(OPERATION_ORDER), EMBEDDING_DIMENSION):
        raise FinancialSemanticError("prototype embedding matrix shape mismatch")
    if not all(
        np.isfinite(matrix).all()
        for matrix in (block_matrix, example_matrix, prototype_matrix)
    ):
        raise FinancialSemanticError("semantic embedding is not finite")
    scores: dict[tuple[int, str], float] = {}
    for block_index in range(len(allowed_operations)):
        for operation in allowed_operations:
            indices = [
                index
                for index, row in enumerate(examples)
                if row.get("operation") == operation
            ]
            if not indices:
                raise FinancialSemanticError("semantic operation lacks TRAIN support")
            train_score = max(
                float(block_matrix[block_index] @ example_matrix[index])
                for index in indices
            )
            prototype_index = OPERATION_ORDER.index(operation)
            prototype_score = float(
                block_matrix[block_index] @ prototype_matrix[prototype_index]
            )
            scores[(block_index, operation)] = (
                0.85 * train_score + 0.15 * prototype_score
            )
    best_assignment: tuple[str, ...] | None = None
    best_score = -math.inf
    for assignment in itertools.permutations(allowed_operations):
        score = sum(
            scores[(index, operation)]
            for index, operation in enumerate(assignment)
        )
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and (best_assignment is None or assignment < best_assignment)
        ):
            best_assignment = assignment
            best_score = score
    if best_assignment is None:
        raise FinancialSemanticError("semantic assignment is empty")
    return best_assignment, tuple(
        scores[(index, operation)]
        for index, operation in enumerate(best_assignment)
    )


def build_financial_semantic_plan(
    *,
    instruction: str,
    asset: Mapping[str, Any],
    encoder: Callable[[Sequence[str]], Any],
    qa: Callable[[str, str], tuple[str, float]],
    minilm_runtime_receipt: Mapping[str, Any] | None = None,
    qa_runtime_receipt: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if asset.get("asset_version") != FINANCIAL_SEMANTIC_ASSET_VERSION:
        raise FinancialSemanticError("financial asset was not validated")
    blocks = split_numbered_question_blocks(instruction)
    allowed = OPERATIONS_BY_BLOCK_COUNT[len(blocks)]
    examples = asset["train_examples"]
    example_texts = [str(row["text"]) for row in examples]
    prototype_texts = [OPERATION_PROTOTYPES[row] for row in OPERATION_ORDER]
    block_embeddings = encoder(blocks)
    example_embeddings = encoder(example_texts)
    prototype_embeddings = encoder(prototype_texts)
    assignment, semantic_scores = _semantic_assignment(
        block_embeddings=block_embeddings,
        example_embeddings=example_embeddings,
        prototype_embeddings=prototype_embeddings,
        examples=examples,
        allowed_operations=allowed,
    )
    operations: list[dict[str, Any]] = []
    for index, (block, operation, semantic_score) in enumerate(
        zip(blocks, assignment, semantic_scores), start=1
    ):
        entity, qa_score = qa(QA_PROMPTS[operation], block)
        entity = re.sub(r"\s+", " ", entity).strip(" \t\r\n.,;:!?\"'")
        if not entity or len(entity) > MAXIMUM_ENTITY_CHARACTERS:
            raise FinancialSemanticError("semantic entity is invalid")
        operations.append(
            {
                "question_index": index,
                "answer_key": f"q{index}_answer",
                "operation": operation,
                "entity": entity,
                "entity_sha256": hashlib.sha256(entity.encode("utf-8")).hexdigest(),
                "top_k": (
                    _parse_top_k(block)
                    if operation
                    in {"quarter_increase_rank", "q3_manager_rank"}
                    else None
                ),
                "question_block_sha256": hashlib.sha256(
                    block.encode("utf-8")
                ).hexdigest(),
                "semantic_score": round(float(semantic_score), 9),
                "qa_score": round(float(qa_score), 9),
                "entity_resolution": "direct_qa_span",
            }
        )
    aum = next(
        (row for row in operations if row["operation"] == "q3_aum"), None
    )
    count = next(
        (row for row in operations if row["operation"] == "q3_stock_count"),
        None,
    )
    if aum is not None and count is not None:
        left = set(_normalize_name(aum["entity"]).split())
        right = set(_normalize_name(count["entity"]).split())
        if left and right and (left <= right or right <= left):
            count["entity"] = aum["entity"]
            count["entity_sha256"] = aum["entity_sha256"]
            count["entity_resolution"] = "same_item_alias_to_q3_aum"
    plan: dict[str, Any] = {
        "plan_version": FINANCIAL_SEMANTIC_PLAN_VERSION,
        "operator_version": FINANCIAL_SEMANTIC_OPERATOR_VERSION,
        "candidate_id": asset["candidate_id"],
        "candidate_manifest_hash": asset["manifest_hash"],
        "instruction_sha256": hashlib.sha256(
            instruction.encode("utf-8")
        ).hexdigest(),
        "question_block_count": len(blocks),
        "operations": operations,
        "operation_set_hash": _payload_hash(operations),
        "minilm_runtime_asset_manifest_hash": asset[
            "minilm_runtime_asset_manifest_hash"
        ],
        "qa_runtime_asset_manifest_hash": asset[
            "qa_runtime_asset_manifest_hash"
        ],
        "operator_source_sha256": asset["operator_source_sha256"],
        "online_calls": 0,
        "raw_instruction_persisted": False,
    }
    plan["plan_hash"] = _payload_hash(plan)
    receipt: dict[str, Any] = {
        "receipt_version": "financial_semantic_extraction_receipt_v1",
        "candidate_id": asset["candidate_id"],
        "candidate_manifest_hash": asset["manifest_hash"],
        "plan_hash": plan["plan_hash"],
        "instruction_sha256": plan["instruction_sha256"],
        "question_block_hashes": [
            row["question_block_sha256"] for row in operations
        ],
        "semantic_assignment": [row["operation"] for row in operations],
        "entity_hashes": [row["entity_sha256"] for row in operations],
        "minilm_runtime_receipt": dict(minilm_runtime_receipt or {}),
        "qa_runtime_receipt": dict(qa_runtime_receipt or {}),
        "operator_created_raw_instruction_artifact": False,
        "online_calls": 0,
    }
    receipt["receipt_hash"] = _payload_hash(receipt)
    return plan, receipt


def validate_financial_semantic_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if plan.get("plan_version") != FINANCIAL_SEMANTIC_PLAN_VERSION:
        raise FinancialSemanticError("financial plan version mismatch")
    _verify_self_hash(plan, field="plan_hash", label="financial plan")
    for field in (
        "candidate_id",
        "candidate_manifest_hash",
        "instruction_sha256",
        "operation_set_hash",
        "minilm_runtime_asset_manifest_hash",
        "qa_runtime_asset_manifest_hash",
        "operator_source_sha256",
    ):
        _require_sha256(plan.get(field), f"financial plan {field}")
    if plan.get("operator_source_sha256") != _sha256_file(
        Path(__file__).resolve(strict=True)
    ):
        raise FinancialSemanticError("financial plan operator source mismatch")
    if plan.get("online_calls") != 0 or plan.get(
        "raw_instruction_persisted"
    ) is not False:
        raise FinancialSemanticError("financial plan execution boundary drifted")
    count = plan.get("question_block_count")
    if count not in OPERATIONS_BY_BLOCK_COUNT:
        raise FinancialSemanticError("financial plan block count is invalid")
    operations = plan.get("operations")
    if (
        not isinstance(operations, list)
        or len(operations) != count
        or _payload_hash(operations) != plan.get("operation_set_hash")
    ):
        raise FinancialSemanticError("financial plan operations are malformed")
    seen_operations: set[str] = set()
    for index, row in enumerate(operations, start=1):
        if not isinstance(row, dict):
            raise FinancialSemanticError("financial operation row is malformed")
        operation = row.get("operation")
        entity = row.get("entity")
        if (
            operation not in OPERATIONS_BY_BLOCK_COUNT[count]
            or operation in seen_operations
            or row.get("question_index") != index
            or row.get("answer_key") != f"q{index}_answer"
            or not isinstance(entity, str)
            or not entity
            or len(entity) > MAXIMUM_ENTITY_CHARACTERS
            or hashlib.sha256(entity.encode("utf-8")).hexdigest()
            != row.get("entity_sha256")
        ):
            raise FinancialSemanticError("financial operation row is invalid")
        _require_sha256(
            row.get("question_block_sha256"),
            "financial operation question block hash",
        )
        for score_field in ("semantic_score", "qa_score"):
            score = row.get(score_field)
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise FinancialSemanticError(
                    "financial operation score is invalid"
                )
        seen_operations.add(str(operation))
        if operation in {"quarter_increase_rank", "q3_manager_rank"}:
            top_k = row.get("top_k")
            if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
                raise FinancialSemanticError("financial rank scalar is invalid")
        elif row.get("top_k") is not None:
            raise FinancialSemanticError("financial scalar attached to non-rank op")
    if seen_operations != set(OPERATIONS_BY_BLOCK_COUNT[count]):
        raise FinancialSemanticError("financial operation set is incomplete")
    return dict(plan)


def _latest_non_notice_cover(cover: Any) -> Any:
    import pandas as pd

    dates = pd.to_datetime(
        cover["REPORTCALENDARORQUARTER"],
        format="%d-%b-%Y",
        errors="coerce",
    )
    latest = dates.max()
    selected = cover[dates == latest]
    selected = selected[
        ~selected["REPORTTYPE"].str.contains("NOTICE", case=False, na=False)
    ]
    if selected.empty:
        raise FinancialSemanticError("cover page has no latest non-notice filing")
    return selected


def _best_manager(cover: Any, entity: str) -> tuple[str, str, tuple[float, ...]]:
    selected = _latest_non_notice_cover(cover)
    best_index = max(
        selected.index,
        key=lambda index: (
            _name_score(entity, selected.at[index, "FILINGMANAGER_NAME"]),
            _normalize_name(selected.at[index, "FILINGMANAGER_NAME"]),
        ),
    )
    return (
        str(selected.at[best_index, "ACCESSION_NUMBER"]),
        str(selected.at[best_index, "FILINGMANAGER_NAME"]),
        _name_score(entity, selected.at[best_index, "FILINGMANAGER_NAME"]),
    )


def _read_cover(root: Path) -> Any:
    import pandas as pd

    path = root / "COVERPAGE.tsv"
    if not path.is_file():
        raise FinancialSemanticError("financial cover page is missing")
    return pd.read_csv(path, sep="\t", dtype=str)


def execute_financial_semantic_plan(
    *,
    plan: Mapping[str, Any],
    q2_root: str | Path,
    q3_root: str | Path,
    output_path: str | Path,
    receipt_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validated = validate_financial_semantic_plan(plan)
    try:
        import pandas as pd
    except ImportError as error:
        raise FinancialSemanticError("financial query runtime lacks pandas") from error
    q2 = Path(q2_root).expanduser().resolve(strict=True)
    q3 = Path(q3_root).expanduser().resolve(strict=True)
    q2_info = q2 / "INFOTABLE.tsv"
    q3_info = q3 / "INFOTABLE.tsv"
    if not q2_info.is_file() or not q3_info.is_file():
        raise FinancialSemanticError("financial holdings table is missing")
    cover2 = _read_cover(q2)
    cover3 = _read_cover(q3)
    operations = validated["operations"]
    by_type = {row["operation"]: row for row in operations}
    manager_receipts: dict[str, dict[str, Any]] = {}
    for operation in ("q3_aum", "q3_stock_count", "quarter_increase_rank"):
        row = by_type.get(operation)
        if row is None:
            continue
        accession3, manager3, score3 = _best_manager(cover3, row["entity"])
        receipt: dict[str, Any] = {
            "q3_accession_sha256": hashlib.sha256(accession3.encode()).hexdigest(),
            "q3_manager_sha256": hashlib.sha256(manager3.encode()).hexdigest(),
            "q3_name_score": [round(value, 9) for value in score3],
        }
        if operation == "quarter_increase_rank":
            accession2, manager2, score2 = _best_manager(cover2, row["entity"])
            receipt.update(
                {
                    "q2_accession_sha256": hashlib.sha256(
                        accession2.encode()
                    ).hexdigest(),
                    "q2_manager_sha256": hashlib.sha256(
                        manager2.encode()
                    ).hexdigest(),
                    "q2_name_score": [round(value, 9) for value in score2],
                    "q2_accession": accession2,
                }
            )
        receipt["q3_accession"] = accession3
        manager_receipts[operation] = receipt

    aum_total = 0.0
    stock_count = 0
    q3_holdings: dict[str, float] = {}
    issuer_pairs: dict[tuple[str, str], float] = {}
    use_columns = [
        "ACCESSION_NUMBER",
        "NAMEOFISSUER",
        "TITLEOFCLASS",
        "CUSIP",
        "VALUE",
    ]
    dtype = {
        "ACCESSION_NUMBER": str,
        "NAMEOFISSUER": str,
        "TITLEOFCLASS": str,
        "CUSIP": str,
        "VALUE": float,
    }
    stock_classes = set(TRAIN_DEFINED_STOCK_CLASSES)
    for chunk in pd.read_csv(
        q3_info,
        sep="\t",
        usecols=use_columns,
        dtype=dtype,
        chunksize=QUERY_CHUNK_ROWS,
    ):
        if "q3_aum" in manager_receipts:
            accession = manager_receipts["q3_aum"]["q3_accession"]
            aum_total += float(
                chunk.loc[chunk["ACCESSION_NUMBER"] == accession, "VALUE"].sum()
            )
        if "q3_stock_count" in manager_receipts:
            accession = manager_receipts["q3_stock_count"]["q3_accession"]
            selected = chunk[chunk["ACCESSION_NUMBER"] == accession]
            stock_count += int(
                selected["TITLEOFCLASS"].str.casefold().isin(stock_classes).sum()
            )
        if "quarter_increase_rank" in manager_receipts:
            accession = manager_receipts["quarter_increase_rank"]["q3_accession"]
            selected = chunk[
                (chunk["ACCESSION_NUMBER"] == accession)
                & chunk["TITLEOFCLASS"].str.casefold().isin(stock_classes)
            ]
            grouped = selected.groupby("CUSIP")["VALUE"].sum()
            for cusip, value in grouped.items():
                q3_holdings[str(cusip)] = q3_holdings.get(str(cusip), 0.0) + float(
                    value
                )
        if "q3_manager_rank" in by_type:
            pairs = chunk[["NAMEOFISSUER", "CUSIP", "VALUE"]].dropna()
            grouped = pairs.groupby(["NAMEOFISSUER", "CUSIP"])["VALUE"].sum()
            for (issuer, cusip), value in grouped.items():
                key = (str(issuer), str(cusip))
                issuer_pairs[key] = issuer_pairs.get(key, 0.0) + float(value)

    q2_holdings: dict[str, float] = {}
    if "quarter_increase_rank" in manager_receipts:
        accession = manager_receipts["quarter_increase_rank"]["q2_accession"]
        for chunk in pd.read_csv(
            q2_info,
            sep="\t",
            usecols=["ACCESSION_NUMBER", "TITLEOFCLASS", "CUSIP", "VALUE"],
            dtype={
                "ACCESSION_NUMBER": str,
                "TITLEOFCLASS": str,
                "CUSIP": str,
                "VALUE": float,
            },
            chunksize=QUERY_CHUNK_ROWS,
        ):
            selected = chunk[
                (chunk["ACCESSION_NUMBER"] == accession)
                & chunk["TITLEOFCLASS"].str.casefold().isin(stock_classes)
            ]
            grouped = selected.groupby("CUSIP")["VALUE"].sum()
            for cusip, value in grouped.items():
                q2_holdings[str(cusip)] = q2_holdings.get(str(cusip), 0.0) + float(
                    value
                )

    selected_company_cusip: str | None = None
    company_receipt: dict[str, Any] | None = None
    manager_values: dict[str, float] = {}
    if "q3_manager_rank" in by_type:
        target = str(by_type["q3_manager_rank"]["entity"])
        if not issuer_pairs:
            raise FinancialSemanticError("financial issuer inventory is empty")
        best_pair = max(
            issuer_pairs,
            key=lambda pair: (
                _name_score(target, pair[0]),
                issuer_pairs[pair],
                pair[1],
            ),
        )
        selected_company_cusip = best_pair[1]
        company_receipt = {
            "issuer_sha256": hashlib.sha256(best_pair[0].encode()).hexdigest(),
            "cusip_sha256": hashlib.sha256(best_pair[1].encode()).hexdigest(),
            "name_score": [
                round(value, 9) for value in _name_score(target, best_pair[0])
            ],
        }
        for chunk in pd.read_csv(
            q3_info,
            sep="\t",
            usecols=["ACCESSION_NUMBER", "CUSIP", "VALUE"],
            dtype={"ACCESSION_NUMBER": str, "CUSIP": str, "VALUE": float},
            chunksize=QUERY_CHUNK_ROWS,
        ):
            selected = chunk[chunk["CUSIP"] == selected_company_cusip]
            grouped = selected.groupby("ACCESSION_NUMBER")["VALUE"].sum()
            for accession, value in grouped.items():
                manager_values[str(accession)] = manager_values.get(
                    str(accession), 0.0
                ) + float(value)

    name_by_accession = dict(
        zip(cover3["ACCESSION_NUMBER"], cover3["FILINGMANAGER_NAME"])
    )
    answers: dict[str, Any] = {}
    for row in operations:
        operation = row["operation"]
        key = row["answer_key"]
        if operation == "q3_aum":
            answers[key] = float(aum_total)
        elif operation == "q3_stock_count":
            answers[key] = int(stock_count)
        elif operation == "quarter_increase_rank":
            cusips = set(q2_holdings) | set(q3_holdings)
            deltas = {
                cusip: q3_holdings.get(cusip, 0.0) - q2_holdings.get(cusip, 0.0)
                for cusip in cusips
            }
            answers[key] = [
                cusip
                for cusip, value in sorted(
                    deltas.items(), key=lambda item: (-item[1], item[0])
                )
                if value > 0
            ][: row["top_k"]]
        elif operation == "q3_manager_rank":
            ranked = sorted(
                manager_values.items(), key=lambda item: (-item[1], item[0])
            )[: row["top_k"]]
            try:
                answers[key] = [str(name_by_accession[accession]) for accession, _ in ranked]
            except KeyError as error:
                raise FinancialSemanticError(
                    "ranked manager is absent from cover page"
                ) from error
        else:  # pragma: no cover - validated above
            raise FinancialSemanticError("unknown financial operation")

    _atomic_write_json(output_path, answers)
    output = Path(output_path).expanduser().resolve(strict=True)
    safe_manager_receipts = {
        operation: {
            key: value
            for key, value in receipt.items()
            if key not in {"q2_accession", "q3_accession"}
        }
        for operation, receipt in manager_receipts.items()
    }
    receipt: dict[str, Any] = {
        "receipt_version": FINANCIAL_QUERY_RECEIPT_VERSION,
        "plan_hash": validated["plan_hash"],
        "candidate_id": validated["candidate_id"],
        "candidate_manifest_hash": validated["candidate_manifest_hash"],
        "minilm_runtime_asset_manifest_hash": validated[
            "minilm_runtime_asset_manifest_hash"
        ],
        "qa_runtime_asset_manifest_hash": validated[
            "qa_runtime_asset_manifest_hash"
        ],
        "operator_source_sha256": validated["operator_source_sha256"],
        "manager_resolution_receipts": safe_manager_receipts,
        "company_resolution_receipt": company_receipt,
        "output_sha256": _sha256_file(output),
        "answer_key_set_hash": _payload_hash(sorted(answers)),
        "query_chunk_rows": QUERY_CHUNK_ROWS,
        "network_calls": 0,
        "verifier_content_accessed": False,
        "raw_instruction_persisted": False,
    }
    receipt["receipt_hash"] = _payload_hash(receipt)
    if receipt_path is not None:
        _atomic_write_json(receipt_path, receipt)
    return answers, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    qa_asset = subparsers.add_parser("build-qa-runtime-asset")
    qa_asset.add_argument("--snapshot-root", type=Path, required=True)
    qa_asset.add_argument("--output", type=Path, required=True)
    asset = subparsers.add_parser("build-asset")
    asset.add_argument("--benchmark-root", type=Path, required=True)
    asset.add_argument("--minilm-runtime-asset", type=Path, required=True)
    asset.add_argument("--qa-runtime-asset", type=Path, required=True)
    asset.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--q2-root", type=Path, required=True)
    execute.add_argument("--q3-root", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--receipt-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-qa-runtime-asset":
        payload = build_qa_runtime_asset(
            snapshot_root=args.snapshot_root,
            output_path=args.output,
        )
    elif args.command == "build-asset":
        payload = build_financial_semantic_asset(
            benchmark_root=args.benchmark_root,
            minilm_runtime_asset_path=args.minilm_runtime_asset,
            qa_runtime_asset_path=args.qa_runtime_asset,
            output_path=args.output,
        )
    elif args.command == "execute":
        _, payload = execute_financial_semantic_plan(
            plan=_read_json(args.plan),
            q2_root=args.q2_root,
            q3_root=args.q3_root,
            output_path=args.output,
            receipt_path=args.receipt_output,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
