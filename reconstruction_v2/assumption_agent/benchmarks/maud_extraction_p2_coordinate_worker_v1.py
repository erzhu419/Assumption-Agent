"""Bounded local GPU coordinate workers for MAUD extraction P2.

The two roles consume the same private, label-free contract archive.  MiniLM
embeds each contract's passages once and then its 22 questions.  The
cross-encoder scores every question/passage pair once.  Outputs contain opaque
work IDs and quantized coordinates only; source text and gold never leave the
private input archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import stat
from typing import Callable, Mapping, Sequence


VERSION = "maud_extraction_p2_coordinate_worker_v1"
STUDY_ID = "MAUD_EXTRACTION_P2_CGROUP_BOUNDED_EVALUATOR_V1"
ROLE_MINILM = "MINILM"
ROLE_CROSS_ENCODER = "CROSS_ENCODER"
ROLES = (ROLE_MINILM, ROLE_CROSS_ENCODER)
SCALE = 1_000_000
MINILM_BATCH_SIZE = 64
CROSS_ENCODER_BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 512
# ``text`` is the frozen ASCII JSON serialization of a raw passage, not the
# raw <=1,400-code-point substring itself.  ``ensure_ascii=True`` can expand a
# non-BMP code point to a 12-character surrogate pair, so the IPC bound must
# cover that serialization (and match the isolated official worker).
MAX_SERIALIZED_PASSAGE_CHARACTERS = 20_000
NATIVE_THREAD_ENVIRONMENT_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class MaudCoordinateError(RuntimeError):
    """A private coordinate input, model, output, or runtime drifted."""


def _require_native_thread_environment() -> None:
    if any(
        os.environ.get(key) != "1"
        for key in NATIVE_THREAD_ENVIRONMENT_KEYS
    ):
        raise MaudCoordinateError(
            "native BLAS/OpenMP thread environment drifted"
        )


def _configure_torch_threads(torch: object) -> None:
    torch.set_num_threads(1)  # type: ignore[attr-defined]
    torch.set_num_interop_threads(1)  # type: ignore[attr-defined]
    if (
        torch.get_num_threads() != 1  # type: ignore[attr-defined]
        or torch.get_num_interop_threads() != 1  # type: ignore[attr-defined]
    ):
        raise MaudCoordinateError("torch thread configuration drifted")


def canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MaudCoordinateError("coordinate value is not canonical JSON") from exc


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _opaque(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise MaudCoordinateError(f"{field} is not an opaque SHA-256 identity")
    return value


def _text(value: object, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise MaudCoordinateError(f"{field} is invalid")
    return value


def validate_private_input(value: object) -> tuple[dict[str, object], ...]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"contracts", "schema", "study_id"}
        or value.get("schema") != f"{VERSION}_private_input_v1"
        or value.get("study_id") != STUDY_ID
    ):
        raise MaudCoordinateError("private coordinate envelope drifted")
    raw_contracts = value.get("contracts")
    if (
        isinstance(raw_contracts, (str, bytes))
        or not isinstance(raw_contracts, Sequence)
        or not raw_contracts
    ):
        raise MaudCoordinateError("private coordinate contracts drifted")
    contracts: list[dict[str, object]] = []
    seen_contracts: set[str] = set()
    seen_work: set[str] = set()
    for raw_contract in raw_contracts:
        if (
            not isinstance(raw_contract, Mapping)
            or set(raw_contract) != {"contract_id", "passages", "queries"}
        ):
            raise MaudCoordinateError("private contract shape drifted")
        contract_id = _opaque(raw_contract.get("contract_id"), "contract_id")
        if contract_id in seen_contracts:
            raise MaudCoordinateError("duplicate private contract")
        seen_contracts.add(contract_id)
        raw_passages = raw_contract.get("passages")
        if (
            isinstance(raw_passages, (str, bytes))
            or not isinstance(raw_passages, Sequence)
            or len(raw_passages) < 5
            or len(raw_passages) > 4096
        ):
            raise MaudCoordinateError("private passages drifted")
        passages: list[dict[str, object]] = []
        for ordinal, raw_passage in enumerate(raw_passages):
            if (
                not isinstance(raw_passage, Mapping)
                or set(raw_passage) != {"end", "ordinal", "start", "text"}
                or raw_passage.get("ordinal") != ordinal
            ):
                raise MaudCoordinateError("private passage shape drifted")
            start = raw_passage.get("start")
            end = raw_passage.get("end")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or end <= start
            ):
                raise MaudCoordinateError("private passage offsets drifted")
            passages.append(
                {
                    "ordinal": ordinal,
                    "start": start,
                    "end": end,
                    "text": _text(
                        raw_passage.get("text"),
                        "passage text",
                        MAX_SERIALIZED_PASSAGE_CHARACTERS,
                    ),
                }
            )
        raw_queries = raw_contract.get("queries")
        if (
            isinstance(raw_queries, (str, bytes))
            or not isinstance(raw_queries, Sequence)
            or len(raw_queries) != 22
        ):
            raise MaudCoordinateError("private query set is not the frozen 22")
        queries: list[dict[str, object]] = []
        for raw_query in raw_queries:
            if (
                not isinstance(raw_query, Mapping)
                or set(raw_query) != {"family", "question", "work_id"}
            ):
                raise MaudCoordinateError("private query shape drifted")
            work_id = _opaque(raw_query.get("work_id"), "work_id")
            if work_id in seen_work:
                raise MaudCoordinateError("duplicate private work_id")
            seen_work.add(work_id)
            family = raw_query.get("family")
            if family not in {
                "definition_reference",
                "condition_obligation",
                "protection_exception_remedy",
            }:
                raise MaudCoordinateError("private query family drifted")
            queries.append(
                {
                    "work_id": work_id,
                    "family": family,
                    "question": _text(raw_query.get("question"), "question", 1_000),
                }
            )
        contracts.append(
            {
                "contract_id": contract_id,
                "passages": passages,
                "queries": queries,
            }
        )
    return tuple(contracts)


def private_input_payload(contracts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    body = {
        "schema": f"{VERSION}_private_input_v1",
        "study_id": STUDY_ID,
        "contracts": list(contracts),
    }
    checked = validate_private_input(body)
    return {
        "schema": body["schema"],
        "study_id": STUDY_ID,
        "contracts": list(checked),
    }


def _quantize_unit_interval(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MaudCoordinateError("coordinate is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise MaudCoordinateError("coordinate escaped [0,1]")
    return int(round(number * SCALE))


def _matrix(value: object, rows: int) -> list[list[float]]:
    try:
        outer = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise MaudCoordinateError("embedding output is not iterable") from exc
    if len(outer) != rows:
        raise MaudCoordinateError("embedding row count drifted")
    matrix: list[list[float]] = []
    width: int | None = None
    for row in outer:
        try:
            values = [float(item) for item in row]
        except (TypeError, ValueError) as exc:
            raise MaudCoordinateError("embedding row drifted") from exc
        if width is None:
            width = len(values)
        if not width or len(values) != width or not all(map(math.isfinite, values)):
            raise MaudCoordinateError("embedding shape or finiteness drifted")
        norm = math.sqrt(sum(item * item for item in values))
        if abs(norm - 1.0) > 2e-5:
            raise MaudCoordinateError("embedding is not normalized")
        matrix.append(values)
    return matrix


def compute_minilm(
    contracts: Sequence[Mapping[str, object]],
    encoder: Callable[[Sequence[str]], object],
) -> dict[str, list[dict[str, object]]]:
    """Compute cosine coordinates with one passage encoding per contract."""

    checked = validate_private_input(private_input_payload(contracts))
    rows: list[dict[str, object]] = []
    contract_pairwise: list[dict[str, object]] = []
    for contract in checked:
        passages = contract["passages"]  # type: ignore[assignment]
        queries = contract["queries"]  # type: ignore[assignment]
        texts = [row["text"] for row in passages] + [row["question"] for row in queries]  # type: ignore[index]
        matrix = _matrix(encoder(texts), len(texts))
        passage_vectors = matrix[: len(passages)]
        query_vectors = matrix[len(passages) :]
        pairwise = []
        for left in passage_vectors:
            pairwise_row = []
            for right in passage_vectors:
                cosine = sum(a * b for a, b in zip(left, right))
                cosine = max(-1.0, min(1.0, cosine))
                pairwise_row.append(
                    _quantize_unit_interval((cosine + 1.0) / 2.0)
                )
            pairwise.append(pairwise_row)
        contract_pairwise.append(
            {
                "contract_id": contract["contract_id"],
                "pairwise_scores": pairwise,
            }
        )
        for query, query_vector in zip(queries, query_vectors):  # type: ignore[arg-type]
            scores = []
            for passage_vector in passage_vectors:
                cosine = sum(a * b for a, b in zip(query_vector, passage_vector))
                cosine = max(-1.0, min(1.0, cosine))
                scores.append(_quantize_unit_interval((cosine + 1.0) / 2.0))
            rows.append({"work_id": query["work_id"], "scores": scores})
    return {"rows": rows, "contract_pairwise": contract_pairwise}


def compute_cross_encoder(
    contracts: Sequence[Mapping[str, object]],
    scorer: Callable[[Sequence[tuple[str, str]]], object],
    *,
    batch_size: int = CROSS_ENCODER_BATCH_SIZE,
) -> list[dict[str, object]]:
    """Compute sigmoid coordinates for every frozen query/passage pair once."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise MaudCoordinateError("cross-encoder batch size drifted")
    checked = validate_private_input(private_input_payload(contracts))
    rows: list[dict[str, object]] = []
    for contract in checked:
        passages = contract["passages"]  # type: ignore[assignment]
        for query in contract["queries"]:  # type: ignore[index]
            pairs = [(query["question"], passage["text"]) for passage in passages]  # type: ignore[index]
            logits: list[float] = []
            for start in range(0, len(pairs), batch_size):
                try:
                    values = list(scorer(pairs[start : start + batch_size]))  # type: ignore[arg-type]
                except TypeError as exc:
                    raise MaudCoordinateError("cross-encoder output drifted") from exc
                if len(values) != len(pairs[start : start + batch_size]):
                    raise MaudCoordinateError("cross-encoder score count drifted")
                for value in values:
                    if isinstance(value, bool) or not isinstance(value, Real):
                        raise MaudCoordinateError("cross-encoder logit drifted")
                    number = float(value)
                    if not math.isfinite(number):
                        raise MaudCoordinateError("cross-encoder logit is nonfinite")
                    logits.append(number)
            scores = [
                _quantize_unit_interval(
                    1.0
                    if value >= 40.0
                    else 0.0
                    if value <= -40.0
                    else 1.0 / (1.0 + math.exp(-value))
                )
                for value in logits
            ]
            rows.append({"work_id": query["work_id"], "scores": scores})
    return rows


def coordinate_output(
    *,
    role: str,
    rows: Sequence[Mapping[str, object]],
    input_sha256: str,
    model_tree_sha256: str,
    contract_pairwise: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    if role not in ROLES:
        raise MaudCoordinateError("coordinate role drifted")
    _opaque(input_sha256, "input_sha256")
    _opaque(model_tree_sha256, "model_tree_sha256")
    checked_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != {"scores", "work_id"}:
            raise MaudCoordinateError("coordinate row shape drifted")
        work_id = _opaque(raw.get("work_id"), "work_id")
        scores = raw.get("scores")
        if (
            work_id in seen
            or isinstance(scores, (str, bytes))
            or not isinstance(scores, Sequence)
            or len(scores) < 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > SCALE
                for value in scores
            )
        ):
            raise MaudCoordinateError("coordinate row values drifted")
        seen.add(work_id)
        checked_rows.append({"work_id": work_id, "scores": list(scores)})
    checked_pairwise: list[dict[str, object]] = []
    seen_contracts: set[str] = set()
    for raw in contract_pairwise:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"contract_id", "pairwise_scores"}
        ):
            raise MaudCoordinateError("pairwise contract row drifted")
        contract_id = _opaque(raw.get("contract_id"), "contract_id")
        matrix = raw.get("pairwise_scores")
        if (
            contract_id in seen_contracts
            or isinstance(matrix, (str, bytes))
            or not isinstance(matrix, Sequence)
            or len(matrix) < 5
            or any(
                isinstance(row, (str, bytes))
                or not isinstance(row, Sequence)
                or len(row) != len(matrix)
                for row in matrix
            )
        ):
            raise MaudCoordinateError("pairwise matrix shape drifted")
        copied = [list(row) for row in matrix]  # type: ignore[arg-type]
        for left in range(len(copied)):
            for right in range(len(copied)):
                value = copied[left][right]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= SCALE
                    or value != copied[right][left]
                    or (left == right and value != SCALE)
                ):
                    raise MaudCoordinateError("pairwise matrix values drifted")
        seen_contracts.add(contract_id)
        checked_pairwise.append(
            {"contract_id": contract_id, "pairwise_scores": copied}
        )
    if (role == ROLE_MINILM) != bool(checked_pairwise):
        raise MaudCoordinateError("pairwise output does not match coordinate role")
    body = {
        "schema": f"{VERSION}_private_output_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "role": role,
        "input_sha256": input_sha256,
        "model_tree_sha256": model_tree_sha256,
        "row_count": len(checked_rows),
        "rows": checked_rows,
        "contract_pairwise": checked_pairwise,
        "retry_replay_resample_count": 0,
        "dynamic_batch_resize_count": 0,
        "network_or_API_call_count": 0,
    }
    return {**body, "self_sha256": semantic_sha256(body)}


def _short_model_root(value: str, expected: str) -> Path:
    path = Path(value)
    if path.parts != (expected,) or not path.is_symlink():
        raise MaudCoordinateError("model argv is not the frozen cwd-local alias")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise MaudCoordinateError("model alias target is not a directory")
    return path


def _load_input(path: Path) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    if path.is_symlink() or not path.is_file():
        raise MaudCoordinateError("private input is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaudCoordinateError("private input JSON drifted") from exc
    if canonical_bytes(value) != raw:
        raise MaudCoordinateError("private input is not canonical")
    return value, validate_private_input(value)


def _write_output(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise MaudCoordinateError("coordinate output is already consumed")
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(fd, "wb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(canonical_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
        if stat.S_IMODE(os.fstat(handle.fileno()).st_mode) != 0o600:
            raise MaudCoordinateError("coordinate output mode drifted")


def _production_minilm(model: Path):
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    _configure_torch_threads(torch)
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model_instance = SentenceTransformer(
        str(model),
        device="cuda:0",
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={
            "local_files_only": True,
            "torch_dtype": torch.float32,
            "use_safetensors": True,
        },
        config_kwargs={"local_files_only": True, "trust_remote_code": False},
    )
    model_instance.max_seq_length = MAX_SEQUENCE_LENGTH
    model_instance.eval()

    def encode(texts: Sequence[str]):
        return model_instance.encode(
            list(texts),
            batch_size=MINILM_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device="cuda:0",
        )

    return encode


def _production_cross_encoder(model: Path):
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    _configure_torch_threads(torch)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    model_instance = AutoModelForSequenceClassification.from_pretrained(
        str(model),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
    ).eval().to("cuda:0")

    def score(pairs: Sequence[tuple[str, str]]):
        encoded = tokenizer(
            [row[0] for row in pairs],
            [row[1] for row in pairs],
            max_length=MAX_SEQUENCE_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        with torch.inference_mode():
            return model_instance(**encoded).logits.detach().cpu().reshape(-1).tolist()

    return score


def run_worker(
    *,
    role: str,
    input_path: Path,
    output_path: Path,
    model_alias: str,
    model_tree_sha256: str,
) -> dict[str, object]:
    value, contracts = _load_input(input_path)
    input_sha256 = semantic_sha256(value)
    if role == ROLE_MINILM:
        model = _short_model_root(model_alias, "minilm")
        computed = compute_minilm(contracts, _production_minilm(model))
        rows = computed["rows"]
        pairwise = computed["contract_pairwise"]
    elif role == ROLE_CROSS_ENCODER:
        model = _short_model_root(model_alias, "cross_encoder")
        rows = compute_cross_encoder(contracts, _production_cross_encoder(model))
        pairwise = []
    else:
        raise MaudCoordinateError("coordinate role drifted")
    payload = coordinate_output(
        role=role,
        rows=rows,
        input_sha256=input_sha256,
        model_tree_sha256=model_tree_sha256,
        contract_pairwise=pairwise,
    )
    _write_output(output_path, payload)
    return {
        "status": "passed",
        "role": role,
        "contract_count": len(contracts),
        "row_count": len(rows),
        "input_sha256": input_sha256,
        "output_self_sha256": payload["self_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-tree-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    _require_native_thread_environment()
    from replication_runtime.maud_extraction_p2_official_v1 import (
        worker as thread_guard,
    )

    monitor = thread_guard._ProcessThreadPeakMonitor.start(os.getpid())
    try:
        result = run_worker(
            role=args.role,
            input_path=args.input,
            output_path=args.output,
            model_alias=args.model,
            model_tree_sha256=args.model_tree_sha256,
        )
    finally:
        observed_process_thread_peak = monitor.stop()
    # CUDA and Transformers may create helper threads even when native BLAS,
    # OpenMP, and both Torch compute pools are fixed to one. Total OS thread
    # count is diagnostic only; the worker count and outer cgroup bound use
    # of CPU, memory, processes, and tasks.
    result["observed_process_thread_peak"] = observed_process_thread_peak
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
