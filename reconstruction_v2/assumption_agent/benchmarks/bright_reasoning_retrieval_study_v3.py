"""Prospective BRIGHT v3 executor repair on still-unopened efficacy blocks.

V3 reuses the complete, pre-view v2 corpus tensor and the query-only G block
solely for executor qualification.  It changes only Qwen scheduling and its
wall-clock bound.  All retrieval recipes, evaluator candidates, labels,
metrics, promotion rules, HippoRAG execution, and later block assignments are
the frozen v1 study contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v1 as base
from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v2 as v2
from replication_runtime.bright_minilm_v1.encoder import float32_matrix_sha256


VERSION = "bright_reasoning_retrieval_study_v3"
DESIGN_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_executor_repair_design_v3.json"
)
FREEZE_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_study_implementation_freeze_v3.json"
)
DESIGN_SELF_SHA256 = "6769700d0e1a119ef7f21625f3ce68ba9f64fec8b0d3947ad2867cdae595ed43"
DESIGN_FILE_SHA256 = "61fa32095be8a7afb01f71fb593c3629d969c79aea67fd00786f4f94a99db2c7"
FORMAL_ROOT_RELATIVE = Path("artifacts/bright_reasoning_retrieval_study_v3")
CORPUS_TENSOR_ROOT_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_study_v2/corpus"
)
CORPUS_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_corpus_tensor_v2.json"
)
CORPUS_RESULT_SCHEMA = "bright_reasoning_retrieval_study_v2_corpus_result"

G_RESULT_RELATIVE = Path("manifests/bright_reasoning_retrieval_G_form_v3.json")
A_FORM_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_A_form_v3.json"
)
F_RESULT_RELATIVE = Path("manifests/bright_reasoning_retrieval_F_search_v3.json")
A_HOLD_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_A_hold_v3.json"
)
M_RESULT_RELATIVE = Path("manifests/bright_reasoning_retrieval_M_search_v3.json")

QWEN_WORKER_MODULE = "replication_runtime.bright_query_generator_v2.worker"
QWEN_TIMEOUT_SECONDS = 3_600


def _load_corpus_v3(project_root: Path) -> dict[str, base.CorpusFamily]:
    """Load and fully revalidate the complete v2 tensor without copying it."""

    result = base._load_json(
        project_root / CORPUS_RESULT_RELATIVE, "corpus result", canonical=True
    )
    if (
        result.get("schema") != CORPUS_RESULT_SCHEMA
        or result.get("status") != "corpus_tensor_complete"
    ):
        raise base.BrightStudyError("reused corpus result did not complete")
    base.verify_self_hash(result, "result_sha256")
    rows = result.get("family_tensors")
    if not isinstance(rows, Mapping) or set(rows) != set(base.core.FAMILY_ORDER):
        raise base.BrightStudyError("reused corpus family registry drifted")
    output: dict[str, base.CorpusFamily] = {}
    for family in base.core.FAMILY_ORDER:
        binding = rows[family]
        if not isinstance(binding, Mapping):
            raise base.BrightStudyError("reused corpus family binding drifted")
        root = project_root / CORPUS_TENSOR_ROOT_RELATIVE / family
        id_path = root / "ids.json"
        matrix_path = root / "embeddings.npy"
        if (
            base.file_sha256(id_path) != binding.get("id_pack_file_sha256")
            or base.file_sha256(matrix_path) != binding.get("embedding_file_sha256")
        ):
            raise base.BrightStudyError("reused corpus tensor files drifted")
        id_pack = base._load_json(id_path, "corpus ID pack", canonical=True)
        if (
            base.verify_self_hash(id_pack, "pack_sha256")
            != binding.get("id_pack_sha256")
        ):
            raise base.BrightStudyError("reused corpus ID pack drifted")
        ids_raw = id_pack.get("document_ids")
        if not isinstance(ids_raw, list):
            raise base.BrightStudyError("reused corpus IDs drifted")
        ids = tuple(base._required_text(value, "corpus document ID") for value in ids_raw)
        if len(ids) != binding.get("document_count") or len(set(ids)) != len(ids):
            raise base.BrightStudyError("reused corpus document identity drifted")
        try:
            matrix = np.load(matrix_path, allow_pickle=False)
        except Exception as exc:
            raise base.BrightStudyError("reused corpus embedding file is invalid") from exc
        matrix = np.asarray(matrix, dtype=np.float32)
        if (
            matrix.shape != (len(ids), 384)
            or binding.get("embedding_shape") != [len(ids), 384]
            or float32_matrix_sha256(matrix)
            != binding.get("embedding_float32_bytes_sha256")
            or not np.isfinite(matrix).all()
        ):
            raise base.BrightStudyError("reused corpus embedding matrix drifted")
        output[family] = base.CorpusFamily(
            family=family, ids=ids, embeddings=matrix
        )
    return output


def _parse_worker_schedule(raw: bytes, *, item_count: int) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise base.BrightStudyError("Qwen v2 worker stdout is invalid") from exc
    expected = {
        "batch_count",
        "batch_sizes",
        "input_item_count",
        "maximum_padded_prompt_tokens",
        "maximum_prompt_tokens",
        "oversized_singleton_count",
        "padded_prompt_token_budget",
        "status",
        "valid_generation_count",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("status") != "passed"
        or value.get("input_item_count") != item_count
        or not isinstance(value.get("batch_sizes"), list)
        or sum(value["batch_sizes"]) != item_count
        or value.get("batch_count") != len(value["batch_sizes"])
        or value.get("padded_prompt_token_budget") != 4096
    ):
        raise base.BrightStudyError("Qwen v2 worker schedule drifted")
    for field in (
        "batch_count",
        "maximum_padded_prompt_tokens",
        "maximum_prompt_tokens",
        "oversized_singleton_count",
        "valid_generation_count",
    ):
        observed = value.get(field)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise base.BrightStudyError("Qwen v2 worker schedule value drifted")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 8
        for size in value["batch_sizes"]
    ):
        raise base.BrightStudyError("Qwen v2 batch sizes drifted")
    return value


def _run_qwen_v3(
    project_root: Path, stage_root: Path, items: Sequence[base.ViewItem]
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_payload = {
        "items": [
            {"ordinal": item.ordinal, "query": item.query} for item in items
        ],
        "schema": base.QWEN_INPUT_SCHEMA,
    }
    input_path = stage_root / "qwen.input.json"
    output_path = stage_root / "qwen.output.json"
    base._write_exclusive(
        input_path, base.qwen_canonical_json_bytes(input_payload), mode=0o600
    )
    for name in ("home", "hf", "tmp"):
        (stage_root / name).mkdir(mode=0o700)
    trace_prefix = "qwen.network.trace"
    command = [
        "/usr/bin/strace",
        "-ff",
        "-e",
        "trace=network",
        "-o",
        str(stage_root / trace_prefix),
        str(project_root / base.HIPPORAG_PYTHON_RELATIVE),
        "-I",
        "-B",
        "-m",
        QWEN_WORKER_MODULE,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        str(project_root / base.QWEN_MODEL_RELATIVE),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HOME": str(stage_root / "hf"),
            "HF_HUB_OFFLINE": "1",
            "HOME": str(stage_root / "home"),
            "MPLCONFIGDIR": str(stage_root / "tmp" / "mpl"),
            "TOKENIZERS_PARALLELISM": "false",
            "TMPDIR": str(stage_root / "tmp"),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=False,
            capture_output=True,
            timeout=QWEN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise base.BrightStudyError("Qwen v2 worker timed out") from exc
    if completed.returncode != 0:
        raise base.BrightStudyError(
            "Qwen v2 worker failed: " + hashlib.sha256(completed.stderr).hexdigest()
        )
    if not output_path.is_file() or output_path.is_symlink():
        raise base.BrightStudyError("Qwen v2 worker output is unavailable")
    output = base.parse_qwen_output(output_path.read_bytes())
    if len(output["items"]) != len(items):
        raise base.BrightStudyError("Qwen v2 output item count drifted")
    schedule = _parse_worker_schedule(completed.stdout, item_count=len(items))
    valid_count = sum(row["generation_valid"] for row in output["items"])
    if schedule["valid_generation_count"] != valid_count:
        raise base.BrightStudyError("Qwen v2 validity receipt drifted")
    network = base._network_trace_receipt(stage_root, trace_prefix)
    receipt = {
        "input_file_sha256": base.file_sha256(input_path),
        "network_audit": network,
        "output_file_sha256": base.file_sha256(output_path),
        "schedule": schedule,
        "timeout_seconds": QWEN_TIMEOUT_SECONDS,
        "valid_generation_count": valid_count,
        "worker_stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "worker_stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }
    return output, receipt


def _activate_v3() -> None:
    public = {
        "G_form": G_RESULT_RELATIVE,
        "A_form": A_FORM_RESULT_RELATIVE,
        "F_search": F_RESULT_RELATIVE,
        "A_hold": A_HOLD_RESULT_RELATIVE,
        "M_search": M_RESULT_RELATIVE,
    }
    predecessors = {
        "G_form": CORPUS_RESULT_RELATIVE,
        "A_form": G_RESULT_RELATIVE,
        "F_search": A_FORM_RESULT_RELATIVE,
        "A_hold": F_RESULT_RELATIVE,
        "M_search": A_HOLD_RESULT_RELATIVE,
    }
    updates: dict[str, Any] = {
        "VERSION": VERSION,
        "DESIGN_SCHEMA": f"{VERSION}_design",
        "FREEZE_SCHEMA": f"{VERSION}_implementation_freeze",
        "CORPUS_RESULT_SCHEMA": CORPUS_RESULT_SCHEMA,
        "STAGE_RESULT_SCHEMA": f"{VERSION}_stage_result",
        "ACTION_SCHEMA": f"{VERSION}_local_action_pack",
        "SCORED_SCHEMA": f"{VERSION}_scored_pack",
        "MARKER_SCHEMA": f"{VERSION}_attempt",
        "DESIGN_RELATIVE": DESIGN_RELATIVE,
        "FREEZE_RELATIVE": FREEZE_RELATIVE,
        "DESIGN_SELF_SHA256": DESIGN_SELF_SHA256,
        "DESIGN_FILE_SHA256": DESIGN_FILE_SHA256,
        "FORMAL_ROOT_RELATIVE": FORMAL_ROOT_RELATIVE,
        "CORPUS_RESULT_RELATIVE": CORPUS_RESULT_RELATIVE,
        "G_RESULT_RELATIVE": G_RESULT_RELATIVE,
        "A_FORM_RESULT_RELATIVE": A_FORM_RESULT_RELATIVE,
        "F_RESULT_RELATIVE": F_RESULT_RELATIVE,
        "A_HOLD_RESULT_RELATIVE": A_HOLD_RESULT_RELATIVE,
        "M_RESULT_RELATIVE": M_RESULT_RELATIVE,
        "PUBLIC_STAGE_RESULTS": public,
        "STAGE_PREDECESSORS": predecessors,
        "_load_corpus": _load_corpus_v3,
        "_read_source_documents": v2._read_source_documents_v2,
        "_run_qwen": _run_qwen_v3,
    }
    for name, value in updates.items():
        setattr(base, name, value)


def run(command: str, project_root: Path) -> dict[str, Any]:
    _activate_v3()
    functions = {
        "G-form": base.run_g_form,
        "A-form": base.run_a_form,
        "F-search": base.run_f_search,
        "A-hold": base.run_a_hold,
        "M-search": base.run_m_search,
    }
    if command not in functions:
        raise base.BrightStudyError("v3 command is invalid")
    return functions[command](project_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("G-form", "A-form", "F-search", "A-hold", "M-search")
    )
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run(arguments.command, arguments.project_root)
    print(
        base.canonical_json_bytes(
            {"result_sha256": result["result_sha256"], "status": result["status"]}
        ).decode("ascii"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
