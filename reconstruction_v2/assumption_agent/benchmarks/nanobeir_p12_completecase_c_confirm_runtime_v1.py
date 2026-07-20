"""Prospective P12 complete-case C_confirm with byte-reused HippoRAG outputs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p11_c_confirm_runtime_v1 as p11_runtime,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_c_confirm_runtime_v1 as p12_runtime,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_completecase_acquisition_v1 as acquisition,
)


SCHEMA = "nanobeir_p12_completecase_c_confirm_runtime_result_v1"
ATTEMPT_SCHEMA = "nanobeir_p12_completecase_c_confirm_runtime_attempt_v1"
INTENT_SCHEMA = "nanobeir_p12_completecase_c_confirm_runtime_intents_v1"
ACTION_SCHEMA = "nanobeir_p12_completecase_c_confirm_runtime_actions_v1"
FREEZE_SCHEMA = "nanobeir_p12_completecase_c_confirm_runtime_freeze_v1"

ITEMS_PER_FAMILY = 10
ITEM_COUNT = 30
RUN_ROOT_RELATIVE = Path(
    "artifacts/nanobeir_p12_completecase_c_confirm_runtime_v1"
)
P11_INTERNAL_RESULT_RELATIVE = RUN_ROOT_RELATIVE / "internal.p11.result.json"
P12_INTERNAL_RESULT_RELATIVE = RUN_ROOT_RELATIVE / "internal.p12.result.json"
RESULT_RELATIVE = Path(
    "manifests/nanobeir_p12_completecase_c_confirm_runtime_result_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/nanobeir_p12_completecase_c_confirm_runtime_freeze_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/"
    "nanobeir_p12_completecase_c_confirm_runtime_v1.py"
)
TEST_RELATIVE = Path(
    "tests/test_nanobeir_p12_completecase_c_confirm_runtime_v1.py"
)
DEPENDENCY_RELATIVES = (
    Path("assumption_agent/benchmarks/nanobeir_p11_c_confirm_runtime_v1.py"),
    Path("assumption_agent/benchmarks/nanobeir_p12_c_confirm_runtime_v1.py"),
    Path("assumption_agent/benchmarks/p11_raw_ce_rrf_v1.py"),
    Path("assumption_agent/benchmarks/fiqa_bridge_expansion_train_runtime_v1.py"),
    acquisition.IMPLEMENTATION_RELATIVE,
    Path("replication_runtime/bright_official_hipporag_v1/contract.py"),
)
ACQUISITION_RESULT_RELATIVE = acquisition.RESULT_RELATIVE
CANDIDATE_FREEZE_SELF_SHA256 = (
    "2421b8c9fec755f6a7087621771b376dd77a4a726ef23ee8c248268044a5bd9e"
)
STUDY_DESIGN_SELF_SHA256 = (
    "758563170e7c51a1ee503f4da53ef7d63f452711f7c85b7c74188b356ba4ad8f"
)


class CompleteCaseCConfirmError(RuntimeError):
    """The frozen complete-case C_confirm runtime failed closed."""


class OneShotRefusal(CompleteCaseCConfirmError):
    """The formal runtime root or public result is already consumed."""


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompleteCaseCConfirmError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompleteCaseCConfirmError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise CompleteCaseCConfirmError(f"{name} is not an object")
    return value


def _verify_self(value: Mapping[str, Any], expected: str, name: str) -> None:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if declared != expected or acquisition.stable_hash(body) != expected:
        raise CompleteCaseCConfirmError(f"{name} self hash drifted")


def _runtime_freeze(base: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "C_confirm freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise CompleteCaseCConfirmError("C_confirm freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise CompleteCaseCConfirmError("C_confirm freeze hash is absent")
    _verify_self(value, declared, "C_confirm freeze")
    bindings = value.get("dependency_bindings")
    if not isinstance(bindings, list):
        raise CompleteCaseCConfirmError("C_confirm dependency bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in bindings
        if isinstance(row, Mapping)
    }
    required = {path.as_posix() for path in DEPENDENCY_RELATIVES}
    if set(observed) != required:
        raise CompleteCaseCConfirmError("C_confirm dependency set drifted")
    for relative, expected in observed.items():
        if (
            not isinstance(expected, str)
            or acquisition.file_sha256(base / str(relative)) != expected
        ):
            raise CompleteCaseCConfirmError("C_confirm dependency drifted")
    return value


def _load_acquisition(base: Path) -> Mapping[str, Any]:
    freeze = _runtime_freeze(base)
    binding = freeze.get("acquisition_result_binding")
    if not isinstance(binding, Mapping):
        raise CompleteCaseCConfirmError("acquisition result binding is absent")
    path = base / ACQUISITION_RESULT_RELATIVE
    if acquisition.file_sha256(path) != binding.get("file_sha256"):
        raise CompleteCaseCConfirmError("acquisition result file drifted")
    value = _read_json(path, "acquisition result")
    expected_self = binding.get("self_sha256")
    if not isinstance(expected_self, str):
        raise CompleteCaseCConfirmError("acquisition result hash is absent")
    _verify_self(value, expected_self, "acquisition result")
    if (
        value.get("schema") != acquisition.SCHEMA
        or value.get("status")
        != (
            "passed_99_item_private_acquisition_ready_for_"
            "P12_completecase_C_confirm_runtime"
        )
        or value.get("study_design_self_sha256") != STUDY_DESIGN_SELF_SHA256
    ):
        raise CompleteCaseCConfirmError("acquisition completion drifted")
    return value


def _pack_binding(result: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    bindings = result.get("pack_bindings")
    if not isinstance(bindings, Mapping) or not isinstance(
        bindings.get(name), Mapping
    ):
        raise CompleteCaseCConfirmError(f"{name} binding is absent")
    return bindings[name]


def _load_pack(
    base: Path, result: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    binding = _pack_binding(result, name)
    relative = binding.get("relative_path")
    if not isinstance(relative, str):
        raise CompleteCaseCConfirmError(f"{name} path drifted")
    path = base / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != binding.get("size_bytes")
        or acquisition.file_sha256(path) != binding.get("file_sha256")
    ):
        raise CompleteCaseCConfirmError(f"{name} file drifted")
    value = _read_json(path, name)
    body = dict(value)
    observed = body.pop("pack_sha256", None)
    if (
        observed != binding.get("pack_sha256")
        or acquisition.stable_hash(body) != observed
    ):
        raise CompleteCaseCConfirmError(f"{name} pack hash drifted")
    return value


def load_cached_hipporag(
    base: Path, acquisition_result: Mapping[str, Any]
) -> Mapping[int, Mapping[str, Any]]:
    view = _load_pack(base, acquisition_result, "C_confirm_view")
    hippo = _load_pack(base, acquisition_result, "C_confirm_hipporag")
    if (
        view.get("schema") != acquisition.VIEW_SCHEMA
        or hippo.get("schema") != acquisition.HIPPO_SCHEMA
        or view.get("block") != "C_confirm"
        or hippo.get("block") != "C_confirm"
    ):
        raise CompleteCaseCConfirmError("C_confirm cache envelope drifted")
    view_rows = view.get("items")
    hippo_rows = hippo.get("items")
    if (
        not isinstance(view_rows, list)
        or not isinstance(hippo_rows, list)
        or len(view_rows) != ITEM_COUNT
        or len(hippo_rows) != ITEM_COUNT
    ):
        raise CompleteCaseCConfirmError("C_confirm cache item count drifted")
    output: dict[int, Mapping[str, Any]] = {}
    for ordinal, (view_row, cache_row) in enumerate(zip(view_rows, hippo_rows)):
        if (
            not isinstance(view_row, Mapping)
            or not isinstance(cache_row, Mapping)
            or cache_row.get("item_key") != view_row.get("item_key")
            or cache_row.get("family") != view_row.get("family")
        ):
            raise CompleteCaseCConfirmError("C_confirm cache identity drifted")
        base_pool = cache_row.get("base_pool")
        raw_top10 = cache_row.get("raw_top10")
        top_rows = cache_row.get("top_rows")
        if not acquisition.valid_cached_rank_sets(
            base_pool, raw_top10, top_rows
        ):
            raise CompleteCaseCConfirmError("C_confirm cached ranks drifted")
        source_ordinal = cache_row.get("source_screen_ordinal")
        expected_relative = (
            acquisition.AVAILABILITY_ROOT_RELATIVE
            / "hipporag"
            / f"item_{source_ordinal:03d}"
            / "output.json"
            if isinstance(source_ordinal, int) and not isinstance(source_ordinal, bool)
            else None
        )
        if (
            expected_relative is None
            or cache_row.get("source_output_relative_path")
            != expected_relative.as_posix()
        ):
            raise CompleteCaseCConfirmError("C_confirm source path drifted")
        source = base / expected_relative
        if acquisition.file_sha256(source) != cache_row.get(
            "source_output_file_sha256"
        ):
            raise CompleteCaseCConfirmError("C_confirm source output drifted")
        try:
            payload = p11_runtime.train.hippo_contract.parse_output(
                source.read_bytes()
            )
        except Exception as exc:
            raise CompleteCaseCConfirmError(
                "C_confirm source output is invalid"
            ) from exc
        derived = [base_pool[index] for index in payload["top_ordinals"]]
        if (
            derived != top_rows
            or payload["graph_node_count"] != cache_row.get("graph_node_count")
            or payload["graph_edge_count"] != cache_row.get("graph_edge_count")
        ):
            raise CompleteCaseCConfirmError("C_confirm source receipt drifted")
        output[ordinal] = cache_row
    return output


def cached_hipporag_runner(
    *, base: Path, cached: Mapping[int, Mapping[str, Any]]
):
    expected_base = base.resolve()

    def run(
        *,
        base: Path,
        item_root: Path,
        candidate_rows: Sequence[int],
        patched_source: Path,
        semaphore: Any,
        counter: Any,
    ) -> Mapping[str, Any]:
        del patched_source, semaphore, counter
        if base.resolve() != expected_base:
            raise p11_runtime.NanoBEIRCConfirmError(
                "cached HippoRAG base root drifted"
            )
        try:
            ordinal = int(item_root.name.removeprefix("item_"))
        except ValueError as exc:
            raise p11_runtime.NanoBEIRCConfirmError(
                "cached HippoRAG item identity drifted"
            ) from exc
        row = cached.get(ordinal)
        if row is None or list(candidate_rows) != row.get("base_pool"):
            raise p11_runtime.NanoBEIRCConfirmError(
                "cached HippoRAG base pool drifted"
            )
        relative = row["source_output_relative_path"]
        source = base / relative
        if acquisition.file_sha256(source) != row["source_output_file_sha256"]:
            raise p11_runtime.NanoBEIRCConfirmError(
                "cached HippoRAG output bytes drifted"
            )
        payload = p11_runtime.train.hippo_contract.parse_output(source.read_bytes())
        top_rows = [candidate_rows[index] for index in payload["top_ordinals"]]
        if (
            top_rows != row["top_rows"]
            or payload["graph_node_count"] != row["graph_node_count"]
            or payload["graph_edge_count"] != row["graph_edge_count"]
        ):
            raise p11_runtime.NanoBEIRCConfirmError(
                "cached HippoRAG parsed output drifted"
            )
        receipt = acquisition.self_hashed(
            {
                "comparator_relaunch_count": 0,
                "item_ordinal": ordinal,
                "schema": "nanobeir_p12_completecase_hipporag_reuse_receipt_v1",
                "source_output_file_sha256": row[
                    "source_output_file_sha256"
                ],
                "source_output_relative_path": relative,
            },
            field="pack_sha256",
        )
        receipt_path = item_root / "reused.screen.output.json"
        p11_runtime.train.bright_runtime._write_json(receipt_path, receipt)
        return {
            "comparator_relaunch_count": 0,
            "graph_edge_count": payload["graph_edge_count"],
            "graph_node_count": payload["graph_node_count"],
            "output_file_sha256": row["source_output_file_sha256"],
            "reuse_receipt_file_sha256": acquisition.file_sha256(receipt_path),
            "source_screen_ordinal": row["source_screen_ordinal"],
            "stderr_sha256": row["source_stderr_sha256"],
            "stdout_sha256": row["source_stdout_sha256"],
            "top_rows": top_rows,
        }

    return run


def _replace_pack_schema(
    original: Any, base: Path, binding: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    value = original(base, binding, name)
    replacements = {
        acquisition.VIEW_SCHEMA: "nanobeir_p11_private_view_v1",
        acquisition.LABEL_SCHEMA: "nanobeir_p11_private_labels_v1",
    }
    if value.get("schema") in replacements:
        value = dict(value)
        value["schema"] = replacements[value["schema"]]
    return value


@contextmanager
def _patched_runtime(
    *,
    base: Path,
    acquisition_result: Mapping[str, Any],
    cached: Mapping[int, Mapping[str, Any]],
) -> Iterator[None]:
    acquisition_self = acquisition_result["self_sha256"]
    acquisition_file = acquisition.file_sha256(
        base / ACQUISITION_RESULT_RELATIVE
    )
    p12_replacements = {
        "SCHEMA": SCHEMA,
        "ATTEMPT_SCHEMA": ATTEMPT_SCHEMA,
        "INTENT_SCHEMA": INTENT_SCHEMA,
        "ACTION_SCHEMA": ACTION_SCHEMA,
        "FREEZE_SCHEMA": FREEZE_SCHEMA,
        "RUN_ROOT_RELATIVE": RUN_ROOT_RELATIVE,
        "INTERNAL_RESULT_RELATIVE": P11_INTERNAL_RESULT_RELATIVE,
        "RESULT_RELATIVE": P12_INTERNAL_RESULT_RELATIVE,
        "FREEZE_RELATIVE": FREEZE_RELATIVE,
        "IMPLEMENTATION_RELATIVE": IMPLEMENTATION_RELATIVE,
        "TEST_RELATIVE": TEST_RELATIVE,
        "ACQUISITION_RESULT_RELATIVE": ACQUISITION_RESULT_RELATIVE,
        "ACQUISITION_RESULT_FILE_SHA256": acquisition_file,
        "ACQUISITION_RESULT_SELF_SHA256": acquisition_self,
        "CANDIDATE_FREEZE_SELF_SHA256": CANDIDATE_FREEZE_SELF_SHA256,
        "STUDY_DESIGN_SELF_SHA256": STUDY_DESIGN_SELF_SHA256,
        "acquisition": acquisition,
        "_load_acquisition": _load_acquisition,
        "_replace_pack_schema": _replace_pack_schema,
    }
    p11_replacements = {
        "ITEMS_PER_FAMILY": ITEMS_PER_FAMILY,
        "ITEM_COUNT": ITEM_COUNT,
        "_run_hardened_hipporag_item": cached_hipporag_runner(
            base=base, cached=cached
        ),
    }
    originals12 = {name: getattr(p12_runtime, name) for name in p12_replacements}
    originals11 = {name: getattr(p11_runtime, name) for name in p11_replacements}
    try:
        for name, value in p12_replacements.items():
            setattr(p12_runtime, name, value)
        for name, value in p11_replacements.items():
            setattr(p11_runtime, name, value)
        yield
    finally:
        for name, value in originals11.items():
            setattr(p11_runtime, name, value)
        for name, value in originals12.items():
            setattr(p12_runtime, name, value)


def run_formal(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    root = base / RUN_ROOT_RELATIVE
    result_path = base / RESULT_RELATIVE
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("complete-case C_confirm root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("complete-case C_confirm result already exists")
    acquisition_result = _load_acquisition(base)
    cached = load_cached_hipporag(base, acquisition_result)
    with _patched_runtime(
        base=base, acquisition_result=acquisition_result, cached=cached
    ):
        internal = p12_runtime.run_formal(project_root)
    body = dict(internal)
    body.pop("self_sha256", None)
    execution = body.get("execution")
    if not isinstance(execution, Mapping):
        raise CompleteCaseCConfirmError("internal execution receipt drifted")
    body["execution"] = {
        **execution,
        "HippoRAG_exact_screen_output_reuse_count": ITEM_COUNT,
        "HippoRAG_peak_process_concurrency": 0,
        "HippoRAG_relaunch_count": 0,
    }
    claim = body.get("claim_boundary")
    if not isinstance(claim, Mapping):
        raise CompleteCaseCConfirmError("internal claim boundary drifted")
    body["claim_boundary"] = {
        **claim,
        "comparator_terminal_complete_case_scope_only": True,
        "full_source_population_claim": False,
    }
    body["internal_p12_result_file_sha256"] = acquisition.file_sha256(
        base / P12_INTERNAL_RESULT_RELATIVE
    )
    body["recorded_date"] = "2026-07-21"
    result = acquisition.self_hashed(body)
    acquisition._write_exclusive(
        result_path, acquisition.canonical_json_bytes(result), mode=0o644
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.formal:
        raise SystemExit("--formal is required")
    result = run_formal(arguments.project_root)
    print(
        json.dumps(
            {
                "primary_passed": result["primary_passed"],
                "self_sha256": result["self_sha256"],
                "status": result["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
