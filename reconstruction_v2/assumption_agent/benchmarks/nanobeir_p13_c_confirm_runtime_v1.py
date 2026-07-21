"""Prospective P13 C_confirm with bridge-safe queries and cached HippoRAG."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
    nanobeir_p12_completecase_c_confirm_runtime_v1 as mature,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p13_acquisition_v1 as acquisition,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p13_bridge_safe_candidate_v1 as candidate,
)


SCHEMA = "nanobeir_p13_c_confirm_runtime_result_v1"
ATTEMPT_SCHEMA = "nanobeir_p13_c_confirm_runtime_attempt_v1"
INTENT_SCHEMA = "nanobeir_p13_c_confirm_runtime_intents_v1"
ACTION_SCHEMA = "nanobeir_p13_c_confirm_runtime_actions_v1"
FREEZE_SCHEMA = "nanobeir_p13_c_confirm_runtime_freeze_v1"
CANDIDATE_NAME = candidate.CANDIDATE_NAME
ITEMS_PER_FAMILY = 10
ITEM_COUNT = 30

RUN_ROOT_RELATIVE = Path("artifacts/nanobeir_p13_c_confirm_runtime_v1")
P11_INTERNAL_RESULT_RELATIVE = RUN_ROOT_RELATIVE / "internal.p11.result.json"
P12_INTERNAL_RESULT_RELATIVE = RUN_ROOT_RELATIVE / "internal.p12.result.json"
MATURE_INTERNAL_RESULT_RELATIVE = (
    RUN_ROOT_RELATIVE / "internal.completecase.result.json"
)
RESULT_RELATIVE = Path("manifests/nanobeir_p13_c_confirm_runtime_result_v1.json")
FREEZE_RELATIVE = Path("manifests/nanobeir_p13_c_confirm_runtime_freeze_v1.json")
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/nanobeir_p13_c_confirm_runtime_v1.py"
)
TEST_RELATIVE = Path("tests/test_nanobeir_p13_c_confirm_runtime_v1.py")
ACQUISITION_RESULT_RELATIVE = acquisition.RESULT_RELATIVE
CANDIDATE_FREEZE_SELF_SHA256 = (
    "17f9865483cd3c4846db8a63c1047f8af6bdaa24b78ece09245f3e568e0457f0"
)
STUDY_DESIGN_SELF_SHA256 = (
    "7d7230e3af8b1cc906e494851c52dc1dcc9ca04b4f247eeee7a3248494fb4e08"
)
DEPENDENCY_RELATIVES = (
    mature.IMPLEMENTATION_RELATIVE,
    Path("assumption_agent/benchmarks/nanobeir_p11_c_confirm_runtime_v1.py"),
    Path("assumption_agent/benchmarks/nanobeir_p12_c_confirm_runtime_v1.py"),
    Path("assumption_agent/benchmarks/p11_raw_ce_rrf_v1.py"),
    Path("assumption_agent/benchmarks/fiqa_bridge_expansion_train_runtime_v1.py"),
    acquisition.IMPLEMENTATION_RELATIVE,
    Path("assumption_agent/benchmarks/nanobeir_p13_bridge_safe_candidate_v1.py"),
    Path("replication_runtime/bright_official_hipporag_v1/contract.py"),
)


class P13CConfirmError(RuntimeError):
    """The frozen P13 C_confirm runtime failed closed."""


class OneShotRefusal(P13CConfirmError):
    """The formal P13 runtime root or result is already consumed."""


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise P13CConfirmError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P13CConfirmError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise P13CConfirmError(f"{name} is not an object")
    return value


def _verify_self(value: Mapping[str, Any], expected: str, name: str) -> None:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if declared != expected or acquisition.stable_hash(body) != expected:
        raise P13CConfirmError(f"{name} self hash drifted")


def _runtime_freeze(base: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "P13 C_confirm freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise P13CConfirmError("P13 C_confirm freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise P13CConfirmError("P13 C_confirm freeze hash is absent")
    _verify_self(value, declared, "P13 C_confirm freeze")
    commit = value.get("formal_implementation_commit")
    if (
        not isinstance(commit, str)
        or not acquisition.mature._git_is_ancestor(commit, base.parent)
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
    ):
        raise P13CConfirmError("P13 C_confirm freeze prerequisite drifted")
    rows = value.get("dependency_bindings")
    if not isinstance(rows, list):
        raise P13CConfirmError("P13 dependency bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    expected = {path.as_posix() for path in DEPENDENCY_RELATIVES}
    if set(observed) != expected:
        raise P13CConfirmError("P13 dependency set drifted")
    for relative, digest in observed.items():
        if (
            not isinstance(digest, str)
            or acquisition.file_sha256(base / str(relative)) != digest
        ):
            raise P13CConfirmError("P13 dependency file drifted")
    return value


def _load_acquisition(base: Path) -> Mapping[str, Any]:
    freeze = _runtime_freeze(base)
    binding = freeze.get("acquisition_result_binding")
    if not isinstance(binding, Mapping):
        raise P13CConfirmError("P13 acquisition result binding is absent")
    path = base / ACQUISITION_RESULT_RELATIVE
    if acquisition.file_sha256(path) != binding.get("file_sha256"):
        raise P13CConfirmError("P13 acquisition result file drifted")
    value = _read_json(path, "P13 acquisition result")
    expected_self = binding.get("self_sha256")
    if not isinstance(expected_self, str):
        raise P13CConfirmError("P13 acquisition result hash is absent")
    _verify_self(value, expected_self, "P13 acquisition result")
    if (
        value.get("schema") != acquisition.SCHEMA
        or value.get("status")
        != "passed_99_item_private_acquisition_ready_for_P13_C_confirm_runtime"
        or value.get("study_design_self_sha256") != STUDY_DESIGN_SELF_SHA256
    ):
        raise P13CConfirmError("P13 acquisition completion drifted")
    return value


def load_corpora(base: Path) -> Mapping[str, p11_runtime.FamilyCorpus]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise P13CConfirmError("pyarrow is unavailable") from exc
    output: dict[str, p11_runtime.FamilyCorpus] = {}
    excluded_counts: dict[str, int] = {}
    for family in acquisition.FAMILIES:
        path = (
            base
            / acquisition.SOURCE_ROOT_RELATIVE
            / "corpus"
            / f"{family}-00000-of-00001.parquet"
        )
        expected = acquisition.SOURCE_FILES[
            f"corpus/{family}-00000-of-00001.parquet"
        ]
        if acquisition.file_sha256(path) != expected:
            raise P13CConfirmError("P13 corpus source drifted")
        table = pq.read_table(path)
        if table.column_names != ["_id", "text"]:
            raise P13CConfirmError("P13 corpus schema drifted")
        ids: list[str] = []
        contents: list[str] = []
        excluded = 0
        for row in table.to_pylist():
            identifier = row.get("_id")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                excluded += 1
                continue
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in ids
            ):
                raise P13CConfirmError("P13 corpus identity drifted")
            contents.append(acquisition.project_document(text))
            ids.append(identifier)
        if len(ids) < 32 or len(ids) != len(set(ids)):
            raise P13CConfirmError("P13 corpus capacity drifted")
        excluded_counts[family] = excluded
        output[family] = p11_runtime.FamilyCorpus(tuple(ids), tuple(contents))
    if excluded_counts != {
        "NanoFiQA2018": 27,
        "NanoNFCorpus": 0,
        "NanoTouche2020": 0,
    }:
        raise P13CConfirmError("P13 shared corpus filter drifted")
    return output


def _rename_p11_slot_to_p13(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            (key.replace("P11", "P13") if isinstance(key, str) else key): (
                _rename_p11_slot_to_p13(child)
            )
            for key, child in value.items()
            if key != "self_sha256"
        }
    if isinstance(value, list):
        return [_rename_p11_slot_to_p13(child) for child in value]
    if isinstance(value, str):
        return value.replace("P11", "P13")
    return value


def _p13_totalizer(original: Any):
    def project(output: Mapping[str, Any], items: Sequence[Any]):
        current = p12_runtime.totalize_qwen_output
        p12_runtime.totalize_qwen_output = original
        try:
            return candidate.totalize_and_project_qwen_output(output, items)
        finally:
            p12_runtime.totalize_qwen_output = current

    return project


@contextmanager
def _patched_mature_runtime() -> Iterator[None]:
    mature_replacements = {
        "SCHEMA": SCHEMA,
        "ATTEMPT_SCHEMA": ATTEMPT_SCHEMA,
        "INTENT_SCHEMA": INTENT_SCHEMA,
        "ACTION_SCHEMA": ACTION_SCHEMA,
        "FREEZE_SCHEMA": FREEZE_SCHEMA,
        "RUN_ROOT_RELATIVE": RUN_ROOT_RELATIVE,
        "P11_INTERNAL_RESULT_RELATIVE": P11_INTERNAL_RESULT_RELATIVE,
        "P12_INTERNAL_RESULT_RELATIVE": P12_INTERNAL_RESULT_RELATIVE,
        "RESULT_RELATIVE": MATURE_INTERNAL_RESULT_RELATIVE,
        "FREEZE_RELATIVE": FREEZE_RELATIVE,
        "IMPLEMENTATION_RELATIVE": IMPLEMENTATION_RELATIVE,
        "TEST_RELATIVE": TEST_RELATIVE,
        "DEPENDENCY_RELATIVES": DEPENDENCY_RELATIVES,
        "ACQUISITION_RESULT_RELATIVE": ACQUISITION_RESULT_RELATIVE,
        "CANDIDATE_FREEZE_SELF_SHA256": CANDIDATE_FREEZE_SELF_SHA256,
        "STUDY_DESIGN_SELF_SHA256": STUDY_DESIGN_SELF_SHA256,
        "acquisition": acquisition,
        "_load_acquisition": _load_acquisition,
    }
    mature_originals = {
        name: getattr(mature, name) for name in mature_replacements
    }
    original_load_corpora = p11_runtime.load_corpora
    original_totalizer = p12_runtime.totalize_qwen_output
    original_candidate_name = p12_runtime.CANDIDATE_NAME
    original_slot_rename = p12_runtime._rename_public_slot
    try:
        for name, value in mature_replacements.items():
            setattr(mature, name, value)
        p11_runtime.load_corpora = load_corpora
        p12_runtime.CANDIDATE_NAME = CANDIDATE_NAME
        p12_runtime._rename_public_slot = _rename_p11_slot_to_p13
        p12_runtime.totalize_qwen_output = _p13_totalizer(original_totalizer)
        yield
    finally:
        p12_runtime.totalize_qwen_output = original_totalizer
        p12_runtime._rename_public_slot = original_slot_rename
        p12_runtime.CANDIDATE_NAME = original_candidate_name
        p11_runtime.load_corpora = original_load_corpora
        for name, value in mature_originals.items():
            setattr(mature, name, value)


def run_formal(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    root = base / RUN_ROOT_RELATIVE
    result_path = base / RESULT_RELATIVE
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("P13 C_confirm root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("P13 C_confirm result already exists")
    with _patched_mature_runtime():
        internal = mature.run_formal(project_root)
    body = dict(internal)
    body.pop("self_sha256", None)
    execution = body.get("execution")
    if not isinstance(execution, Mapping):
        raise P13CConfirmError("P13 internal execution receipt drifted")
    body["execution"] = {
        **execution,
        "bridge_safe_projection": {
            "candidate": CANDIDATE_NAME,
            "maximum_anchor_characters": 96,
            "maximum_composed_bridge_query_characters": 768,
            "typed_query_character_cap": 671,
        },
    }
    alias = body.get("internal_action_slot_alias")
    if not isinstance(alias, Mapping):
        raise P13CConfirmError("P13 internal action alias drifted")
    body["internal_action_slot_alias"] = {
        **alias,
        "public_candidate": "P13",
    }
    body["internal_completecase_result_file_sha256"] = acquisition.file_sha256(
        base / MATURE_INTERNAL_RESULT_RELATIVE
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
