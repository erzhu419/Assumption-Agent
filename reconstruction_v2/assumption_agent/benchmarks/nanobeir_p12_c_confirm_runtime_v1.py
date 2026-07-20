"""One-shot offline NanoBEIR C_confirm runtime for P12, RAW, and HippoRAG.

P12 preserves the frozen P11 ranker and totalizes only Qwen rows that fail the
frozen typed-query grammar.  The mature P11 controller is reused under an
explicit, restored compatibility context; the final public result renames the
legacy internal rank slot from P11 to P12.
"""

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
    nanobeir_p12_acquisition_v1 as acquisition,
)


SCHEMA = "nanobeir_p12_c_confirm_runtime_result_v1"
ATTEMPT_SCHEMA = "nanobeir_p12_c_confirm_runtime_attempt_v1"
INTENT_SCHEMA = "nanobeir_p12_c_confirm_runtime_intents_v1"
ACTION_SCHEMA = "nanobeir_p12_c_confirm_runtime_actions_v1"
FREEZE_SCHEMA = "nanobeir_p12_c_confirm_runtime_implementation_freeze_v1"
TOTALIZED_SCHEMA = "nanobeir_p12_totalized_typed_queries_v1"
CANDIDATE_NAME = "P12_P11_TOTAL_TYPED_QUERY_V1"

RUN_ROOT_RELATIVE = Path("artifacts/nanobeir_p12_c_confirm_runtime_v1")
INTERNAL_RESULT_RELATIVE = RUN_ROOT_RELATIVE / "internal.p11_slot.result.json"
RESULT_RELATIVE = Path("manifests/nanobeir_p12_c_confirm_runtime_result_v1.json")
FREEZE_RELATIVE = Path(
    "manifests/nanobeir_p12_c_confirm_runtime_implementation_freeze_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/nanobeir_p12_c_confirm_runtime_v1.py"
)
TEST_RELATIVE = Path("tests/test_nanobeir_p12_c_confirm_runtime_v1.py")
ACQUISITION_RESULT_RELATIVE = Path(
    "manifests/nanobeir_p12_acquisition_result_v1.json"
)
ACQUISITION_RESULT_FILE_SHA256 = (
    "17d0893048bb5a392a9adac0c7ff20f713e0fa417cd77b1d476a45181fe26196"
)
ACQUISITION_RESULT_SELF_SHA256 = (
    "e653caeddcc9b21f02eb533ec62390f1d9775780bfcbeab996819ec95b424fa9"
)
CANDIDATE_FREEZE_SELF_SHA256 = (
    "2421b8c9fec755f6a7087621771b376dd77a4a726ef23ee8c248268044a5bd9e"
)
STUDY_DESIGN_SELF_SHA256 = (
    "07d8bd294977696910a55c2e56e37870ffe04eab67262177d304c8f9a8bb78b4"
)

FALLBACK_SUFFIXES = (
    "named entities terminology",
    "relationship comparison",
    "causal mechanism explanation",
    "conditions exclusions context",
)


class NanoBEIRP12CConfirmError(RuntimeError):
    """The frozen prospective P12 C_confirm runtime failed closed."""


class OneShotRefusal(NanoBEIRP12CConfirmError):
    """The P12 formal runtime root or result is already consumed."""


def _fallback_base(query: str) -> str:
    if not isinstance(query, str):
        raise NanoBEIRP12CConfirmError("fallback query is not text")
    value = " ".join(query.split())[:900].strip()
    if not value or "\x00" in value:
        raise NanoBEIRP12CConfirmError("fallback query is empty or unsafe")
    return value


def totalize_qwen_output(
    output: Mapping[str, Any], items: Sequence[Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = output.get("items")
    if not isinstance(rows, list) or len(rows) != len(items):
        raise NanoBEIRP12CConfirmError("Qwen output shape drifted before totalization")
    totalized_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    source_valid_count = 0
    totalized_count = 0
    for ordinal, (row, item) in enumerate(zip(rows, items)):
        if not isinstance(row, Mapping) or row.get("ordinal") != ordinal:
            raise NanoBEIRP12CConfirmError("Qwen row identity drifted")
        source_valid = row.get("generation_valid") is True
        copied = dict(row)
        if source_valid:
            expansions = copied.get("expansions")
            if (
                not isinstance(expansions, list)
                or len(expansions) != 4
                or any(not isinstance(value, str) or not value for value in expansions)
            ):
                raise NanoBEIRP12CConfirmError("valid Qwen row expansions drifted")
            source_valid_count += 1
        else:
            base = _fallback_base(item.query)
            expansions = [f"{base} {suffix}" for suffix in FALLBACK_SUFFIXES]
            if len(set(expansions)) != 4 or any(len(value) > 1000 for value in expansions):
                raise NanoBEIRP12CConfirmError("deterministic fallback drifted")
            copied["expansions"] = expansions
            copied["generation_valid"] = True
            totalized_count += 1
        totalized_rows.append(copied)
        audit_rows.append(
            {
                "completion_sha256": row.get("completion_sha256"),
                "completion_token_count": row.get("completion_token_count"),
                "expansions": list(copied["expansions"]),
                "ordinal": ordinal,
                "source_generation_valid": source_valid,
                "totalization_used": not source_valid,
            }
        )
    totalized = {"items": totalized_rows, "schema": output.get("schema")}
    audit = acquisition.self_hashed(
        {
            "candidate": CANDIDATE_NAME,
            "items": audit_rows,
            "schema": TOTALIZED_SCHEMA,
            "source_valid_generation_count": source_valid_count,
            "totalized_generation_count": totalized_count,
        },
        field="pack_sha256",
    )
    return totalized, audit


def _load_acquisition(base: Path) -> Mapping[str, Any]:
    path = base / ACQUISITION_RESULT_RELATIVE
    if acquisition.file_sha256(path) != ACQUISITION_RESULT_FILE_SHA256:
        raise p11_runtime.NanoBEIRCConfirmError("acquisition result file drifted")
    value = p11_runtime._read_json(path, "acquisition result")
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if (
        declared != ACQUISITION_RESULT_SELF_SHA256
        or acquisition.stable_hash(body) != ACQUISITION_RESULT_SELF_SHA256
        or value.get("schema") != acquisition.SCHEMA
        or value.get("status")
        != "passed_138_item_private_acquisition_ready_for_P12_C_confirm_runtime"
        or value.get("study_design_self_sha256") != STUDY_DESIGN_SELF_SHA256
    ):
        raise p11_runtime.NanoBEIRCConfirmError("acquisition completion drifted")
    return value


def _replace_pack_schema(
    original: Any, base: Path, binding: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    value = original(base, binding, name)
    schema = value.get("schema")
    replacements = {
        "nanobeir_p12_private_view_v1": "nanobeir_p11_private_view_v1",
        "nanobeir_p12_private_labels_v1": "nanobeir_p11_private_labels_v1",
    }
    if schema in replacements:
        value = dict(value)
        value["schema"] = replacements[schema]
    return value


def _rename_public_slot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            (key.replace("P11", "P12") if isinstance(key, str) else key): _rename_public_slot(
                child
            )
            for key, child in value.items()
            if key != "self_sha256"
        }
    if isinstance(value, list):
        return [_rename_public_slot(child) for child in value]
    if isinstance(value, str):
        return value.replace("P11", "P12")
    return value


@contextmanager
def _patched_controller() -> Iterator[dict[str, Any]]:
    bright_runtime = p11_runtime.train.bright_runtime
    original_run_qwen = bright_runtime._run_qwen
    original_load_pack = p11_runtime._load_pack
    original_self_hashed = acquisition.self_hashed
    totalized_state: dict[str, Any] = {}

    def run_qwen(base: Path, root: Path, items: Sequence[Any]):
        output, receipt = original_run_qwen(base, root, items)
        totalized, audit = totalize_qwen_output(output, items)
        audit_path = root / "qwen.totalized.json"
        p11_runtime.train.bright_runtime._write_json(audit_path, audit)
        totalized_state.update(
            {
                "file_sha256": acquisition.file_sha256(audit_path),
                "pack_sha256": audit["pack_sha256"],
                "source_valid_generation_count": audit[
                    "source_valid_generation_count"
                ],
                "totalized_generation_count": audit["totalized_generation_count"],
            }
        )
        updated_receipt = dict(receipt)
        updated_receipt.update(totalized_state)
        updated_receipt["valid_generation_count"] = len(items)
        return totalized, updated_receipt

    def self_hashed(value: Mapping[str, Any], field: str = "self_sha256"):
        body = dict(value)
        if body.get("schema") == INTENT_SCHEMA:
            if not totalized_state:
                raise p11_runtime.NanoBEIRCConfirmError(
                    "typed-query totalization seal is absent"
                )
            body["typed_query_totalized_pack_sha256"] = totalized_state[
                "pack_sha256"
            ]
        return original_self_hashed(body, field=field)

    replacements = {
        "SCHEMA": SCHEMA,
        "ATTEMPT_SCHEMA": ATTEMPT_SCHEMA,
        "INTENT_SCHEMA": INTENT_SCHEMA,
        "ACTION_SCHEMA": ACTION_SCHEMA,
        "FREEZE_SCHEMA": FREEZE_SCHEMA,
        "RUN_ROOT_RELATIVE": RUN_ROOT_RELATIVE,
        "RESULT_RELATIVE": INTERNAL_RESULT_RELATIVE,
        "FREEZE_RELATIVE": FREEZE_RELATIVE,
        "IMPLEMENTATION_RELATIVE": IMPLEMENTATION_RELATIVE,
        "TEST_RELATIVE": TEST_RELATIVE,
        "ACQUISITION_RESULT_RELATIVE": ACQUISITION_RESULT_RELATIVE,
        "ACQUISITION_RESULT_FILE_SHA256": ACQUISITION_RESULT_FILE_SHA256,
        "ACQUISITION_RESULT_SELF_SHA256": ACQUISITION_RESULT_SELF_SHA256,
        "CANDIDATE_FREEZE_SELF_SHA256": CANDIDATE_FREEZE_SELF_SHA256,
        "STUDY_DESIGN_SELF_SHA256": STUDY_DESIGN_SELF_SHA256,
        "acquisition": acquisition,
        "_load_acquisition": _load_acquisition,
    }
    original_runtime = {name: getattr(p11_runtime, name) for name in replacements}
    original_candidate_name = p11_runtime.p11.CANDIDATE_NAME
    acquisition_aliases = {
        "NanoBEIRAcquisitionError": getattr(
            acquisition, "NanoBEIRAcquisitionError", None
        ),
        "verify_self_hash": getattr(acquisition, "verify_self_hash", None),
    }
    try:
        for name, replacement in replacements.items():
            setattr(p11_runtime, name, replacement)
        p11_runtime._load_pack = lambda base, binding, name: _replace_pack_schema(
            original_load_pack, base, binding, name
        )
        p11_runtime.p11.CANDIDATE_NAME = CANDIDATE_NAME
        acquisition.NanoBEIRAcquisitionError = acquisition.NanoBEIRP12AcquisitionError
        acquisition.verify_self_hash = acquisition._verify_self_hash
        acquisition.self_hashed = self_hashed
        bright_runtime._run_qwen = run_qwen
        yield totalized_state
    finally:
        bright_runtime._run_qwen = original_run_qwen
        acquisition.self_hashed = original_self_hashed
        for name, value in acquisition_aliases.items():
            if value is None:
                delattr(acquisition, name)
            else:
                setattr(acquisition, name, value)
        p11_runtime.p11.CANDIDATE_NAME = original_candidate_name
        p11_runtime._load_pack = original_load_pack
        for name, value in original_runtime.items():
            setattr(p11_runtime, name, value)


def run_formal(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    root = base / RUN_ROOT_RELATIVE
    result_path = base / RESULT_RELATIVE
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("P12 C_confirm root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("P12 C_confirm result already exists")
    with _patched_controller() as totalized_state:
        internal = p11_runtime.run_formal(project_root)
    if not totalized_state:
        raise NanoBEIRP12CConfirmError("totalization receipt is absent")
    body = _rename_public_slot(internal)
    execution = body.get("execution")
    if not isinstance(execution, Mapping):
        raise NanoBEIRP12CConfirmError("internal execution receipt drifted")
    body["execution"] = {
        **execution,
        "typed_query_totalization": dict(totalized_state),
    }
    body["recorded_date"] = "2026-07-21"
    body["internal_action_slot_alias"] = {
        "legacy_slot": "P11",
        "public_candidate": "P12",
        "ranker_weights_changed": False,
    }
    result = acquisition.self_hashed(body)
    p11_runtime.train.bright_runtime._write_json(result_path, result, mode=0o644)
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
