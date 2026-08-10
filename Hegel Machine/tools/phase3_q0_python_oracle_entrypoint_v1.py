#!/usr/bin/env python3
"""Isolated JSON endpoint for the Phase-3A-Q0 Python oracle.

The script is deliberately self-bootstrapping so Docker may execute it with
``python -I -S -B`` against a read-only source snapshot.  It emits one compact
JSON object and never reads target truth, split material, or formal M3 roots.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

# The normal package initializer exposes historical target and split APIs.
# Q0 is target-blind, so install only an empty package namespace and import the
# two source-frozen endpoint modules directly beneath it.  The isolated host
# also supplies a source snapshot that contains no target or split modules.
if "hegel_machine" not in sys.modules:
    package = ModuleType("hegel_machine")
    package.__path__ = [str(SOURCE_ROOT / "hegel_machine")]  # type: ignore[attr-defined]
    package.__package__ = "hegel_machine"
    sys.modules["hegel_machine"] = package

from hegel_machine import phase3_q0_quotient_contract_v1 as contract  # noqa: E402
from hegel_machine import phase3_q0_quotient_oracle_v1 as oracle  # noqa: E402


SCHEMA_VERSION = "hegel-q0-python-micro-oracle/1"
ERROR_SCHEMA_VERSION = "hegel-q0-python-micro-oracle-error/1"
SOURCE_PATHS = (
    "src/hegel_machine/phase3_q0_input_adapter_v1.py",
    "src/hegel_machine/phase3_q0_evaluator_v1.py",
    "src/hegel_machine/phase3_q0_quotient_contract_v1.py",
    "src/hegel_machine/phase3_q0_quotient_oracle_v1.py",
)


def _root_id(value: bytes) -> str:
    if type(value) is not bytes or len(value) != 32:
        raise TypeError("root must contain exactly 32 bytes")
    return "sha256:" + value.hex()


def _source_root() -> str:
    digest = sha256()
    for relative in SOURCE_PATHS:
        payload = (PROJECT_ROOT / relative).read_bytes()
        path_bytes = relative.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(4, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return _root_id(digest.digest())


def _coverage(rows: tuple[tuple[int, int, int, int, int, int], ...]) -> list[dict[str, int]]:
    return [
        {
            "operator_code": row[0],
            "eligible_raw": row[1],
            "strict_admitted": row[2],
            "rewrite_collapses": row[3],
            "canonical_duplicates": row[4],
            "new_canonical": row[5],
        }
        for row in rows
    ]


def _rounds(rows: tuple[oracle.RoundDeltaV1, ...]) -> list[dict[str, object]]:
    return [
        {
            "round_index": row.round_index,
            "queued_application_count": row.queued_application_count,
            "new_canonical_program_count": row.new_canonical_program_count,
            "new_behavior_class_count": row.new_behavior_class_count,
            "frontier_mutation_count": row.frontier_mutation_count,
            "cohort_bank_mutation_count": row.bank_mutation_count,
            "complete_state_changed": row.complete_state_changed,
        }
        for row in rows
    ]


def endpoint_object() -> dict[str, object]:
    result = oracle.run_q0_python_oracle_v1()
    probe = contract.Q0ProbeInputV1()
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_id": oracle.ENDPOINT_IMPLEMENTATION_ID.decode("ascii"),
        "terminal_status": result.endpoint_status,
        "dsl_version": contract.DSL_VERSION,
        "dsl_freeze_version": contract.DSL_FREEZE_VERSION,
        "closure_semantics_version": contract.CLOSURE_SEMANTICS_VERSION,
        "q0_freeze_version": contract.Q0_FREEZE_VERSION,
        "projection_id": contract.Q0_PROJECTION_ID,
        "probe_input_signature_id": contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        "probe_canonical_cbor_hex": probe.canonical_bytes.hex(),
        "probe_universe_root": _root_id(probe.universe_root),
        "frozen_leaf_count": len(oracle.Q0_FROZEN_LEAF_SEEDS),
        "canonical_syntax_count": result.canonical_syntax_program_count,
        "syntax_raw_operator_applications": result.syntax_raw_application_count,
        "quotient_raw_operator_applications": result.quotient_raw_application_count,
        "syntax_strict_admitted_applications": (
            result.strict_admitted_syntax_application_count
        ),
        "quotient_strict_admitted_applications": (
            result.strict_admitted_quotient_application_count
        ),
        "syntax_rewrite_collapses": result.rewrite_collapse_syntax_count,
        "quotient_rewrite_collapses": result.rewrite_collapse_quotient_count,
        "behavior_class_count": result.behavior_class_count,
        "frontier_point_count": result.frontier_point_count,
        "maximum_frontier_size": result.maximum_frontier_points_per_class,
        "syntax_continuation_bank_point_count": (
            result.syntax_continuation_bank_point_count
        ),
        "quotient_continuation_bank_point_count": (
            result.quotient_continuation_bank_point_count
        ),
        "maximum_syntax_bank_points_per_class": (
            result.maximum_syntax_bank_points_per_class
        ),
        "maximum_quotient_bank_points_per_class": (
            result.maximum_quotient_bank_points_per_class
        ),
        "syntax_saturation_rounds": result.saturation_round_count,
        "direct_saturation_rounds": result.saturation_round_count,
        "work_queue_empty": result.work_queue_empty,
        "zero_delta_full_round": result.zero_delta_full_round,
        "all_typed_operator_frontier_tuples_covered": result.work_queue_empty,
        "exhaustive_syntax_oracle_complete": result.zero_delta_full_round,
        "syntax_direct_states_equal": (
            result.syntax_class_archive_root == result.direct_class_archive_root
        ),
        "final_class_delta": result.final_class_delta,
        "final_frontier_delta": result.final_frontier_mutation_delta,
        "final_bank_delta": result.final_bank_mutation_delta,
        "projection_manifest_root": _root_id(result.projection_manifest_root),
        "semantic_binding_root": _root_id(result.semantic_binding_root),
        "syntax_program_root": _root_id(result.syntax_program_archive_root),
        "syntax_class_archive_root": _root_id(result.syntax_class_archive_root),
        "direct_class_archive_root": _root_id(result.direct_class_archive_root),
        "syntax_state_root": _root_id(result.syntax_state_root),
        "direct_state_root": _root_id(result.direct_state_root),
        "syntax_saturation_state_preimage_cbor_hex": (
            result.syntax_saturation_state_preimage_bytes.hex()
        ),
        "direct_saturation_state_preimage_cbor_hex": (
            result.direct_saturation_state_preimage_bytes.hex()
        ),
        "syntax_coverage_root": _root_id(result.syntax_operator_coverage_root),
        "direct_coverage_root": _root_id(result.quotient_operator_coverage_root),
        "syntax_coverage": _coverage(result.syntax_coverage_records),
        "direct_coverage": _coverage(result.quotient_coverage_records),
        "direct_rounds": _rounds(result.round_deltas),
        "python_source_root": _source_root(),
        "endpoint_state_root": _root_id(result.endpoint_state_root),
        "endpoint_state_cbor_hex": result.canonical_state_bytes.hex(),
        "resource_guards_ok": result.all_guards_respected,
        "target_truth_accessed": result.target_truth_accessed,
        "split_accessed": result.split_accessed,
        "role_evaluation_performed": result.role_evaluation_performed,
        "formal_roots_generated": result.formal_roots_generated,
        "authority_claimed": result.authoritative_claim_allowed,
    }


def main() -> int:
    try:
        payload = endpoint_object()
        output = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(output.encode("ascii")) + 1 > contract.Q0_MAX_OUTPUT_BYTES:
            raise oracle.Q0OracleError(
                "INCONCLUSIVE_RESOURCE_LIMIT",
                "diagnostic output-byte guard reached",
                guard_id=contract.Q0ResourceGuardId.OUTPUT_BYTES,
            )
    except oracle.Q0OracleError as error:
        guard_id = error.guard_id
        payload = {
            "schema_version": ERROR_SCHEMA_VERSION,
            "status": error.code,
            "error_code": error.code,
            "resource_guard_id": (
                None if guard_id is None else int(guard_id)
            ),
            "detail": error.detail,
            "authority_claimed": False,
        }
        output = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        print(output)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
