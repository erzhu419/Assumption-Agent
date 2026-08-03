"""Machine-readable CLI for the Python M3 bounded closure enumerator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .phase3_m3_bounded_enumerator_v1 import (
    CANONICAL_PROGRAM_BUDGET,
    RAW_APPLICATION_CAP,
    BoundedEnumerationError,
    BoundedEnumerationResultV1,
    EnumerationBindingsV1,
    enumerate_bounded_closure_v1,
)
from .strict_cbor_v1 import canonical_cbor_encode


SCHEMA = "hegel-m3-python-closure-enumerator-report/1"
IMPLEMENTATION_MACHINE_ID = "hegel-python-m3-bounded-closure-enumerator-v1"
FREEZE_VERSION = "hegel-freeze-p2b-p3-v1.1.2"


def _hex_root(value: str) -> bytes:
    source = value.removeprefix("0x")
    if len(source) != 64:
        raise argparse.ArgumentTypeError("root must be exactly 64 hexadecimal digits")
    try:
        result = bytes.fromhex(source)
    except ValueError as error:
        raise argparse.ArgumentTypeError("root contains non-hexadecimal input") from error
    if len(result) != 32:
        raise argparse.ArgumentTypeError("root must decode to exactly 32 bytes")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hegel-python-m3-bounded-enumerator-v1")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--binding-material", action="store_true")
    modes.add_argument("--enumerate-prefix", action="store_true")
    parser.add_argument("--child-dsl-spec-root", type=_hex_root)
    parser.add_argument("--operator-semantics-root", type=_hex_root)
    parser.add_argument("--identifier-registry-root", type=_hex_root)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--diagnostic-canonical-budget", type=int)
    parser.add_argument("--diagnostic-raw-application-cap", type=int)
    return parser


def binding_material() -> dict[str, object]:
    return {
        "schema_version": "hegel-m3-python-enumerator-binding-material/1",
        "implementation": "python",
        "implementation_id": 1,
        "implementation_machine_id": IMPLEMENTATION_MACHINE_ID,
        "entrypoint": "python3 phase3_m3_isolated_entrypoint_v1.py",
        "source_paths": [
            "Hegel Machine/src/hegel_machine/phase3_m3_isolated_entrypoint_v1.py",
            "Hegel Machine/src/hegel_machine/phase3_m3_bounded_enumerator_v1.py",
            "Hegel Machine/src/hegel_machine/phase3_m3_bounded_enumerator_cli_v1.py",
            "Hegel Machine/src/hegel_machine/phase3_m3_dsl_core_v1.py",
            "Hegel Machine/src/hegel_machine/phase3_m3_shrink1_core_v1.py",
            "Hegel Machine/src/hegel_machine/phase3_m3_record_wire_v1.py",
            "Hegel Machine/src/hegel_machine/strict_ast_v1.py",
            "Hegel Machine/src/hegel_machine/strict_ast_shrink1_v1.py",
            "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
        ],
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
    }


def result_report(
    result: BoundedEnumerationResultV1,
    bindings: EnumerationBindingsV1,
    *,
    canonical_budget: int,
    raw_cap: int,
) -> dict[str, object]:
    exact = canonical_budget == CANONICAL_PROGRAM_BUDGET and raw_cap == RAW_APPLICATION_CAP
    return {
        "schema_version": SCHEMA,
        "claim_level": (
            "FORMAL_PROFILE_CANDIDATE_NOT_AUTHORITY"
            if exact
            else "NON_FORMAL_DIAGNOSTIC_TEST_PROFILE"
        ),
        "authoritative_claim_allowed": False,
        "implementation": "python",
        "implementation_id": 1,
        "implementation_machine_id": IMPLEMENTATION_MACHINE_ID,
        "dsl_version": result.dsl_version,
        "freeze_version": FREEZE_VERSION,
        "canonicalizer_profile": "hegel-canonical-ast-v1",
        "mdl_code_table_id": "hegel-mdl-prefix-v1.0.0",
        "closure_status": result.closure_status,
        "closure_status_id": 2 if result.closure_status == "DSL_TOO_LARGE" else 1 if result.closure_status == "COMPLETE" else None,
        "raw_operator_application_count": result.raw_operator_application_count,
        "canonical_program_count": result.canonical_program_count,
        "closure_cardinality_or_null": result.canonical_program_count if result.closure_status == "COMPLETE" else None,
        "frontier_exhausted": result.closure_status == "COMPLETE",
        "all_type_buckets_closed": result.closure_status == "COMPLETE",
        "raw_expansion_limit_hit": False,
        "wall_clock_abort_hit": False,
        "canonical_program_archive_root_or_null": result.canonical_program_archive_root.hex(),
        "program_chunk_manifest_root_or_null": result.program_chunk_manifest_root.hex(),
        "bucket_accounting_root_or_null": result.bucket_accounting_root.hex(),
        "first_out_of_budget_program_hash_or_null": None if result.first_out_of_budget_program_hash is None else result.first_out_of_budget_program_hash.hex(),
        "first_out_of_budget_program_cbor_hex_or_null": None if result.first_out_of_budget_cbor is None else result.first_out_of_budget_cbor.hex(),
        "program_record_count": len(result.canonical_program_records),
        "chunk_manifest_count": len(result.program_chunk_manifests),
        "bucket_record_count": len(result.bucket_accounting_records),
        "records_per_chunk": 4096,
        "maximum_canonical_programs": canonical_budget,
        "maximum_raw_operator_applications": raw_cap,
        "traversal_prefix_complete": result.traversal_prefix_complete,
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
        "aliases_excluded_before_count": ["greater_equal", "approx_equal:tolerance=0"],
        "active_aggregate_map_ids": [0, 1, 5],
        "tombstoned_aggregate_map_ids": [2, 3, 4],
        "child_dsl_spec_root": bindings.child_dsl_spec_root.hex(),
        "operator_semantics_root": bindings.operator_semantics_root.hex(),
        "identifier_registry_root": bindings.identifier_registry_root.hex(),
    }


def _framed(records: Sequence[tuple[object, ...]]) -> bytes:
    output = bytearray()
    for record in records:
        encoded = canonical_cbor_encode(record)
        if len(encoded) > 0xFFFFFFFF:
            raise BoundedEnumerationError("FAIL_ENUMERATOR_OUTPUT", "formal record exceeds uint32")
        output.extend(len(encoded).to_bytes(4, "big"))
        output.extend(encoded)
    return bytes(output)


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_artifacts(
    directory: Path,
    result: BoundedEnumerationResultV1,
    report: dict[str, object],
) -> None:
    directory.mkdir(mode=0o755, parents=True, exist_ok=False)
    _exclusive_write(directory / "canonical_program_records.cborframed", _framed(result.canonical_program_records))
    _exclusive_write(directory / "program_chunk_manifests.cborframed", _framed(result.program_chunk_manifests))
    _exclusive_write(directory / "bucket_accounting_records.cborframed", _framed(result.bucket_accounting_records))
    _exclusive_write(
        directory / "report.json",
        (json.dumps(report, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode("utf-8"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.binding_material:
        forbidden = (
            args.child_dsl_spec_root,
            args.operator_semantics_root,
            args.identifier_registry_root,
            args.output_directory,
            args.diagnostic_canonical_budget,
            args.diagnostic_raw_application_cap,
        )
        if any(value is not None for value in forbidden):
            parser.error("--binding-material takes no enumeration arguments")
        print(json.dumps(binding_material(), sort_keys=True, indent=2))
        return 0
    roots = (
        args.child_dsl_spec_root,
        args.operator_semantics_root,
        args.identifier_registry_root,
    )
    if any(value is None for value in roots):
        parser.error("--enumerate-prefix requires all three binding roots")
    canonical_budget = (
        CANONICAL_PROGRAM_BUDGET
        if args.diagnostic_canonical_budget is None
        else args.diagnostic_canonical_budget
    )
    raw_cap = (
        RAW_APPLICATION_CAP
        if args.diagnostic_raw_application_cap is None
        else args.diagnostic_raw_application_cap
    )
    if (args.diagnostic_canonical_budget is None) != (args.diagnostic_raw_application_cap is None):
        parser.error("diagnostic budget and raw cap must be supplied together")
    bindings = EnumerationBindingsV1(*roots)
    result = enumerate_bounded_closure_v1(
        bindings,
        canonical_budget=canonical_budget,
        raw_application_cap=raw_cap,
    )
    report = result_report(result, bindings, canonical_budget=canonical_budget, raw_cap=raw_cap)
    if args.output_directory is not None:
        write_artifacts(args.output_directory, result, report)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BoundedEnumerationError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
