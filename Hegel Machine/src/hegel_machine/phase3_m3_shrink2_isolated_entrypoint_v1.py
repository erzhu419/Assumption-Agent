"""Direct, target-free entrypoint for the shrink-step-2 dual diagnostic.

This file must be executed by path under ``python -I -S -B``.  It installs a
minimal package shell instead of importing ``hegel_machine.__init__`` and then
enforces an exact project-module closure before and after enumeration.
"""

from __future__ import annotations

import sys


if __package__ not in {None, ""}:
    raise RuntimeError("entrypoint must be executed directly by file path")
if sys.flags.isolated != 1 or sys.flags.no_site != 1 or not sys.dont_write_bytecode:
    raise RuntimeError("entrypoint requires python -I -S -B")
if any(name.startswith("hegel_machine.") for name in sys.modules):
    raise RuntimeError("hegel_machine project modules were loaded before isolation")

import argparse
import json
import os
from pathlib import Path
from types import ModuleType
from typing import NoReturn, Sequence


_PACKAGE_DIRECTORY = Path(__file__).resolve().parent
_package = ModuleType("hegel_machine")
_package.__path__ = [str(_PACKAGE_DIRECTORY)]  # type: ignore[attr-defined]
_package.__package__ = "hegel_machine"
sys.modules["hegel_machine"] = _package
__package__ = "hegel_machine"

from . import phase3_m3_bounded_enumerator_shrink2_v1 as _enumerator
from . import phase3_m3_shrink2_diagnostic_profile_v1 as _profile


_EXPECTED_PROJECT_MODULES = frozenset(
    {
        "hegel_machine.phase3_m3_bounded_enumerator_shrink2_v1",
        "hegel_machine.phase3_m3_bounded_enumerator_v1",
        "hegel_machine.phase3_m3_dsl_core_v1",
        "hegel_machine.phase3_m3_record_wire_v1",
        "hegel_machine.phase3_m3_shrink1_core_v1",
        "hegel_machine.phase3_m3_shrink2_core_v1",
        "hegel_machine.phase3_m3_shrink2_diagnostic_profile_v1",
        "hegel_machine.strict_ast_shrink1_v1",
        "hegel_machine.strict_ast_shrink2_v1",
        "hegel_machine.strict_ast_v1",
        "hegel_machine.strict_cbor_v1",
    }
)
_FORBIDDEN_MODULE_FRAGMENTS = (
    "_evaluator",
    "_odd",
    "_role",
    "_seed",
    "_sink",
    "_split_",
    "_target",
)


def _fail(detail: str) -> NoReturn:
    raise RuntimeError(f"FAIL_SHRINK2_TARGET_FREE_ENTRYPOINT: {detail}")


def _loaded_project_modules() -> frozenset[str]:
    return frozenset(
        name for name in sys.modules if name.startswith("hegel_machine.")
    )


def _assert_module_closure() -> tuple[str, ...]:
    loaded = _loaded_project_modules()
    forbidden = sorted(
        name
        for name in loaded
        if any(fragment in name for fragment in _FORBIDDEN_MODULE_FRAGMENTS)
    )
    if forbidden:
        _fail(f"target/split/seed/role dependency loaded: {forbidden!r}")
    if loaded != _EXPECTED_PROJECT_MODULES:
        _fail(
            "dependency closure drift; "
            f"missing={sorted(_EXPECTED_PROJECT_MODULES - loaded)!r}; "
            f"unexpected={sorted(loaded - _EXPECTED_PROJECT_MODULES)!r}"
        )
    return tuple(sorted(loaded))


def _root(value: str) -> bytes:
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
    parser = argparse.ArgumentParser(
        prog="hegel-python-m3-shrink2-target-free-diagnostic-v1"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--target-free-self-check", action="store_true")
    mode.add_argument("--enumerate-diagnostic", action="store_true")
    parser.add_argument("--child-dsl-spec-root", type=_root)
    parser.add_argument("--operator-semantics-root", type=_root)
    parser.add_argument("--identifier-registry-root", type=_root)
    parser.add_argument("--output-directory", type=Path)
    return parser


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


def _framed(records: Sequence[tuple[object, ...]]) -> bytes:
    output = bytearray()
    encode = _enumerator._parent.canonical_cbor_encode
    for record in records:
        encoded = encode(record)
        if len(encoded) > 0xFFFFFFFF:
            _fail("formal record exceeds uint32")
        output.extend(len(encoded).to_bytes(4, "big"))
        output.extend(encoded)
    return bytes(output)


def _write_artifacts(
    directory: Path,
    result: _enumerator.BoundedEnumerationResultV1,
    report: dict[str, object],
) -> None:
    directory.mkdir(mode=0o755, parents=False, exist_ok=False)
    _exclusive_write(
        directory / "canonical_program_records.cborframed",
        _framed(result.canonical_program_records),
    )
    _exclusive_write(
        directory / "program_chunk_manifests.cborframed",
        _framed(result.program_chunk_manifests),
    )
    _exclusive_write(
        directory / "bucket_accounting_records.cborframed",
        _framed(result.bucket_accounting_records),
    )
    _exclusive_write(
        directory / "report.json",
        (
            json.dumps(report, sort_keys=True, indent=2, separators=(",", ": "))
            + "\n"
        ).encode("utf-8"),
    )


def _augment_report(
    report: dict[str, object], loaded: tuple[str, ...]
) -> dict[str, object]:
    roots = _profile.diagnostic_root_hex_v1()
    report.update(
        {
            "claim_level": _profile.CLAIM_LEVEL,
            "binding_profile_id": _profile.BINDING_PROFILE_ID,
            "profile_id": _profile.PROFILE_ID,
            "implementation_id": 1,
            "implementation_machine_id": (
                "hegel-python-m3-shrink2-complete-closure-diagnostic-v1"
            ),
            "canonicalizer_profile": "hegel-canonical-ast-v1",
            "mdl_code_table_id": "hegel-mdl-prefix-v1.0.0",
            "closure_status_id": (
                2 if report["closure_status"] == "DSL_TOO_LARGE" else 1
            ),
            "maximum_canonical_programs": _enumerator.CANONICAL_PROGRAM_BUDGET,
            "maximum_raw_operator_applications": _enumerator.RAW_APPLICATION_CAP,
            "maximum_ast_depth": 4,
            "maximum_ast_node_count": 7,
            "formal_bucket_count": 175,
            "raw_expansion_limit_hit": False,
            "wall_clock_abort_hit": False,
            "aliases_excluded_before_count": [
                "greater_equal",
                "approx_equal:tolerance=0",
            ],
            "active_aggregate_map_ids": [0, 1, 5],
            "tombstoned_aggregate_map_ids": [2, 3, 4],
            "active_rational_parameter_ids": [1, 3, 5],
            "tombstoned_rational_parameter_ids": [0, 2, 4, 6],
            "reserved_rational_parameter_ids": [7],
            "child_dsl_spec_root": roots["child_dsl_spec_root"],
            "operator_semantics_root": roots["operator_semantics_root"],
            "identifier_registry_root": roots["identifier_registry_root"],
            "canonical_ast_schema_root": roots["canonical_ast_schema_root"],
            "canonical_cbor_profile_root": roots["canonical_cbor_profile_root"],
            "canonical_program_archive_root_or_null": report[
                "canonical_program_archive_root"
            ],
            "program_chunk_manifest_root_or_null": report[
                "program_chunk_manifest_root"
            ],
            "bucket_accounting_root_or_null": report["bucket_accounting_root"],
            "first_out_of_budget_program_hash_or_null": report[
                "first_out_of_budget_program_hash_or_null"
            ],
            "first_out_of_budget_program_cbor_hex_or_null": report[
                "first_out_of_budget_program_cbor_hex_or_null"
            ],
            "first_out_of_budget_program_ordinal_or_null": report[
                "first_out_of_budget_ordinal_or_null"
            ],
            "loaded_hegel_modules": list(loaded),
            "target_free_isolation_verified": True,
            "target_or_split_modules_loaded": False,
        }
    )
    for legacy_alias in (
        "bucket_accounting_root",
        "canonical_program_archive_root",
        "canonical_program_budget",
        "diagnostic_child_dsl_spec_root",
        "diagnostic_identifier_registry_root",
        "diagnostic_operator_semantics_root",
        "first_out_of_budget_ordinal_or_null",
        "program_chunk_manifest_root",
        "raw_operator_application_cap",
    ):
        del report[legacy_alias]
    return report


def _self_check_report(loaded: tuple[str, ...]) -> dict[str, object]:
    return {
        "schema_version": "hegel-m3-shrink2-python-target-free-self-check/1",
        "profile_id": _profile.PROFILE_ID,
        "claim_level": _profile.CLAIM_LEVEL,
        "diagnostic_only": True,
        "authoritative_claim_allowed": False,
        "execution_state": "NOT_RUN",
        "formal_roots_generated": False,
        "formal_roots": None,
        "complete_closure_enumerated": False,
        "loaded_hegel_modules": list(loaded),
        "target_free_isolation_verified": True,
        "target_or_split_modules_loaded": False,
        **_profile.diagnostic_root_hex_v1(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    loaded_before = _assert_module_closure()
    parser = _parser()
    args = parser.parse_args(argv)
    if args.target_free_self_check:
        if any(
            value is not None
            for value in (
                args.child_dsl_spec_root,
                args.operator_semantics_root,
                args.identifier_registry_root,
                args.output_directory,
            )
        ):
            parser.error("--target-free-self-check takes no other arguments")
        report = _self_check_report(loaded_before)
    else:
        supplied = (
            args.child_dsl_spec_root,
            args.operator_semantics_root,
            args.identifier_registry_root,
        )
        if any(value is None for value in supplied) or args.output_directory is None:
            parser.error(
                "--enumerate-diagnostic requires all three roots and "
                "--output-directory"
            )
        if supplied != _profile.NON_FORMAL_SYNTHETIC_CHILD_BINDINGS:
            parser.error("roots differ from NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1")
        bindings = _enumerator.EnumerationBindingsV1(*supplied)
        result = _enumerator.enumerate_bounded_closure_shrink2_v1(bindings)
        if _assert_module_closure() != loaded_before:
            _fail("project dependency closure changed during enumeration")
        report = _augment_report(
            _enumerator.diagnostic_report_shrink2_v1(
                result,
                bindings,
                loaded_hegel_modules=loaded_before,
            ),
            loaded_before,
        )
        _write_artifacts(args.output_directory, result, report)
    if _assert_module_closure() != loaded_before:
        _fail("project dependency closure changed before report publication")
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, _enumerator.BoundedEnumerationError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
