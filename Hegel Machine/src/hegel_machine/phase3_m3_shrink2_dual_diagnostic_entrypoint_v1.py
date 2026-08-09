"""Direct target-free host entrypoint for shrink-step-2 dual replay.

Execute this file by path under ``python -I -S -B``.  It creates a minimal
``hegel_machine`` package shell, so the package initializer cannot expose target,
split, seed, role, or evaluator material to the host replay process.
"""

from __future__ import annotations

import sys


if __package__ not in {None, ""}:
    raise RuntimeError("host entrypoint must be executed directly by file path")
if sys.flags.isolated != 1 or sys.flags.no_site != 1 or not sys.dont_write_bytecode:
    raise RuntimeError("host entrypoint requires python -I -S -B")
if any(name.startswith("hegel_machine.") for name in sys.modules):
    raise RuntimeError("hegel_machine project modules were loaded before host isolation")

import argparse
import json
from pathlib import Path
from types import ModuleType
from typing import NoReturn, Sequence


_PACKAGE_DIRECTORY = Path(__file__).resolve().parent
_package = ModuleType("hegel_machine")
_package.__path__ = [str(_PACKAGE_DIRECTORY)]  # type: ignore[attr-defined]
_package.__package__ = "hegel_machine"
sys.modules["hegel_machine"] = _package
__package__ = "hegel_machine"

from . import phase3_m3_shrink2_dual_diagnostic_v1 as _validator


_EXPECTED_PROJECT_MODULES = frozenset(
    {
        "hegel_machine.phase3_m3_bounded_enumerator_shrink2_v1",
        "hegel_machine.phase3_m3_bounded_enumerator_v1",
        "hegel_machine.phase3_m3_dsl_core_v1",
        "hegel_machine.phase3_m3_record_wire_v1",
        "hegel_machine.phase3_m3_shrink1_core_v1",
        "hegel_machine.phase3_m3_shrink2_core_v1",
        "hegel_machine.phase3_m3_shrink2_diagnostic_profile_v1",
        "hegel_machine.phase3_m3_shrink2_dual_diagnostic_v1",
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
    raise RuntimeError(f"FAIL_SHRINK2_TARGET_FREE_HOST_ENTRYPOINT: {detail}")


def _assert_module_closure() -> tuple[str, ...]:
    loaded = frozenset(
        name for name in sys.modules if name.startswith("hegel_machine.")
    )
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
    if loaded != _validator._EXPECTED_HOST_PROJECT_MODULES:
        _fail("entrypoint and validator module allowlists disagree")
    return tuple(sorted(loaded))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hegel-python-m3-shrink2-target-free-host-replay-v1"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--host-self-check", action="store_true")
    mode.add_argument("--validate-dual", action="store_true")
    parser.add_argument("--python-output-directory", type=Path)
    parser.add_argument("--rust-output-directory", type=Path)
    return parser


def _self_check(loaded: tuple[str, ...]) -> dict[str, object]:
    return {
        "schema_version": "hegel-m3-shrink2-target-free-host-self-check/1",
        "profile_id": _validator.PROFILE_ID,
        "claim_level": _validator.CLAIM_LEVEL,
        "diagnostic_only": True,
        "authoritative_claim_allowed": False,
        "execution_state": "NOT_RUN",
        "formal_roots_generated": False,
        "formal_roots": None,
        "dual_replay_executed": False,
        "loaded_hegel_modules": list(loaded),
        "target_free_isolation_verified": True,
        "target_or_split_modules_loaded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    loaded_before = _assert_module_closure()
    parser = _parser()
    args = parser.parse_args(argv)
    if args.host_self_check:
        if (
            args.python_output_directory is not None
            or args.rust_output_directory is not None
        ):
            parser.error("--host-self-check takes no output directories")
        report = _self_check(loaded_before)
    else:
        if (
            args.python_output_directory is None
            or args.rust_output_directory is None
        ):
            parser.error("--validate-dual requires both output directories")
        report = _validator.validate_shrink2_dual_diagnostic_v1(
            args.python_output_directory,
            args.rust_output_directory,
        )
    loaded_after = _assert_module_closure()
    if loaded_after != loaded_before:
        _fail("project dependency closure changed before receipt publication")
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
