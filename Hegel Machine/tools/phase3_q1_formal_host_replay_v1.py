#!/usr/bin/env python3
"""Direct offline entrypoint for the trusted-host Q1 formal archive replay."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import types
from typing import NoReturn, Sequence


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "hegel_machine"
EXPECTED_MODULE_SOURCE_PATHS = (
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink2_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink3_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink4_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink5_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink6_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_v1.py",
    "src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "src/hegel_machine/phase3_m3_record_wire_v1.py",
    "src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink2_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink3_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink4_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink5_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink6_core_v1.py",
    "src/hegel_machine/phase3_q0_evaluator_v1.py",
    "src/hegel_machine/phase3_q0_input_adapter_v1.py",
    "src/hegel_machine/phase3_q1_archive_projection_v1.py",
    "src/hegel_machine/phase3_q1_capacity_preflight_v1.py",
    "src/hegel_machine/phase3_q1_external_sort_profile_v1.py",
    "src/hegel_machine/phase3_q1_formal_archive_contract_v1.py",
    "src/hegel_machine/phase3_q1_formal_host_replay_v1.py",
    "src/hegel_machine/phase3_q1_partition_snapshot_v1.py",
    "src/hegel_machine/phase3_q1_quotient_contract_v1.py",
    "src/hegel_machine/phase3_q1_universe_v1.py",
    "src/hegel_machine/strict_ast_shrink1_v1.py",
    "src/hegel_machine/strict_ast_shrink2_v1.py",
    "src/hegel_machine/strict_ast_shrink3_v1.py",
    "src/hegel_machine/strict_ast_shrink4_v1.py",
    "src/hegel_machine/strict_ast_shrink5_v1.py",
    "src/hegel_machine/strict_ast_shrink6_v1.py",
    "src/hegel_machine/strict_ast_v1.py",
    "src/hegel_machine/strict_cbor_v1.py",
)


class Q1FormalHostReplayCliError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1FormalHostReplayCliError(code, detail)


def _loaded_project_modules() -> tuple[str, ...]:
    expected = {
        "hegel_machine." + Path(relative).stem
        for relative in EXPECTED_MODULE_SOURCE_PATHS
    }
    loaded: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("hegel_machine."):
            continue
        source = getattr(module, "__file__", None)
        if type(source) is not str:
            _fail("Q1_FORMAL_HOST_MODULE_CLOSURE", f"{name} has no source")
        try:
            relative = Path(source).resolve().relative_to(PROJECT_ROOT).as_posix()
        except (OSError, ValueError):
            _fail("Q1_FORMAL_HOST_MODULE_CLOSURE", f"{name} escaped project")
        if name not in expected or relative not in EXPECTED_MODULE_SOURCE_PATHS:
            _fail(
                "Q1_FORMAL_HOST_MODULE_CLOSURE",
                f"unexpected project module {name}",
            )
        loaded.add(name)
    if loaded != expected:
        _fail(
            "Q1_FORMAL_HOST_MODULE_CLOSURE",
            f"module closure differs; missing={sorted(expected - loaded)!r}",
        )
    return tuple(sorted(loaded))


def _load_core():
    """Load the target-blind package closure without executing ``__init__``."""

    if not PACKAGE_ROOT.is_dir():
        _fail("Q1_FORMAL_HOST_CORE_UNAVAILABLE", "package source root differs")
    package = types.ModuleType("hegel_machine")
    package.__package__ = "hegel_machine"
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["hegel_machine"] = package
    try:
        core = importlib.import_module(
            "hegel_machine.phase3_q1_formal_host_replay_v1"
        )
    except (ImportError, OSError) as error:
        _fail("Q1_FORMAL_HOST_CORE_UNAVAILABLE", str(error))
    _loaded_project_modules()
    return core


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail("Q1_FORMAL_HOST_RESULT_JSON", str(error))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "replay completed Python and Rust Q1 formal archive directories; "
            "never enumerate, publish state, or start Q1"
        )
    )
    parser.add_argument("--python-root", required=True)
    parser.add_argument("--rust-root", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--capsule-digest", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-genesis-root", required=True)
    parser.add_argument("--prestart-admission-receipt-root", required=True)
    return parser


_CORE = _load_core()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = _CORE.dual_formal_host_replay_v1(
            Path(args.python_root),
            Path(args.rust_root),
            expected_action_id=args.action,
            expected_capsule_digest=args.capsule_digest,
            expected_source_commit=args.source_commit,
            expected_run_genesis_root=args.run_genesis_root,
            expected_prestart_admission_receipt_root=(
                args.prestart_admission_receipt_root
            ),
        )
        sys.stdout.buffer.write(_canonical_json_bytes(value))
        sys.stdout.buffer.flush()
        return 0
    except Q1FormalHostReplayCliError as error:
        print(f"{error.code}: {error.detail}", file=sys.stderr)
        return 2
    except Exception as error:
        if (
            error.__class__.__name__ == "Q1FormalHostReplayError"
            and hasattr(error, "code")
            and hasattr(error, "detail")
        ):
            print(f"{error.code}: {error.detail}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
