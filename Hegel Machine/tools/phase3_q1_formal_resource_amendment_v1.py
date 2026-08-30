#!/usr/bin/env python3
"""Read-only preflight for the post-v6 formal resource candidate.

The command reads exactly three small local files and emits one deterministic
JSON line.  It has no apply mode and cannot write an artifact, ledger or run
genesis, start a process, access the network, invoke an endpoint or start Q1.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import stat
import sys
from typing import Any, NoReturn, Sequence


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = (
    PROJECT_ROOT
    / "src/hegel_machine/phase3_q1_formal_resource_amendment_v1.py"
)
DEFAULT_ARTIFACT_PATH = (
    PROJECT_ROOT / "artifacts/phase3_q1_full_node6_capacity_preflight_v6.json"
)
DEFAULT_RUN_ROOT = Path(
    "/home/erzhu419/.local/state/hegel-machine/q1-full-node6-capacity-v6-run"
)
DEFAULT_PYTHON_STDOUT_PATH = DEFAULT_RUN_ROOT / "python.stdout"
DEFAULT_RUST_STDOUT_PATH = DEFAULT_RUN_ROOT / "rust.stdout"
MAXIMUM_INPUT_BYTES = 16 * 1024 * 1024


class FormalResourcePreflightCliError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise FormalResourcePreflightCliError(code, detail)


def _load_core() -> Any:
    name = "_hegel_q1_formal_resource_amendment_core_v1"
    spec = importlib.util.spec_from_file_location(name, CORE_PATH)
    if spec is None or spec.loader is None:
        _fail("Q1_RESOURCE_CORE_UNAVAILABLE", "core import specification differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError) as error:
        _fail("Q1_RESOURCE_CORE_UNAVAILABLE", str(error))
    return module


def _read_small_regular_file(path_text: str, label: str) -> bytes:
    path = Path(path_text)
    if not path.is_absolute():
        _fail("Q1_RESOURCE_INPUT_PATH", f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
        status = path.stat(follow_symlinks=False)
    except OSError as error:
        _fail("Q1_RESOURCE_INPUT_PATH", f"{label}: {error}")
    if path.is_symlink() or resolved != path or not stat.S_ISREG(status.st_mode):
        _fail(
            "Q1_RESOURCE_INPUT_PATH",
            f"{label} must be one canonical nonsymlink regular file",
        )
    if not 0 < status.st_size <= MAXIMUM_INPUT_BYTES:
        _fail("Q1_RESOURCE_INPUT_SIZE", f"{label} byte length differs")
    try:
        payload = path.read_bytes()
    except OSError as error:
        _fail("Q1_RESOURCE_INPUT_READ", f"{label}: {error}")
    if len(payload) != status.st_size:
        _fail("Q1_RESOURCE_INPUT_READ", f"{label} byte count changed while read")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "derive the non-authoritative Q1 formal resource candidate from "
            "the exact successful v6 small-file evidence"
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--preflight",
        action="store_true",
        help="read, validate and derive; never publish or start Q1",
    )
    parser.add_argument(
        "--artifact",
        default=str(DEFAULT_ARTIFACT_PATH),
        help="absolute path to the exact v6 success artifact",
    )
    parser.add_argument(
        "--python-stdout",
        default=str(DEFAULT_PYTHON_STDOUT_PATH),
        help="absolute path to the exact v6 Python endpoint stdout",
    )
    parser.add_argument(
        "--rust-stdout",
        default=str(DEFAULT_RUST_STDOUT_PATH),
        help="absolute path to the exact v6 Rust endpoint stdout",
    )
    return parser


_CORE = _load_core()


def preflight_v1(
    artifact_path: str,
    python_stdout_path: str,
    rust_stdout_path: str,
) -> dict[str, object]:
    """Read the three admitted files and return the pure candidate object."""

    artifact = _read_small_regular_file(artifact_path, "v6 artifact")
    python_stdout = _read_small_regular_file(
        python_stdout_path, "Python endpoint stdout"
    )
    rust_stdout = _read_small_regular_file(rust_stdout_path, "Rust endpoint stdout")
    return _CORE.derive_formal_resource_amendment_v1(
        artifact,
        python_stdout,
        rust_stdout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = preflight_v1(args.artifact, args.python_stdout, args.rust_stdout)
        sys.stdout.buffer.write(_CORE.canonical_json_bytes_v1(value))
        sys.stdout.buffer.flush()
        return 0
    except FormalResourcePreflightCliError as error:
        print(f"{error.code}: {error.detail}", file=sys.stderr)
        return 2
    except Exception as error:
        if (
            error.__class__.__name__ == "FormalResourceAmendmentError"
            and hasattr(error, "code")
            and hasattr(error, "detail")
        ):
            print(f"{error.code}: {error.detail}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
