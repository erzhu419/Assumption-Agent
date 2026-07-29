"""Execute one offline qualification worker under a frozen Landlock policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

if __package__:
    from .landlock_runtime import apply_landlock
else:
    from landlock_runtime import apply_landlock


SCHEMA = "hipporag_zero_weight_totality_landlock_exec_v1"
ENVIRONMENT_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "HF_DATASETS_OFFLINE",
        "HF_HOME",
        "HOME",
        "LANG",
        "LC_ALL",
        "MKL_NUM_THREADS",
        "MPLCONFIGDIR",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "SENTENCE_TRANSFORMERS_HOME",
        "TMPDIR",
        "TOKENIZERS_PARALLELISM",
        "TORCH_HOME",
        "TRANSFORMERS_OFFLINE",
        "VECLIB_MAXIMUM_THREADS",
        "XDG_CACHE_HOME",
    }
)


class LandlockExecError(RuntimeError):
    """The worker execution contract is malformed or cannot be installed."""


def _absolute_paths(value: Any, field: str) -> tuple[Path, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, str) for row in value)
    ):
        raise LandlockExecError(f"{field} is invalid")
    paths = tuple(Path(row) for row in value)
    if any(not path.is_absolute() for path in paths):
        raise LandlockExecError(f"{field} contains a relative path")
    return paths


def _read_spec(path: Path) -> Mapping[str, Any]:
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise LandlockExecError("execution spec metadata drifted")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LandlockExecError("execution spec is invalid") from exc
    if not isinstance(value, Mapping):
        raise LandlockExecError("execution spec is not an object")
    return value


def _validated_spec(path: Path) -> tuple[
    tuple[str, ...],
    Path,
    dict[str, str],
    tuple[Path, ...],
    tuple[Path, ...],
    tuple[Path, ...],
]:
    value = _read_spec(path)
    if set(value) != {
        "argv",
        "cwd",
        "device_paths",
        "environment",
        "read_paths",
        "schema",
        "write_paths",
    } or value.get("schema") != SCHEMA:
        raise LandlockExecError("execution spec schema drifted")
    argv_value = value.get("argv")
    if (
        not isinstance(argv_value, list)
        or len(argv_value) < 3
        or any(not isinstance(row, str) or not row for row in argv_value)
    ):
        raise LandlockExecError("worker argv is invalid")
    argv = tuple(argv_value)
    if not Path(argv[0]).is_absolute():
        raise LandlockExecError("worker executable is not absolute")
    cwd_value = value.get("cwd")
    if not isinstance(cwd_value, str):
        raise LandlockExecError("worker cwd is invalid")
    cwd = Path(cwd_value)
    if not cwd.is_absolute() or cwd.is_symlink() or not cwd.is_dir():
        raise LandlockExecError("worker cwd is unavailable")
    environment_value = value.get("environment")
    if (
        not isinstance(environment_value, Mapping)
        or set(environment_value) != ENVIRONMENT_KEYS
        or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in environment_value.items()
        )
    ):
        raise LandlockExecError("worker environment drifted")
    read_paths = _absolute_paths(value.get("read_paths"), "read_paths")
    write_paths = _absolute_paths(value.get("write_paths"), "write_paths")
    device_paths = _absolute_paths(value.get("device_paths"), "device_paths")
    if any(not row.exists() for row in (*read_paths, *write_paths, *device_paths)):
        raise LandlockExecError("Landlock allowlist path is unavailable")
    return (
        argv,
        cwd,
        dict(environment_value),
        read_paths,
        write_paths,
        device_paths,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    (
        worker_argv,
        cwd,
        environment,
        read_paths,
        write_paths,
        device_paths,
    ) = _validated_spec(arguments.spec.resolve(strict=True))
    apply_landlock(
        read_paths=read_paths,
        write_paths=write_paths,
        device_paths=device_paths,
    )
    os.chdir(cwd)
    os.execve(worker_argv[0], worker_argv, environment)
    raise AssertionError("os.execve unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
