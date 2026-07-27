"""One-shot, network-denied adapter for the private DSTC9 scorer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping

from .contract import (
    CORPUS_SIZE,
    CUDA_VISIBLE_DEVICES,
    SYSTEMD_NETWORK_PROPERTIES,
    WORKER_ENVIRONMENT_KEYS,
    WORKER_FIXED_ENVIRONMENT_VALUES,
    Dstc9CoordinateScorerError,
    canonical_bytes,
    input_projection,
    parse_output_bytes,
    validate_input,
    verify_typed_core_binding,
)


SYSTEMD_RUN = Path("/usr/bin/systemd-run")
ENV_EXECUTABLE = Path("/usr/bin/env")
SYSTEMD_RUN_FLAGS = ("--user", "--wait", "--pipe", "--collect", "--quiet")
SYSTEMD_PREFLIGHT_TIMEOUT_SECONDS = 30
SYSTEMD_PREFLIGHT_SCRIPT = (
    "import os,socket\n"
    "if set(os.environ)!={'LANG'} or os.environ.get('LANG')!='C.UTF-8':"
    " raise SystemExit(40)\n"
    "probe=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);probe.close()\n"
    "for family in (socket.AF_INET,socket.AF_INET6):\n"
    " try: probe=socket.socket(family,socket.SOCK_STREAM)\n"
    " except OSError: continue\n"
    " probe.close();raise SystemExit(41)\n"
)
TERMINAL_KEYS = frozenset(
    {
        "corpus_count",
        "model_binding_sha256",
        "output_self_sha256",
        "query_count",
        "stage",
        "status",
    }
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _assert_no_symlink_components(path: Path, field: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise Dstc9CoordinateScorerError(
                f"{field} contains a symlink component"
            )


def _direct_file(path: Path, field: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise Dstc9CoordinateScorerError(f"{field} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise Dstc9CoordinateScorerError(f"{field} is unavailable") from exc
    _assert_no_symlink_components(resolved, field)
    if not resolved.is_file():
        raise Dstc9CoordinateScorerError(f"{field} is unavailable")
    return resolved


def _direct_directory(path: Path, field: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise Dstc9CoordinateScorerError(f"{field} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise Dstc9CoordinateScorerError(f"{field} is unavailable") from exc
    _assert_no_symlink_components(resolved, field)
    if not resolved.is_dir():
        raise Dstc9CoordinateScorerError(f"{field} is unavailable")
    return resolved


def _write_exclusive(path: Path, payload: object) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    if path.is_symlink() or (path.stat().st_mode & 0o777) != 0o600:
        raise Dstc9CoordinateScorerError(
            "private input permissions drifted"
        )


def _validate_timeout(timeout_seconds: int) -> int:
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 14_400:
        raise Dstc9CoordinateScorerError(
            "timeout is outside the frozen integer bound"
        )
    return timeout_seconds


def _launcher_environment() -> dict[str, str]:
    """Expose only variables needed to contact the user systemd manager."""

    environment = {
        "HOME": os.environ.get("HOME", "/"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PATH": "/usr/bin:/bin",
    }
    for key in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
        value = os.environ.get(key)
        if value:
            environment[key] = value
    return environment


def _systemd_command_prefix() -> list[str]:
    if not SYSTEMD_RUN.is_file():
        raise Dstc9CoordinateScorerError(
            "systemd network-isolating runtime is unavailable"
        )
    command = [str(SYSTEMD_RUN), *SYSTEMD_RUN_FLAGS]
    for property_value in SYSTEMD_NETWORK_PROPERTIES:
        command.extend(("--property", property_value))
    command.append("--")
    return command


def _clean_environment_exec_prefix(
    environment: Mapping[str, str],
) -> list[str]:
    if not ENV_EXECUTABLE.is_file():
        raise Dstc9CoordinateScorerError(
            "environment-clearing runtime is unavailable"
        )
    if any(
        not isinstance(key, str)
        or not key
        or "=" in key
        or "\x00" in key
        or not isinstance(value, str)
        or "\x00" in value
        or "\n" in value
        for key, value in environment.items()
    ):
        raise Dstc9CoordinateScorerError(
            "systemd child environment is malformed"
        )
    return [
        str(ENV_EXECUTABLE),
        "--ignore-environment",
        "--",
        *(f"{key}={environment[key]}" for key in sorted(environment)),
    ]


def _preflight_systemd_transport() -> None:
    command = (
        _systemd_command_prefix()
        + _clean_environment_exec_prefix({"LANG": "C.UTF-8"})
        + ["/usr/bin/python3", "-I", "-c", SYSTEMD_PREFLIGHT_SCRIPT]
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=_launcher_environment(),
            timeout=SYSTEMD_PREFLIGHT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Dstc9CoordinateScorerError(
            "systemd network-isolation capability preflight failed"
        ) from exc
    if completed.returncode != 0:
        raise Dstc9CoordinateScorerError(
            "systemd network-isolation capability preflight failed; "
            f"returncode={completed.returncode}; "
            f"stdout_sha256={_sha256_bytes(completed.stdout)}; "
            f"stderr_sha256={_sha256_bytes(completed.stderr)}"
        )


def _worker_environment(
    *,
    runtime_python: Path,
    project_root: Path,
    writable_root: Path,
) -> dict[str, str]:
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
        "HOME": str(writable_root / "home"),
        "HF_HOME": str(writable_root / "cache"),
        "HF_HUB_OFFLINE": "1",
        "LANG": "C.UTF-8",
        "PATH": f"{runtime_python.parent}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(project_root),
        "TEMP": str(writable_root / "tmp"),
        "TMP": str(writable_root / "tmp"),
        "TMPDIR": str(writable_root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if frozenset(environment) != WORKER_ENVIRONMENT_KEYS or any(
        environment.get(key) != value
        for key, value in WORKER_FIXED_ENVIRONMENT_VALUES.items()
    ):
        raise Dstc9CoordinateScorerError(
            "worker environment contract drifted"
        )
    return environment


def _parse_terminal(
    raw: bytes,
    *,
    expected_query_count: int,
) -> dict[str, object]:
    try:
        lines = raw.decode("utf-8").strip().splitlines()
        terminal = json.loads(lines[-1])
    except (
        IndexError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise Dstc9CoordinateScorerError(
            "coordinate worker emitted no safe terminal receipt"
        ) from exc
    if not isinstance(terminal, dict) or set(terminal) != TERMINAL_KEYS:
        raise Dstc9CoordinateScorerError(
            "coordinate worker terminal schema drifted"
        )
    binding_hash = terminal.get("model_binding_sha256")
    output_hash = terminal.get("output_self_sha256")
    if (
        terminal.get("status") != "passed"
        or terminal.get("stage") != "coordinate_score"
        or terminal.get("corpus_count") != CORPUS_SIZE
        or terminal.get("query_count") != expected_query_count
        or not isinstance(binding_hash, str)
        or len(binding_hash) != 64
        or any(character not in "0123456789abcdef" for character in binding_hash)
        or not isinstance(output_hash, str)
        or len(output_hash) != 64
        or any(character not in "0123456789abcdef" for character in output_hash)
    ):
        raise Dstc9CoordinateScorerError(
            "coordinate worker terminal identity drifted"
        )
    return terminal


def _launch_worker_once(
    *,
    runtime_python: Path,
    project_root: Path,
    minilm_asset_manifest: Path,
    minilm_model_root: Path,
    cross_encoder_model_root: Path,
    private_input_path: Path,
    private_output_path: Path,
    writable_root: Path,
    query_count: int,
    timeout_seconds: int,
) -> dict[str, object]:
    _preflight_systemd_transport()
    child_environment = _worker_environment(
        runtime_python=runtime_python,
        project_root=project_root,
        writable_root=writable_root,
    )
    command = _systemd_command_prefix()
    command.extend(_clean_environment_exec_prefix(child_environment))
    command.extend(
        [
            str(runtime_python),
            "-B",
            "-m",
            "replication_runtime.dstc9_coordinate_scorer_v1.worker",
            "--input",
            str(private_input_path),
            "--output",
            str(private_output_path),
            "--project-root",
            str(project_root),
            "--minilm-asset-manifest",
            str(minilm_asset_manifest),
            "--minilm-model-root",
            str(minilm_model_root),
            "--cross-encoder-model-root",
            str(cross_encoder_model_root),
        ]
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=_launcher_environment(),
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Dstc9CoordinateScorerError(
            "private coordinate worker failed"
        ) from exc
    if completed.returncode != 0:
        raise Dstc9CoordinateScorerError(
            "private coordinate worker failed; "
            f"returncode={completed.returncode}; "
            f"stdout_sha256={_sha256_bytes(completed.stdout)}; "
            f"stderr_sha256={_sha256_bytes(completed.stderr)}"
        )
    return _parse_terminal(
        completed.stdout,
        expected_query_count=query_count,
    )


def run_dstc9_coordinate_scorer_v1(
    *,
    input_value: object,
    runtime_python: Path,
    project_root: Path,
    minilm_asset_manifest: Path,
    minilm_model_root: Path,
    cross_encoder_model_root: Path,
    work_root: Path,
    timeout_seconds: int = 14_400,
) -> dict[str, object]:
    """Run the frozen private scorer exactly once in a fresh local root."""

    scorer_input = validate_input(input_value)
    project_root = _direct_directory(project_root, "project root")
    verify_typed_core_binding(project_root)
    runtime_python = _direct_file(runtime_python, "runtime Python")
    minilm_asset_manifest = _direct_file(
        minilm_asset_manifest, "MiniLM asset manifest"
    )
    minilm_model_root = _direct_directory(
        minilm_model_root, "MiniLM model root"
    )
    cross_encoder_model_root = _direct_directory(
        cross_encoder_model_root, "cross-encoder model root"
    )
    timeout_seconds = _validate_timeout(timeout_seconds)
    if not isinstance(work_root, Path) or not work_root.is_absolute():
        raise Dstc9CoordinateScorerError(
            "private work root must be an absolute path"
        )
    _assert_no_symlink_components(work_root.parent, "private work root parent")
    try:
        work_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        for name in ("cache", "home", "tmp"):
            (work_root / name).mkdir(mode=0o700)
    except OSError as exc:
        raise Dstc9CoordinateScorerError(
            "fresh private work root could not be created"
        ) from exc
    if (work_root.stat().st_mode & 0o777) != 0o700:
        raise Dstc9CoordinateScorerError(
            "private work root permissions drifted"
        )
    private_input_path = work_root / "private_input.json"
    private_output_path = work_root / "private_coordinate_scores.json"
    _write_exclusive(private_input_path, input_projection(scorer_input))
    terminal = _launch_worker_once(
        runtime_python=runtime_python,
        project_root=project_root,
        minilm_asset_manifest=minilm_asset_manifest,
        minilm_model_root=minilm_model_root,
        cross_encoder_model_root=cross_encoder_model_root,
        private_input_path=private_input_path,
        private_output_path=private_output_path,
        writable_root=work_root,
        query_count=len(scorer_input.histories),
        timeout_seconds=timeout_seconds,
    )
    if (
        private_output_path.is_symlink()
        or not private_output_path.is_file()
        or (private_output_path.stat().st_mode & 0o777) != 0o600
    ):
        raise Dstc9CoordinateScorerError(
            "private coordinate output is unavailable"
        )
    output = parse_output_bytes(
        private_output_path.read_bytes(),
        expected_input=scorer_input,
        expected_model_binding_sha256=str(
            terminal["model_binding_sha256"]
        ),
    )
    if output["self_sha256"] != terminal["output_self_sha256"]:
        raise Dstc9CoordinateScorerError(
            "private output terminal binding drifted"
        )
    return output


__all__ = [
    "run_dstc9_coordinate_scorer_v1",
]
