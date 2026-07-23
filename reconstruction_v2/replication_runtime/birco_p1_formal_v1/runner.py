"""Remote/offline executors for the frozen BIRCO P1 formal controller.

The semantic boundary starts the already-frozen one-request worker with a
minimal private environment.  The HippoRAG boundary starts the official-core
worker below ``strace`` network-syscall injection and ``env -i``.  Both use a
content-addressed, exclusive attempt directory and never retry an incomplete
attempt.  Qrels remain behind the selector's authorization/open marker and are
read only after that marker has been durably consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import threading
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks import birco_p1_action_integration_v1 as integration
from assumption_agent.benchmarks import birco_p1_formal_controller_v1 as controller
from assumption_agent.benchmarks import birco_p1_private_selection_v1 as selection
from replication_runtime.birco_gpt54_semantic_v1 import contract as semantic
from replication_runtime.birco_gpt54_semantic_v1 import worker as semantic_worker
from replication_runtime.birco_official_hipporag_v1 import contract as hippo


VERSION = "birco_p1_formal_runtime_v1"
SEMANTIC_WORKER_MODULE = "replication_runtime.birco_gpt54_semantic_v1.worker"
HIPPO_WORKER_MODULE = "replication_runtime.birco_official_hipporag_v1.worker"

PLUS_ENVIRONMENT_KEYS = frozenset(
    {
        "ASSUMPTION_V2_API_BASE",
        "ASSUMPTION_V2_API_KEY",
        "ASSUMPTION_V2_MODEL",
        "BIRCO_P1_PROVIDER_LABEL",
    }
)
SEMANTIC_MODES = frozenset({"canary", "plan", "matrix", "raw"})
HIPPO_GPU_ASSIGNMENT = ("0", "1", "0", "1")
HIPPO_LOGICAL_SLOT_COUNT = 4
HIPPO_MAXIMUM_PROCESSES_PER_GPU = 2
HIPPO_CPU_THREADS_PER_PROCESS = 2
SEMANTIC_PROCESS_TIMEOUT_SECONDS = 720.0

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_RELATIVE = re.compile(r"[A-Za-z0-9._/-]+\Z")
_NETWORK_CALL = re.compile(
    r"^(?:\[pid\s+\d+\]\s+)?[A-Za-z_][A-Za-z0-9_]*\("
)
_NETWORK_RESUMED = re.compile(
    r"^(?:\[pid\s+\d+\]\s+)?<\.\.\.\s+"
    r"[A-Za-z_][A-Za-z0-9_]*\s+resumed>"
)


class BircoP1FormalRuntimeError(RuntimeError):
    """The process, credential, filesystem, network, or qrel boundary drifted."""


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BircoP1FormalRuntimeError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value, newline=False)).hexdigest()


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise BircoP1FormalRuntimeError("self-hash field was supplied twice")
    result = dict(body)
    result[field] = _stable_hash(result)
    return result


def _verify_self(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    claimed = body.pop(field, None)
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise BircoP1FormalRuntimeError(f"{field} is not SHA-256")
    if _stable_hash(body) != claimed:
        raise BircoP1FormalRuntimeError(f"{field} self hash drifted")
    return claimed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise BircoP1FormalRuntimeError("runtime directory is unsafe")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        os.chmod(path, 0o700)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_bytes(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    _ensure_private_directory(path.parent)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_exclusive_json(
    path: Path, value: Mapping[str, Any], *, mode: int = 0o600
) -> None:
    _write_exclusive_bytes(path, _canonical_bytes(value), mode=mode)


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    expected_mode: int | None = None,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise BircoP1FormalRuntimeError(f"{label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise BircoP1FormalRuntimeError(f"{label} is not a bounded regular file")
        if (
            expected_mode is not None
            and stat.S_IMODE(before.st_mode) != expected_mode
        ):
            raise BircoP1FormalRuntimeError(f"{label} mode drifted")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise BircoP1FormalRuntimeError(f"{label} changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_canonical_json(
    path: Path,
    *,
    label: str,
    expected_mode: int | None = None,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    raw = _read_regular_bytes(
        path,
        label=label,
        expected_mode=expected_mode,
        maximum_bytes=maximum_bytes,
    )
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BircoP1FormalRuntimeError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise BircoP1FormalRuntimeError(f"{label} is not canonical JSON")
    return value


def _safe_absolute_executable(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise BircoP1FormalRuntimeError(f"{label} is not an absolute path")
    # Preserve a venv's lexical ``bin/python`` path.  Resolving that symlink in
    # argv can silently leave the venv and therefore change the frozen module
    # closure.  The resolved target is used only for validation/hashing.
    lexical = Path(os.path.abspath(candidate))
    try:
        resolved = lexical.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise BircoP1FormalRuntimeError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise BircoP1FormalRuntimeError(f"{label} is not executable")
    return lexical


def _executable_target_sha256(path: Path) -> str:
    return _file_sha256(path.resolve(strict=True))


def _safe_relative(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or _SAFE_RELATIVE.fullmatch(value) is None:
        raise BircoP1FormalRuntimeError(f"{label} is not a safe relative path")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise BircoP1FormalRuntimeError(f"{label} escapes the project root")
    return path


def _parse_plus_environment(path: Path) -> tuple[dict[str, str], Mapping[str, object]]:
    raw = _read_regular_bytes(
        path,
        label="Plus credential environment",
        expected_mode=0o600,
        maximum_bytes=16 * 1024,
    )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise BircoP1FormalRuntimeError(
            "Plus credential environment is not UTF-8"
        ) from exc
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise BircoP1FormalRuntimeError(
                "Plus credential environment contains a malformed row"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in PLUS_ENVIRONMENT_KEYS or key in values:
            raise BircoP1FormalRuntimeError(
                "Plus credential environment keys drifted"
            )
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        if not value or "\x00" in value or "\r" in value or "\n" in value:
            raise BircoP1FormalRuntimeError(
                "Plus credential environment contains an invalid value"
            )
        values[key] = value
    if set(values) != PLUS_ENVIRONMENT_KEYS:
        raise BircoP1FormalRuntimeError(
            "Plus credential environment does not contain the exact allowlist"
        )
    if (
        values["ASSUMPTION_V2_MODEL"] != semantic.MODEL_ID
        or values["BIRCO_P1_PROVIDER_LABEL"] != "plus"
    ):
        raise BircoP1FormalRuntimeError("Plus provider route is not frozen")
    try:
        provider = semantic_worker.Provider(
            api_base=values["ASSUMPTION_V2_API_BASE"],
            api_origin=semantic_worker.PROVIDER_ORIGIN,
            api_key=values["ASSUMPTION_V2_API_KEY"],
            model=values["ASSUMPTION_V2_MODEL"],
            label=values["BIRCO_P1_PROVIDER_LABEL"],
        )
    except Exception as exc:
        raise BircoP1FormalRuntimeError("Plus provider route is invalid") from exc
    return values, provider.safe_identity()


class SemanticExecutor:
    """Invoke one semantic worker attempt in a payload-specific directory."""

    def __init__(
        self,
        *,
        project_root: Path,
        runtime_root: Path,
        credential_env_path: Path,
        python_executable: Path | str = Path(sys.executable),
        subprocess_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.project_root = Path(project_root).resolve(strict=True)
        self.runtime_root = Path(runtime_root).resolve(strict=False) / "semantic"
        self.credential_env_path = Path(credential_env_path)
        self.python_executable = _safe_absolute_executable(
            Path(python_executable), label="Python executable"
        )
        self.subprocess_runner = subprocess_runner
        environment, provider_identity = _parse_plus_environment(
            self.credential_env_path
        )
        # Kept only in process memory.  The credential file is not re-read
        # between controller claims, so later filesystem drift cannot switch
        # provider or invalidate an already-started formal lifecycle.
        self._credential_environment = dict(environment)
        self._provider_identity = dict(provider_identity)
        _ensure_private_directory(self.runtime_root)
        self._lock = threading.Lock()

    @property
    def provider_identity(self) -> Mapping[str, object]:
        return dict(self._provider_identity)

    def _validate_claim(
        self,
        value: Mapping[str, Any],
        *,
        mode: str,
        payload: Mapping[str, Any],
        provider_identity: Mapping[str, object],
    ) -> None:
        expected_fields = {
            "input_sha256",
            "mode",
            "provider",
            "schema",
            "status",
            "work_id",
            "self_sha256",
        }
        if (
            set(value) != expected_fields
            or value.get("schema")
            != f"{semantic_worker.VERSION}_durable_pre_http_attempt_claim_v1"
            or value.get("status")
            != "consumed_before_the_only_authorized_HTTP_request"
            or value.get("mode") != mode
            or value.get("work_id") != payload.get("work_id")
            or value.get("input_sha256") != semantic.semantic_hash(payload)
            or value.get("provider") != dict(provider_identity)
        ):
            raise BircoP1FormalRuntimeError("semantic worker attempt claim drifted")
        _verify_self(value, "self_sha256")

    @staticmethod
    def _validate_canary_terminal(
        value: Mapping[str, Any],
        *,
        payload: Mapping[str, Any],
        provider_identity: Mapping[str, object],
    ) -> None:
        expected_fields = {
            "action",
            "attempt_count",
            "generation_valid",
            "input_sha256",
            "mode",
            "model_request_sha256",
            "provider",
            "raw_completion_persisted",
            "response_sha256",
            "retry_replay_resample_or_provider_switch_count",
            "schema",
            "self_sha256",
            "terminal_category",
            "transport",
            "transport_succeeded",
            "work_id",
        }
        action = value.get("action")
        if (
            set(value) != expected_fields
            or value.get("schema") != semantic.TERMINAL_OUTPUT_SCHEMA
            or value.get("mode") != "canary"
            or value.get("input_sha256") != semantic.semantic_hash(payload)
            or value.get("attempt_count") != 1
            or value.get("retry_replay_resample_or_provider_switch_count") != 0
            or value.get("raw_completion_persisted") is not False
            or value.get("provider") != dict(provider_identity)
            or not isinstance(action, Mapping)
            or set(action) != {"nonempty_response"}
            or type(action.get("nonempty_response")) is not bool
        ):
            raise BircoP1FormalRuntimeError("semantic canary terminal drifted")
        _verify_self(value, "self_sha256")

    def _validate_terminal(
        self,
        value: Mapping[str, Any],
        *,
        mode: str,
        payload: Mapping[str, Any],
        provider_identity: Mapping[str, object],
    ) -> None:
        try:
            if mode == "canary":
                self._validate_canary_terminal(
                    value,
                    payload=payload,
                    provider_identity=provider_identity,
                )
            else:
                integration._validate_semantic_terminal(
                    value, mode=mode, expected_input=payload
                )
                if value.get("provider") != dict(provider_identity):
                    raise BircoP1FormalRuntimeError(
                        "semantic terminal provider differs from Plus"
                    )
        except BircoP1FormalRuntimeError:
            raise
        except Exception as exc:
            raise BircoP1FormalRuntimeError("semantic terminal drifted") from exc

    def _recover(
        self,
        directory: Path,
        *,
        mode: str,
        payload: Mapping[str, Any],
        provider_identity: Mapping[str, object],
    ) -> dict[str, Any]:
        input_path = directory / "input.json"
        claim_path = directory / "terminal.json.attempt.json"
        output_path = directory / "terminal.json"
        if _read_canonical_json(
            input_path, label="semantic worker input", expected_mode=0o600
        ) != dict(payload):
            raise BircoP1FormalRuntimeError("semantic worker input drifted")
        claim = _read_canonical_json(
            claim_path,
            label="semantic worker durable attempt claim",
            expected_mode=0o600,
        )
        self._validate_claim(
            claim,
            mode=mode,
            payload=payload,
            provider_identity=provider_identity,
        )
        if not output_path.exists() or output_path.is_symlink():
            raise BircoP1FormalRuntimeError(
                "semantic attempt is consumed without a complete terminal"
            )
        terminal = _read_canonical_json(
            output_path, label="semantic worker terminal", expected_mode=0o600
        )
        self._validate_terminal(
            terminal,
            mode=mode,
            payload=payload,
            provider_identity=provider_identity,
        )
        return terminal

    def __call__(
        self, *, mode: str, payload: Mapping[str, object]
    ) -> Mapping[str, object]:
        if mode not in SEMANTIC_MODES or not isinstance(payload, Mapping):
            raise BircoP1FormalRuntimeError("semantic mode or payload is invalid")
        try:
            validated = semantic_worker._validate_input(mode, dict(payload))
        except Exception as exc:
            raise BircoP1FormalRuntimeError("semantic worker input is invalid") from exc
        environment = dict(self._credential_environment)
        provider_identity = dict(self._provider_identity)
        invocation_hash = _stable_hash({"mode": mode, "payload": validated})
        directory = self.runtime_root / mode / invocation_hash
        with self._lock:
            if directory.exists() or directory.is_symlink():
                return self._recover(
                    directory,
                    mode=mode,
                    payload=validated,
                    provider_identity=provider_identity,
                )
            _ensure_private_directory(directory.parent)
            try:
                directory.mkdir(mode=0o700)
            except OSError as exc:
                raise BircoP1FormalRuntimeError(
                    "semantic attempt directory could not be claimed"
                ) from exc
            _fsync_directory(directory.parent)
            _write_exclusive_bytes(
                directory / "input.json", semantic.canonical_json_bytes(validated)
            )

        command = [
            str(self.python_executable),
            "-m",
            SEMANTIC_WORKER_MODULE,
            "--mode",
            mode,
            "--input",
            str(directory / "input.json"),
            "--output",
            str(directory / "terminal.json"),
        ]
        child_environment = {
            **environment,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(self.project_root),
        }
        try:
            completed = self.subprocess_runner(
                command,
                cwd=self.project_root,
                env=child_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=SEMANTIC_PROCESS_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise BircoP1FormalRuntimeError(
                "semantic worker process did not produce a terminal"
            ) from exc
        # Captured output is deliberately neither emitted nor persisted.
        if type(getattr(completed, "returncode", None)) is not int:
            raise BircoP1FormalRuntimeError("semantic worker process result drifted")
        terminal = self._recover(
            directory,
            mode=mode,
            payload=validated,
            provider_identity=provider_identity,
        )
        if completed.returncode != 0:
            raise BircoP1FormalRuntimeError("semantic worker process failed")
        return terminal


class HippoExecutor:
    """Run official HippoRAG with fixed GPU policy and denied networking."""

    _POLICY_FIELDS = frozenset(
        {
            "model_alias_cwd_relative",
            "llm_model_alias",
            "embedding_model_alias",
            "aliases_are_single_relative_components",
            "subprocess_cwd_is_model_alias_cwd",
            "absolute_model_path_argument_count",
            "logical_slot_count",
            "gpu_assignment",
            "maximum_processes_per_gpu",
            "cpu_threads_per_process",
            "logical_slot_ordinal",
            "visible_gpu",
        }
    )
    _CONTROLLER_RECEIPT_FIELDS = frozenset(
        {
            "model_alias_cwd_relative",
            "subprocess_cwd_relative",
            "llm_model_argument",
            "embedding_model_argument",
            "model_arguments_are_single_relative_components",
            "absolute_model_path_argument_count",
            "logical_slot_ordinal",
            "visible_gpu",
            "configured_cpu_threads",
            "external_network_call_count",
        }
    )
    _BASE_POLICY_FIELDS = _POLICY_FIELDS - frozenset(
        {"logical_slot_ordinal", "visible_gpu"}
    )

    def __init__(
        self,
        *,
        project_root: Path,
        runtime_root: Path,
        python_executable: Path | str = Path(sys.executable),
        strace_executable: Path | str = Path("/usr/bin/strace"),
        env_executable: Path | str = Path("/usr/bin/env"),
        subprocess_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.project_root = Path(project_root).resolve(strict=True)
        self.runtime_root = Path(runtime_root).resolve(strict=False) / "hipporag"
        self.python_executable = _safe_absolute_executable(
            Path(python_executable), label="Python executable"
        )
        self.strace_executable = _safe_absolute_executable(
            Path(strace_executable), label="strace executable"
        )
        self.env_executable = _safe_absolute_executable(
            Path(env_executable), label="env executable"
        )
        self.subprocess_runner = subprocess_runner
        self._python_executable_target_sha256 = _executable_target_sha256(
            self.python_executable
        )
        self._strace_executable_target_sha256 = _executable_target_sha256(
            self.strace_executable
        )
        self._env_executable_target_sha256 = _executable_target_sha256(
            self.env_executable
        )
        _ensure_private_directory(self.runtime_root)
        self._global_semaphore = threading.BoundedSemaphore(
            HIPPO_LOGICAL_SLOT_COUNT
        )
        self._gpu_semaphores = {
            gpu: threading.BoundedSemaphore(HIPPO_MAXIMUM_PROCESSES_PER_GPU)
            for gpu in sorted(set(HIPPO_GPU_ASSIGNMENT))
        }
        self._counter_lock = threading.Lock()
        self._directory_lock = threading.Lock()
        self._active_total = 0
        self._active_by_gpu = {gpu: 0 for gpu in self._gpu_semaphores}
        self._observed_total_peak = 0
        self._observed_by_gpu_peak = {gpu: 0 for gpu in self._gpu_semaphores}

    @staticmethod
    def _validate_input(payload: Mapping[str, object]) -> dict[str, Any]:
        if set(payload) != hippo.INPUT_KEYS or payload.get("schema") != hippo.INPUT_SCHEMA:
            raise BircoP1FormalRuntimeError("HippoRAG input shape drifted")
        try:
            work_id, objective, query, documents, projection_hash = hippo.validate_input(
                payload.get("work_id"),
                payload.get("objective"),
                payload.get("query"),
                payload.get("documents"),
                payload.get("common_projection_sha256"),
            )
        except Exception as exc:
            raise BircoP1FormalRuntimeError("HippoRAG input is invalid") from exc
        canonical = {
            "common_projection_sha256": projection_hash,
            "documents": [
                {"ordinal": row.ordinal, "text": row.text} for row in documents
            ],
            "objective": objective,
            "query": query,
            "schema": hippo.INPUT_SCHEMA,
            "work_id": work_id,
        }
        if canonical != dict(payload):
            raise BircoP1FormalRuntimeError("HippoRAG input is noncanonical")
        return canonical

    def _validate_policy(
        self, value: Mapping[str, object]
    ) -> tuple[str, Path, int, str]:
        if not isinstance(value, Mapping) or set(value) != self._POLICY_FIELDS:
            raise BircoP1FormalRuntimeError("HippoRAG runtime policy shape drifted")
        relative = _safe_relative(
            value.get("model_alias_cwd_relative"), label="model alias cwd"
        )
        relative_text = relative.as_posix()
        slot = value.get("logical_slot_ordinal")
        visible_gpu = value.get("visible_gpu")
        if (
            relative_text in {"", "."}
            or value.get("llm_model_alias") != "smollm2"
            or value.get("embedding_model_alias") != "minilm"
            or value.get("aliases_are_single_relative_components") is not True
            or value.get("subprocess_cwd_is_model_alias_cwd") is not True
            or value.get("absolute_model_path_argument_count") != 0
            or value.get("logical_slot_count") != HIPPO_LOGICAL_SLOT_COUNT
            or value.get("gpu_assignment") != list(HIPPO_GPU_ASSIGNMENT)
            or value.get("maximum_processes_per_gpu")
            != HIPPO_MAXIMUM_PROCESSES_PER_GPU
            or value.get("cpu_threads_per_process")
            != HIPPO_CPU_THREADS_PER_PROCESS
            or isinstance(slot, bool)
            or not isinstance(slot, int)
            or not 0 <= slot < HIPPO_LOGICAL_SLOT_COUNT
            or visible_gpu != HIPPO_GPU_ASSIGNMENT[slot]
        ):
            raise BircoP1FormalRuntimeError("HippoRAG frozen runtime policy drifted")
        model_cwd = self.project_root / relative
        try:
            metadata = model_cwd.lstat()
            resolved_cwd = model_cwd.resolve(strict=True)
        except OSError as exc:
            raise BircoP1FormalRuntimeError("model alias cwd is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise BircoP1FormalRuntimeError("model alias cwd is not a directory")
        try:
            resolved_cwd.relative_to(self.project_root)
        except ValueError as exc:
            raise BircoP1FormalRuntimeError("model alias cwd escaped the project") from exc
        for alias in ("smollm2", "minilm"):
            path = model_cwd / alias
            try:
                alias_metadata = path.lstat()
                target = path.resolve(strict=True)
            except OSError as exc:
                raise BircoP1FormalRuntimeError(
                    "frozen model alias is unavailable"
                ) from exc
            if not stat.S_ISLNK(alias_metadata.st_mode) or not target.is_dir():
                raise BircoP1FormalRuntimeError(
                    "frozen model alias is not a directory symlink"
                )
        return relative_text, resolved_cwd, slot, str(visible_gpu)

    def preflight_base_policy(self, value: Mapping[str, object]) -> None:
        """Verify every frozen slot/model alias before any controller claim."""

        if not isinstance(value, Mapping) or set(value) != self._BASE_POLICY_FIELDS:
            raise BircoP1FormalRuntimeError(
                "HippoRAG base runtime policy shape drifted"
            )
        assignment = value.get("gpu_assignment")
        if assignment != list(HIPPO_GPU_ASSIGNMENT):
            raise BircoP1FormalRuntimeError("HippoRAG GPU assignment drifted")
        for slot, gpu in enumerate(HIPPO_GPU_ASSIGNMENT):
            self._validate_policy(
                {
                    **dict(value),
                    "logical_slot_ordinal": slot,
                    "visible_gpu": gpu,
                }
            )

    @staticmethod
    def _validate_output(
        output_path: Path, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        raw = _read_regular_bytes(
            output_path,
            label="HippoRAG output",
            expected_mode=0o600,
            maximum_bytes=4 * 1024 * 1024,
        )
        try:
            result = hippo.parse_output(raw)
        except Exception as exc:
            raise BircoP1FormalRuntimeError("HippoRAG output drifted") from exc
        candidate_count = len(payload["documents"])
        if (
            result.get("work_id") != payload.get("work_id")
            or result.get("common_projection_sha256")
            != payload.get("common_projection_sha256")
            or result.get("candidate_count") != candidate_count
            or set(result.get("rank_ordinals", ())) != set(range(candidate_count))
        ):
            raise BircoP1FormalRuntimeError("HippoRAG output binding drifted")
        return result

    @staticmethod
    def _audit_network(path: Path) -> dict[str, Any]:
        raw = _read_regular_bytes(
            path,
            label="strace network audit",
            maximum_bytes=64 * 1024 * 1024,
        )
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise BircoP1FormalRuntimeError("strace network audit is not UTF-8") from exc
        attempted = 0
        denied = 0
        for line in lines:
            text = line.strip()
            if not text:
                continue
            if text.startswith("strace:"):
                raise BircoP1FormalRuntimeError(
                    "strace reported an audit or injection failure"
                )
            if text.startswith("+++") or text.startswith("---"):
                continue
            if "<unfinished ...>" in text:
                if _NETWORK_CALL.match(text) is None:
                    raise BircoP1FormalRuntimeError("network audit line drifted")
                continue
            if _NETWORK_CALL.match(text) or _NETWORK_RESUMED.match(text):
                attempted += 1
                if "= -1 EPERM" not in text:
                    raise BircoP1FormalRuntimeError(
                        "a network syscall was not denied by strace"
                    )
                denied += 1
                continue
            raise BircoP1FormalRuntimeError("network audit contains an unknown line")
        if denied != attempted:
            raise BircoP1FormalRuntimeError("network denial audit is incomplete")
        os.chmod(path, 0o400, follow_symlinks=False)
        return {
            "attempted_network_syscall_count": attempted,
            "denied_network_syscall_count": denied,
            "external_network_call_count": 0,
            "strace_file_sha256": hashlib.sha256(raw).hexdigest(),
        }

    def _counter_enter(self, gpu: str) -> Mapping[str, object]:
        with self._counter_lock:
            self._active_total += 1
            self._active_by_gpu[gpu] += 1
            if (
                self._active_total > HIPPO_LOGICAL_SLOT_COUNT
                or self._active_by_gpu[gpu] > HIPPO_MAXIMUM_PROCESSES_PER_GPU
            ):
                raise BircoP1FormalRuntimeError("HippoRAG concurrency cap drifted")
            self._observed_total_peak = max(
                self._observed_total_peak, self._active_total
            )
            self._observed_by_gpu_peak[gpu] = max(
                self._observed_by_gpu_peak[gpu], self._active_by_gpu[gpu]
            )
            return {
                "configured_logical_slot_count": HIPPO_LOGICAL_SLOT_COUNT,
                "configured_gpu_assignment": list(HIPPO_GPU_ASSIGNMENT),
                "configured_maximum_processes_per_gpu": (
                    HIPPO_MAXIMUM_PROCESSES_PER_GPU
                ),
                "observed_total_process_peak": self._observed_total_peak,
                "observed_process_peak_by_gpu": dict(self._observed_by_gpu_peak),
            }

    def _counter_exit(self, gpu: str) -> None:
        with self._counter_lock:
            self._active_by_gpu[gpu] -= 1
            self._active_total -= 1
            if self._active_total < 0 or self._active_by_gpu[gpu] < 0:
                raise BircoP1FormalRuntimeError("HippoRAG concurrency counter drifted")

    @staticmethod
    def _controller_receipt(
        *, relative_cwd: str, slot: int, gpu: str
    ) -> dict[str, object]:
        receipt: dict[str, object] = {
            "model_alias_cwd_relative": relative_cwd,
            "subprocess_cwd_relative": relative_cwd,
            "llm_model_argument": "smollm2",
            "embedding_model_argument": "minilm",
            "model_arguments_are_single_relative_components": True,
            "absolute_model_path_argument_count": 0,
            "logical_slot_ordinal": slot,
            "visible_gpu": gpu,
            "configured_cpu_threads": HIPPO_CPU_THREADS_PER_PROCESS,
            "external_network_call_count": 0,
        }
        if set(receipt) != HippoExecutor._CONTROLLER_RECEIPT_FIELDS:
            raise BircoP1FormalRuntimeError("controller runtime receipt drifted")
        return receipt

    def _recover(
        self,
        directory: Path,
        *,
        payload: Mapping[str, Any],
        relative_cwd: str,
        slot: int,
        gpu: str,
    ) -> dict[str, object]:
        if _read_canonical_json(
            directory / "input.json",
            label="HippoRAG input",
            expected_mode=0o600,
        ) != dict(payload):
            raise BircoP1FormalRuntimeError("HippoRAG persisted input drifted")
        claim = _read_canonical_json(
            directory / "attempt.json",
            label="HippoRAG durable attempt claim",
            expected_mode=0o600,
        )
        if (
            set(claim)
            != {
                "schema",
                "status",
                "input_sha256",
                "logical_slot_ordinal",
                "visible_gpu",
                "attempt_count",
                "retry_count",
                "self_sha256",
            }
            or claim.get("schema") != f"{VERSION}_hipporag_attempt_claim_v1"
            or claim.get("status") != "consumed_before_offline_subprocess"
            or claim.get("input_sha256")
            != hashlib.sha256(hippo.canonical_json_bytes(payload, newline=False)).hexdigest()
            or claim.get("logical_slot_ordinal") != slot
            or claim.get("visible_gpu") != gpu
            or claim.get("attempt_count") != 1
            or claim.get("retry_count") != 0
        ):
            raise BircoP1FormalRuntimeError("HippoRAG attempt claim drifted")
        _verify_self(claim, "self_sha256")
        output_path = directory / "output.json"
        if not output_path.exists() or output_path.is_symlink():
            raise BircoP1FormalRuntimeError(
                "HippoRAG attempt is consumed without a complete output"
            )
        output = self._validate_output(output_path, payload)
        receipt = _read_canonical_json(
            directory / "controller_runtime_receipt.json",
            label="HippoRAG controller runtime receipt",
            expected_mode=0o600,
        )
        expected_receipt = self._controller_receipt(
            relative_cwd=relative_cwd, slot=slot, gpu=gpu
        )
        if receipt != expected_receipt:
            raise BircoP1FormalRuntimeError("HippoRAG runtime receipt drifted")
        audit = _read_canonical_json(
            directory / "runtime_audit_receipt.json",
            label="HippoRAG safe audit receipt",
            expected_mode=0o600,
        )
        if (
            audit.get("schema") != f"{VERSION}_hipporag_safe_runtime_audit_v1"
            or audit.get("input_sha256") != claim.get("input_sha256")
            or audit.get("output_file_sha256") != _file_sha256(output_path)
            or audit.get("external_network_call_count") != 0
            or audit.get("logical_slot_ordinal") != slot
            or audit.get("visible_gpu") != gpu
        ):
            raise BircoP1FormalRuntimeError("HippoRAG safe audit receipt drifted")
        _verify_self(audit, "self_sha256")
        return {"output": output, "runtime_receipt": receipt}

    def __call__(
        self,
        *,
        payload: Mapping[str, object],
        runtime_policy: Mapping[str, object],
    ) -> Mapping[str, object]:
        canonical_input = self._validate_input(payload)
        relative_cwd, model_cwd, slot, gpu = self._validate_policy(runtime_policy)
        input_sha256 = hashlib.sha256(
            hippo.canonical_json_bytes(canonical_input, newline=False)
        ).hexdigest()
        directory = self.runtime_root / input_sha256
        with self._directory_lock:
            if directory.exists() or directory.is_symlink():
                return self._recover(
                    directory,
                    payload=canonical_input,
                    relative_cwd=relative_cwd,
                    slot=slot,
                    gpu=gpu,
                )
            try:
                directory.mkdir(mode=0o700)
            except OSError as exc:
                raise BircoP1FormalRuntimeError(
                    "HippoRAG attempt directory could not be claimed"
                ) from exc
            _fsync_directory(directory.parent)
            _write_exclusive_bytes(
                directory / "input.json", hippo.canonical_json_bytes(canonical_input)
            )
            _write_exclusive_json(
                directory / "index.reservation.json",
                {
                    "index_directory_name": "index",
                    "input_sha256": input_sha256,
                    "schema": f"{VERSION}_hipporag_index_reservation_v1",
                },
            )
            claim = _self_hashed(
                {
                    "schema": f"{VERSION}_hipporag_attempt_claim_v1",
                    "status": "consumed_before_offline_subprocess",
                    "input_sha256": input_sha256,
                    "logical_slot_ordinal": slot,
                    "visible_gpu": gpu,
                    "attempt_count": 1,
                    "retry_count": 0,
                },
                "self_sha256",
            )
            _write_exclusive_json(directory / "attempt.json", claim)

        input_path = directory / "input.json"
        output_path = directory / "output.json"
        index_root = directory / "index"
        audit_path = directory / "network.strace"
        if any(
            path.exists() or path.is_symlink()
            for path in (output_path, index_root, audit_path)
        ):
            raise BircoP1FormalRuntimeError("HippoRAG output path was preclaimed")

        environment_rows = [
            "CUDA_VISIBLE_DEVICES=" + gpu,
            "HF_DATASETS_OFFLINE=1",
            "HF_HUB_DISABLE_TELEMETRY=1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "TOKENIZERS_PARALLELISM=false",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONNOUSERSITE=1",
            "PYTHONPATH=" + str(self.project_root),
            f"OMP_NUM_THREADS={HIPPO_CPU_THREADS_PER_PROCESS}",
            f"MKL_NUM_THREADS={HIPPO_CPU_THREADS_PER_PROCESS}",
            f"OPENBLAS_NUM_THREADS={HIPPO_CPU_THREADS_PER_PROCESS}",
            f"NUMEXPR_NUM_THREADS={HIPPO_CPU_THREADS_PER_PROCESS}",
        ]
        command = [
            str(self.strace_executable),
            "-f",
            "-qq",
            "-e",
            "trace=%network",
            "-e",
            "inject=%network:error=EPERM",
            "-o",
            str(audit_path),
            str(self.env_executable),
            "-i",
            *environment_rows,
            str(self.python_executable),
            "-m",
            HIPPO_WORKER_MODULE,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--index-root",
            str(index_root),
            "--llm-model",
            "smollm2",
            "--embedding-model",
            "minilm",
        ]

        self._global_semaphore.acquire()
        self._gpu_semaphores[gpu].acquire()
        counter_snapshot = self._counter_enter(gpu)
        try:
            try:
                completed = self.subprocess_runner(
                    command,
                    cwd=model_cwd,
                    env={},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=None,
                )
            except Exception as exc:
                raise BircoP1FormalRuntimeError(
                    "HippoRAG process did not produce a terminal"
                ) from exc
            if type(getattr(completed, "returncode", None)) is not int:
                raise BircoP1FormalRuntimeError("HippoRAG process result drifted")
            if completed.returncode != 0:
                raise BircoP1FormalRuntimeError("HippoRAG process failed")
            if index_root.is_symlink() or not index_root.is_dir():
                raise BircoP1FormalRuntimeError("HippoRAG index was not created safely")
            output = self._validate_output(output_path, canonical_input)
            network = self._audit_network(audit_path)
            controller_receipt = self._controller_receipt(
                relative_cwd=relative_cwd, slot=slot, gpu=gpu
            )
            _write_exclusive_json(
                directory / "controller_runtime_receipt.json", controller_receipt
            )
            audit_receipt = _self_hashed(
                {
                    "schema": f"{VERSION}_hipporag_safe_runtime_audit_v1",
                    "status": "offline_worker_complete_with_network_syscalls_denied",
                    "input_sha256": input_sha256,
                    "output_file_sha256": _file_sha256(output_path),
                    "index_created_exclusively": True,
                    "subprocess_environment_was_env_i": True,
                    "offline_environment_frozen": True,
                    "model_arguments_were_short_aliases": True,
                    "resolved_model_paths_persisted": False,
                    "subprocess_stdout_or_stderr_persisted": False,
                    "python_executable_target_sha256": (
                        self._python_executable_target_sha256
                    ),
                    "strace_executable_target_sha256": (
                        self._strace_executable_target_sha256
                    ),
                    "env_executable_target_sha256": (
                        self._env_executable_target_sha256
                    ),
                    "logical_slot_ordinal": slot,
                    "visible_gpu": gpu,
                    "configured_cpu_threads": HIPPO_CPU_THREADS_PER_PROCESS,
                    **counter_snapshot,
                    **network,
                },
                "self_sha256",
            )
            _write_exclusive_json(
                directory / "runtime_audit_receipt.json", audit_receipt
            )
            return {"output": output, "runtime_receipt": controller_receipt}
        finally:
            self._counter_exit(gpu)
            self._gpu_semaphores[gpu].release()
            self._global_semaphore.release()


class QrelOpener:
    """Consume selector capabilities and recover only already-opened qrels."""

    _SCORE_RECEIPTS = {
        "A_form": "A_form_e4_model.json",
        "A_hold": "A_hold_score_and_promotion.json",
        "M_search": "M_search_score.json",
    }

    def __init__(
        self,
        *,
        output_root: Path,
        control_root: Path,
        authorization_root: Path | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve(strict=True)
        self.control_root = Path(control_root).resolve(strict=False)
        self.authorization_root = (
            Path(authorization_root).resolve(strict=False)
            if authorization_root is not None
            else self.control_root / "qrel_authorizations"
        )
        _ensure_private_directory(self.authorization_root)
        self._lock = threading.Lock()

    def _authorization_path(self, block: str) -> Path:
        return self.authorization_root / f"{block}.authorization.json"

    def _score_receipt_path(self, block: str) -> Path:
        return self.control_root / self._SCORE_RECEIPTS[block]

    def _validate_authorization(
        self,
        path: Path,
        *,
        block: str,
        action_archive_sha256s: Sequence[str],
        promotion_sha256: str | None,
    ) -> tuple[dict[str, Any], str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        receipt = selection._load_public_receipt(self.output_root)
        action_binding = selection._pack_binding(receipt, block=block, role="action")
        qrel_binding = selection._pack_binding(receipt, block=block, role="qrels")
        raw = selection._read_stable_regular_bytes(
            path,
            label=f"{block} qrel-open authorization",
            expected_mode=0o600,
        )
        value = selection._strict_json(raw, label=f"{block} qrel-open authorization")
        if (
            not isinstance(value, Mapping)
            or raw != selection._canonical_bytes(value, newline=True)
        ):
            raise BircoP1FormalRuntimeError("qrel authorization is noncanonical")
        authorization = dict(value)
        claimed = authorization.get("authorization_sha256")
        try:
            observed = selection._validate_block_authorization(
                authorization,
                expected_authorization_sha256=str(claimed),
                receipt=receipt,
                block=block,
                action_binding=action_binding,
                qrel_binding=qrel_binding,
            )
        except Exception as exc:
            raise BircoP1FormalRuntimeError("qrel authorization drifted") from exc
        if (
            authorization.get("action_archive_sha256s")
            != sorted(action_archive_sha256s)
            or authorization.get("A_hold_promotion_sha256") != promotion_sha256
        ):
            raise BircoP1FormalRuntimeError("qrel authorization request drifted")
        return authorization, observed, receipt, action_binding, qrel_binding

    def _recover_opened(
        self,
        *,
        block: str,
        authorization_path: Path,
        action_archive_sha256s: Sequence[str],
        promotion_sha256: str | None,
    ) -> dict[str, Any]:
        score_path = self._score_receipt_path(block)
        if score_path.exists() or score_path.is_symlink():
            raise BircoP1FormalRuntimeError(
                "opened qrel cannot be recovered after its score receipt"
            )
        (
            _authorization,
            authorization_sha256,
            receipt,
            action_binding,
            qrel_binding,
        ) = self._validate_authorization(
            authorization_path,
            block=block,
            action_archive_sha256s=action_archive_sha256s,
            promotion_sha256=promotion_sha256,
        )
        marker_path = self.output_root / selection.QREL_OPEN_MARKER_FILENAMES[block]
        raw = selection._read_stable_regular_bytes(
            marker_path,
            label=f"{block} qrel-open marker",
            expected_mode=0o600,
        )
        marker_value = selection._strict_json(raw, label=f"{block} qrel-open marker")
        if (
            not isinstance(marker_value, Mapping)
            or raw != selection._canonical_bytes(marker_value, newline=True)
        ):
            raise BircoP1FormalRuntimeError("qrel-open marker is noncanonical")
        marker = dict(marker_value)
        expected_fields = {
            "schema",
            "version",
            "study_id",
            "status",
            "block",
            "acquisition_sha256",
            "authorization_sha256",
            "same_block_second_open_authorized",
            "open_marker_sha256",
        }
        try:
            marker_sha256 = selection.verify_self_hash(
                marker, "open_marker_sha256"
            )
        except Exception as exc:
            raise BircoP1FormalRuntimeError("qrel-open marker hash drifted") from exc
        if (
            set(marker) != expected_fields
            or marker.get("schema") != f"{selection.VERSION}_qrel_open_marker_v1"
            or marker.get("version") != selection.VERSION
            or marker.get("study_id") != selection.STUDY_ID
            or marker.get("status")
            != "authorization_consumed_immediately_before_numeric_qrel_open"
            or marker.get("block") != block
            or marker.get("acquisition_sha256") != receipt.get("acquisition_sha256")
            or marker.get("authorization_sha256") != authorization_sha256
            or marker.get("same_block_second_open_authorized") is not False
            or _SHA256.fullmatch(marker_sha256) is None
        ):
            raise BircoP1FormalRuntimeError("qrel-open marker drifted")
        try:
            qrel_pack = selection._read_bound_private_pack(
                self.output_root,
                binding=qrel_binding,
                label=f"{block} sealed qrel pack",
            )
            observed = selection._validate_qrel_pack(
                qrel_pack,
                block=block,
                expected_action_pack_sha256=str(action_binding["semantic_sha256"]),
            )
        except Exception as exc:
            raise BircoP1FormalRuntimeError("authorized qrel pack drifted") from exc
        if observed != qrel_binding.get("semantic_sha256"):
            raise BircoP1FormalRuntimeError("authorized qrel commitment drifted")
        return qrel_pack

    def __call__(
        self,
        *,
        block: str,
        action_archive_sha256s: Sequence[str],
        promotion_sha256: str | None,
    ) -> Mapping[str, object]:
        # This permanent gate precedes all qrel/receipt path inspection.
        if block == "F_search":
            raise BircoP1FormalRuntimeError("F_search qrels are permanently sealed")
        if block not in self._SCORE_RECEIPTS:
            raise BircoP1FormalRuntimeError("qrel-open block is invalid")
        archives = tuple(action_archive_sha256s)
        if (
            not archives
            or len(set(archives)) != len(archives)
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in archives
            )
        ):
            raise BircoP1FormalRuntimeError("action archive bindings are invalid")
        if block == "M_search":
            if (
                not isinstance(promotion_sha256, str)
                or _SHA256.fullmatch(promotion_sha256) is None
            ):
                raise BircoP1FormalRuntimeError(
                    "M_search promotion binding is invalid"
                )
        elif promotion_sha256 is not None:
            raise BircoP1FormalRuntimeError("unexpected qrel promotion binding")

        authorization_path = self._authorization_path(block)
        marker_path = self.output_root / selection.QREL_OPEN_MARKER_FILENAMES[block]
        with self._lock:
            if marker_path.exists() or marker_path.is_symlink():
                return self._recover_opened(
                    block=block,
                    authorization_path=authorization_path,
                    action_archive_sha256s=archives,
                    promotion_sha256=promotion_sha256,
                )
            if self._score_receipt_path(block).exists() or self._score_receipt_path(
                block
            ).is_symlink():
                raise BircoP1FormalRuntimeError(
                    "score receipt exists without a qrel-open marker"
                )
            if authorization_path.exists() or authorization_path.is_symlink():
                authorization, authorization_sha256, *_rest = (
                    self._validate_authorization(
                        authorization_path,
                        block=block,
                        action_archive_sha256s=archives,
                        promotion_sha256=promotion_sha256,
                    )
                )
            else:
                try:
                    authorization = selection.write_block_open_authorization(
                        authorization_path,
                        output_root=self.output_root,
                        block=block,
                        action_archive_sha256s=archives,
                        promotion_sha256=promotion_sha256,
                    )
                except Exception as exc:
                    raise BircoP1FormalRuntimeError(
                        "qrel authorization could not be written"
                    ) from exc
                authorization_sha256 = str(authorization.get("authorization_sha256"))
                if _SHA256.fullmatch(authorization_sha256) is None:
                    raise BircoP1FormalRuntimeError(
                        "qrel authorization hash is invalid"
                    )
            try:
                return selection.open_block_qrels(
                    output_root=self.output_root,
                    block=block,
                    authorization_path=authorization_path,
                    expected_authorization_sha256=authorization_sha256,
                )
            except Exception as exc:
                raise BircoP1FormalRuntimeError("authorized qrel open failed") from exc


def _resolve_cli_path(
    value: Path, *, label: str, must_exist: bool, directory: bool = False
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise BircoP1FormalRuntimeError(f"{label} must be absolute")
    if must_exist:
        try:
            supplied_metadata = path.lstat()
        except OSError as exc:
            raise BircoP1FormalRuntimeError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(supplied_metadata.st_mode):
            raise BircoP1FormalRuntimeError(f"{label} traverses a symlink")
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as exc:
        raise BircoP1FormalRuntimeError(f"{label} is unavailable") from exc
    if must_exist:
        metadata = resolved.lstat()
        expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(
            metadata.st_mode
        )
        if stat.S_ISLNK(metadata.st_mode) or not expected:
            raise BircoP1FormalRuntimeError(f"{label} type drifted")
    return resolved


def _load_execution_freeze_for_paths(
    path: Path, *, expected_file_sha256: str, expected_self_sha256: str
) -> dict[str, Any]:
    if (
        _SHA256.fullmatch(expected_file_sha256) is None
        or _SHA256.fullmatch(expected_self_sha256) is None
    ):
        raise BircoP1FormalRuntimeError("execution-freeze authority is invalid")
    if _file_sha256(path) != expected_file_sha256:
        raise BircoP1FormalRuntimeError("execution-freeze file hash drifted")
    value = _read_canonical_json(path, label="execution freeze")
    if _verify_self(value, "self_sha256") != expected_self_sha256:
        raise BircoP1FormalRuntimeError("execution-freeze self hash drifted")
    return value


def _terminal_stdout(value: Mapping[str, Any]) -> dict[str, object]:
    status = value.get("status")
    receipt_hash = value.get("final_receipt_sha256")
    if not isinstance(status, str):
        raise BircoP1FormalRuntimeError("controller terminal status is absent")
    if receipt_hash is not None and (
        not isinstance(receipt_hash, str) or _SHA256.fullmatch(receipt_hash) is None
    ):
        raise BircoP1FormalRuntimeError("controller terminal hash drifted")
    return {
        "schema": f"{VERSION}_safe_terminal_stdout_v1",
        "status": status,
        "final_receipt_sha256": receipt_hash,
        "credential_value_included": False,
        "numeric_qrel_value_included": False,
    }


def _write_runtime_failure_once(control_root: Path, exc: BaseException) -> Mapping[str, Any]:
    path = control_root / "runtime_terminal_failure.json"
    if path.exists() or path.is_symlink():
        return _read_canonical_json(
            path, label="runtime terminal failure", expected_mode=0o400
        )
    failure = _self_hashed(
        {
            "schema": f"{VERSION}_terminal_failure_v1",
            "status": "formal_runtime_failed_closed",
            "exception_type_sha256": hashlib.sha256(
                type(exc).__name__.encode("ascii", "replace")
            ).hexdigest(),
            "exception_message_persisted": False,
            "credential_value_persisted": False,
            "numeric_qrel_value_persisted": False,
            "retry_authorized": False,
        },
        "failure_sha256",
    )
    _write_exclusive_json(path, failure, mode=0o400)
    return failure


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--execution-freeze", required=True, type=Path)
    parser.add_argument("--execution-freeze-file-sha256", required=True)
    parser.add_argument("--execution-freeze-self-sha256", required=True)
    parser.add_argument("--plus-env", required=True, type=Path)
    parser.add_argument(
        "--python-executable", type=Path, default=Path(sys.executable)
    )
    parser.add_argument(
        "--strace-executable", type=Path, default=Path("/usr/bin/strace")
    )
    parser.add_argument("--env-executable", type=Path, default=Path("/usr/bin/env"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    control_root = Path(arguments.control_root)
    try:
        project_root = _resolve_cli_path(
            arguments.project_root,
            label="project root",
            must_exist=True,
            directory=True,
        )
        control_root = _resolve_cli_path(
            arguments.control_root,
            label="control root",
            must_exist=False,
        )
        _ensure_private_directory(control_root)
        execution_freeze_path = _resolve_cli_path(
            arguments.execution_freeze,
            label="execution freeze",
            must_exist=True,
        )
        plus_env = _resolve_cli_path(
            arguments.plus_env,
            label="Plus credential environment",
            must_exist=True,
        )
        freeze = _load_execution_freeze_for_paths(
            execution_freeze_path,
            expected_file_sha256=arguments.execution_freeze_file_sha256,
            expected_self_sha256=arguments.execution_freeze_self_sha256,
        )
        selection_binding = freeze.get("selection_receipt_binding")
        if not isinstance(selection_binding, Mapping):
            raise BircoP1FormalRuntimeError("selection receipt binding is absent")
        selection_relative = _safe_relative(
            selection_binding.get("relative_path"), label="selection receipt path"
        )
        selection_root = (project_root / selection_relative).parent.resolve(strict=True)
        runtime_root = control_root / "executor_runtime"
        semantic_executor = SemanticExecutor(
            project_root=project_root,
            runtime_root=runtime_root,
            credential_env_path=plus_env,
            python_executable=arguments.python_executable,
        )
        if freeze.get("provider_identity") != dict(
            semantic_executor.provider_identity
        ):
            raise BircoP1FormalRuntimeError(
                "Plus credential does not match the frozen provider identity"
            )
        hippo_executor = HippoExecutor(
            project_root=project_root,
            runtime_root=runtime_root,
            python_executable=arguments.python_executable,
            strace_executable=arguments.strace_executable,
            env_executable=arguments.env_executable,
        )
        hippo_policy = freeze.get("hipporag_runtime_policy")
        if not isinstance(hippo_policy, Mapping):
            raise BircoP1FormalRuntimeError(
                "HippoRAG runtime policy is absent from the execution freeze"
            )
        hippo_executor.preflight_base_policy(hippo_policy)
        qrel_opener = QrelOpener(
            output_root=selection_root,
            control_root=control_root,
        )
        formal = controller.FormalController(
            project_root=project_root,
            control_root=control_root,
            execution_freeze_path=execution_freeze_path,
            expected_execution_freeze_file_sha256=(
                arguments.execution_freeze_file_sha256
            ),
            expected_execution_freeze_self_sha256=(
                arguments.execution_freeze_self_sha256
            ),
            agent_executor=semantic_executor,
            raw_executor=semantic_executor,
            hipporag_executor=hippo_executor,
            qrel_opener=qrel_opener,
        )
        terminal = formal.run()
        safe = _terminal_stdout(terminal)
        print(_canonical_bytes(safe, newline=False).decode("ascii"), flush=True)
        return 0
    except Exception as exc:
        try:
            _ensure_private_directory(control_root)
            failure = _write_runtime_failure_once(control_root, exc)
            safe_failure = {
                "schema": f"{VERSION}_safe_terminal_stdout_v1",
                "status": "formal_runtime_failed_closed",
                "failure_sha256": failure.get("failure_sha256"),
                "credential_value_included": False,
                "numeric_qrel_value_included": False,
            }
            print(
                _canonical_bytes(safe_failure, newline=False).decode("ascii"),
                flush=True,
            )
        except Exception:
            print(
                _canonical_bytes(
                    {
                        "schema": f"{VERSION}_safe_terminal_stdout_v1",
                        "status": "formal_runtime_failed_closed_before_receipt",
                        "credential_value_included": False,
                        "numeric_qrel_value_included": False,
                    },
                    newline=False,
                ).decode("ascii"),
                flush=True,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BircoP1FormalRuntimeError",
    "HIPPO_CPU_THREADS_PER_PROCESS",
    "HIPPO_GPU_ASSIGNMENT",
    "HIPPO_LOGICAL_SLOT_COUNT",
    "HIPPO_MAXIMUM_PROCESSES_PER_GPU",
    "HippoExecutor",
    "QrelOpener",
    "SemanticExecutor",
    "main",
]
