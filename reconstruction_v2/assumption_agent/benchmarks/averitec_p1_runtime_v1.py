"""Production two-GPU runtime and one-shot entrypoint for AVeriTeC P1.

GPU0 runs the deterministic MiniLM coordinate worker.  GPU1 runs one hardened
official HippoRAG process for the A_hold corpus.  The executors share a
process-group registry so an eager-lane failure terminates the other lane and
its descendants before control returns.  Every launch is claimed exactly
once, all child environments are allow-listed and offline, and private
stdout/stderr/results remain under remote custody.

``source_free_canary`` exercises the same executors on a fixed synthetic
fixture and never accepts a benchmark source path.  ``formal`` creates one
secret, invokes the already-frozen acquisition once, then hands only
label-free views to the source-blind formal controller.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import threading
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import averitec_p1_acquisition_v1 as acquisition
from assumption_agent.benchmarks import averitec_p1_coordinate_worker_v1 as coordinate
from assumption_agent.benchmarks import averitec_p1_formal_controller_v1 as controller
from assumption_agent.benchmarks import averitec_p1_typed_core_v1 as core
from replication_runtime.averitec_p1_official_v1 import worker as official


VERSION = "averitec_p1_runtime_v1"
CONFIG_SCHEMA = f"{VERSION}_formal_config_v1"
CANARY_SCHEMA = f"{VERSION}_source_free_canary_receipt_v1"
ATTEMPT_SCHEMA = f"{VERSION}_formal_attempt_v1"
LAUNCH_SCHEMA = f"{VERSION}_private_launch_receipt_v1"
TOP_FAILURE_SCHEMA = f"{VERSION}_safe_top_level_failure_v1"
COORDINATE_TIMEOUT_SECONDS = 3_600
OFFICIAL_TIMEOUT_SECONDS = 14_400
MAX_ACTIVE_MODEL_PROCESS_GROUPS = 2
PHYSICAL_GPU0 = "0"
PHYSICAL_GPU1 = "1"
LLM_ALIAS = "smollm2"
MINILM_ALIAS = "minilm"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_BLOCK = re.compile(r"[A-Za-z0-9_]+\Z")


class AveritecP1RuntimeError(RuntimeError):
    """The frozen runtime, child process, or one-shot boundary failed."""


def canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise AveritecP1RuntimeError(
            "runtime value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value, newline=False)).hexdigest()


def self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    value = dict(body)
    if "self_sha256" in value:
        raise AveritecP1RuntimeError("runtime self hash already exists")
    value["self_sha256"] = stable_hash(value)
    return value


def _verify_self(value: Mapping[str, object]) -> str:
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if (
        not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or stable_hash(body) != claimed
    ):
        raise AveritecP1RuntimeError("runtime self hash drifted")
    return claimed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_directory(path: Path, *, fresh: bool) -> None:
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=not fresh)
    except OSError as exc:
        raise AveritecP1RuntimeError(
            "private runtime directory cannot be created"
        ) from exc
    if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise AveritecP1RuntimeError("private runtime directory mode drifted")


def _write_once(
    path: Path,
    raw: bytes,
    *,
    final_mode: int,
) -> str:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, final_mode)
    except OSError as exc:
        raise AveritecP1RuntimeError(
            "private runtime artifact cannot be written once"
        ) from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != final_mode
        or path.read_bytes() != raw
    ):
        raise AveritecP1RuntimeError(
            "private runtime artifact verification failed"
        )
    return hashlib.sha256(raw).hexdigest()


def _write_json_once(
    path: Path, value: Mapping[str, object], *, final_mode: int
) -> str:
    return _write_once(path, canonical_bytes(value), final_mode=final_mode)


def _read_canonical(path: Path, *, mode: int) -> dict[str, object]:
    try:
        info = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AveritecP1RuntimeError(
            "runtime artifact is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != mode
        or not isinstance(value, dict)
        or raw != canonical_bytes(value)
    ):
        raise AveritecP1RuntimeError("runtime artifact metadata drifted")
    return value


def _absolute_existing(
    path: str,
    field: str,
    *,
    directory: bool,
    allow_final_symlink: bool = False,
) -> Path:
    candidate = Path(path)
    if (
        not candidate.is_absolute()
        or (candidate.is_symlink() and not allow_final_symlink)
    ):
        raise AveritecP1RuntimeError(f"{field} is not a direct absolute path")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AveritecP1RuntimeError(f"{field} is unavailable") from exc
    if not allow_final_symlink and resolved != candidate:
        raise AveritecP1RuntimeError(f"{field} traverses a symlink")
    if directory != resolved.is_dir():
        raise AveritecP1RuntimeError(f"{field} type drifted")
    if not directory and not resolved.is_file():
        raise AveritecP1RuntimeError(f"{field} is not a regular file")
    return resolved


@dataclass(frozen=True)
class RuntimePaths:
    project_root: str
    typed_python: str
    typed_site_root: str
    official_python: str
    official_overlay_root: str
    hipporag_source_root: str
    p16_site_root: str
    official_base_site_root: str
    smollm_model_root: str
    minilm_model_root: str
    strace_path: str

    def validate(self) -> None:
        for field in (
            "project_root",
            "typed_site_root",
            "official_overlay_root",
            "hipporag_source_root",
            "p16_site_root",
            "official_base_site_root",
            "smollm_model_root",
            "minilm_model_root",
        ):
            _absolute_existing(getattr(self, field), field, directory=True)
        for field in ("typed_python", "official_python"):
            _absolute_existing(
                getattr(self, field),
                field,
                directory=False,
                allow_final_symlink=True,
            )
        _absolute_existing(self.strace_path, "strace_path", directory=False)

    def typed_pythonpath(self) -> str:
        return f"{self.project_root}:{self.typed_site_root}"

    def official_pythonpath(self) -> str:
        return ":".join(
            (
                self.project_root,
                self.official_overlay_root,
                self.hipporag_source_root,
                self.p16_site_root,
                self.official_base_site_root,
            )
        )


@dataclass(frozen=True)
class FormalConfig:
    execution_binding_sha256: str
    source_root: str
    p0_receipt_path: str
    work_root: str
    runtime: RuntimePaths
    schema: str = CONFIG_SCHEMA
    study_id: str = core.STUDY_ID

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "FormalConfig":
        if set(value) != {
            "execution_binding_sha256",
            "p0_receipt_path",
            "runtime",
            "schema",
            "self_sha256",
            "source_root",
            "study_id",
            "work_root",
        }:
            raise AveritecP1RuntimeError("formal config envelope drifted")
        _verify_self(value)
        runtime = value.get("runtime")
        if not isinstance(runtime, Mapping) or set(runtime) != {
            "hipporag_source_root",
            "minilm_model_root",
            "official_base_site_root",
            "official_overlay_root",
            "official_python",
            "p16_site_root",
            "project_root",
            "smollm_model_root",
            "strace_path",
            "typed_python",
            "typed_site_root",
        }:
            raise AveritecP1RuntimeError("formal runtime config drifted")
        result = cls(
            execution_binding_sha256=str(
                value["execution_binding_sha256"]
            ),
            source_root=str(value["source_root"]),
            p0_receipt_path=str(value["p0_receipt_path"]),
            work_root=str(value["work_root"]),
            runtime=RuntimePaths(**{key: str(row) for key, row in runtime.items()}),
            schema=str(value["schema"]),
            study_id=str(value["study_id"]),
        )
        if (
            result.schema != CONFIG_SCHEMA
            or result.study_id != core.STUDY_ID
            or _HEX64.fullmatch(result.execution_binding_sha256) is None
            or not Path(result.source_root).is_absolute()
            or not Path(result.p0_receipt_path).is_absolute()
            or not Path(result.work_root).is_absolute()
        ):
            raise AveritecP1RuntimeError("formal config binding drifted")
        result.runtime.validate()
        return result


class _ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[bytes]] = set()

    def add(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._processes.add(process)
            if len(self._processes) > MAX_ACTIVE_MODEL_PROCESS_GROUPS:
                raise AveritecP1RuntimeError(
                    "active model process group cap drifted"
                )

    def discard(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._processes.discard(process)

    def cancel_all(self) -> None:
        with self._lock:
            processes = tuple(self._processes)
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def _short_alias(directory: Path, alias: str, target: str) -> None:
    target_path = _absolute_existing(target, f"{alias} model", directory=True)
    link = directory / alias
    try:
        link.symlink_to(target_path, target_is_directory=True)
        resolved = link.resolve(strict=True)
    except OSError as exc:
        raise AveritecP1RuntimeError("short model alias failed") from exc
    if (
        not link.is_symlink()
        or os.readlink(link) != str(target_path)
        or resolved != target_path
        or not os.path.samefile(resolved, target_path)
    ):
        raise AveritecP1RuntimeError("short model alias binding drifted")


def _child_environment(
    *,
    paths: RuntimePaths,
    scratch: Path,
    physical_gpu: str,
    pythonpath: str,
    python: str,
) -> dict[str, str]:
    if physical_gpu not in {PHYSICAL_GPU0, PHYSICAL_GPU1}:
        raise AveritecP1RuntimeError("physical GPU binding drifted")
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": coordinate.CUBLAS_WORKSPACE_CONFIG,
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_VISIBLE_DEVICES": physical_gpu,
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(scratch / "cache"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": str(scratch / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": f"{Path(python).parent}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": pythonpath,
        "TEMP": str(scratch / "tmp"),
        "TMP": str(scratch / "tmp"),
        "TMPDIR": str(scratch / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    if any("PROXY" in key.upper() or "API" in key.upper() for key in environment):
        raise AveritecP1RuntimeError(
            "child environment admitted a provider or proxy field"
        )
    return environment


def _open_log(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "wb")


def _network_audit(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise AveritecP1RuntimeError("network syscall audit is unavailable")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AveritecP1RuntimeError("network syscall audit is invalid") from exc
    ip_family_rows = [
        line
        for line in text.splitlines()
        if "AF_INET" in line or "AF_INET6" in line
    ]
    nonblocked = [
        line
        for line in ip_family_rows
        if "= -1 EAFNOSUPPORT" not in line
    ]
    if nonblocked:
        raise AveritecP1RuntimeError(
            "official worker made an IP-family syscall not blocked by "
            "RestrictAddressFamilies"
        )
    return {
        "blocked_IP_family_syscall_count": len(ip_family_rows),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "nonblocked_IP_family_syscall_count": len(nonblocked),
        "size_bytes": len(raw),
    }


class _BaseExecutor:
    def __init__(
        self,
        *,
        paths: RuntimePaths,
        execution_root: Path,
        registry: _ProcessRegistry,
    ) -> None:
        self.paths = paths
        self.execution_root = execution_root
        self.registry = registry

    def cancel_all(self) -> None:
        self.registry.cancel_all()

    def _run(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
        custody: Path,
        timeout_seconds: int,
    ) -> dict[str, object]:
        stdout_path = custody / "worker.stdout.private.bin"
        stderr_path = custody / "worker.stderr.private.bin"
        with _open_log(stdout_path) as stdout, _open_log(stderr_path) as stderr:
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=cwd,
                    env=dict(environment),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except OSError as exc:
                raise AveritecP1RuntimeError(
                    "model worker launch failed; retry is forbidden"
                ) from exc
            self.registry.add(process)
            try:
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise AveritecP1RuntimeError(
                        "model worker timed out; retry is forbidden"
                    ) from exc
            finally:
                self.registry.discard(process)
                stdout.flush()
                stderr.flush()
                os.fsync(stdout.fileno())
                os.fsync(stderr.fileno())
        receipt = {
            "returncode": returncode,
            "stderr_sha256": _sha256_file(stderr_path),
            "stderr_size_bytes": stderr_path.stat().st_size,
            "stdout_sha256": _sha256_file(stdout_path),
            "stdout_size_bytes": stdout_path.stat().st_size,
        }
        if returncode != 0:
            raise AveritecP1RuntimeError(
                "model worker exited nonzero; retry is forbidden"
            )
        return receipt

    def _roots(self, *, lane: str, block: str) -> tuple[Path, Path]:
        if _SAFE_BLOCK.fullmatch(block) is None:
            raise AveritecP1RuntimeError("runtime block name drifted")
        custody = self.execution_root / "custody" / block / lane
        scratch = self.execution_root / "scratch" / block / lane
        _private_directory(custody, fresh=True)
        _private_directory(scratch, fresh=True)
        for name in ("cache", "home", "tmp", "model_aliases"):
            _private_directory(scratch / name, fresh=True)
        return custody, scratch

    @staticmethod
    def _claim(
        *,
        custody: Path,
        block: str,
        lane: str,
        input_sha256: str,
    ) -> str:
        claim = self_hashed(
            {
                "block": block,
                "input_sha256": input_sha256,
                "lane": lane,
                "retry_replay_or_provider_switch_authorized": False,
                "schema": f"{VERSION}_private_attempt_claim_v1",
                "study_id": core.STUDY_ID,
            }
        )
        return _write_json_once(
            custody / "attempt.private.json", claim, final_mode=0o400
        )


class CoordinateProductionExecutor(_BaseExecutor):
    def __call__(
        self,
        *,
        block: str,
        private_input: Mapping[str, object],
    ) -> Mapping[str, object]:
        coordinate.validate_input(private_input)
        custody, scratch = self._roots(lane="coordinate_gpu0", block=block)
        input_sha = coordinate.stable_hash(private_input)
        claim_sha = self._claim(
            custody=custody,
            block=block,
            lane="coordinate_gpu0",
            input_sha256=input_sha,
        )
        input_path = scratch / "input.private.json"
        output_path = scratch / "output.private.json"
        _write_once(
            input_path,
            coordinate.canonical_bytes(private_input),
            final_mode=0o600,
        )
        alias_cwd = scratch / "model_aliases"
        _short_alias(alias_cwd, MINILM_ALIAS, self.paths.minilm_model_root)
        command = [
            self.paths.typed_python,
            "-S",
            "-B",
            "-m",
            "assumption_agent.benchmarks.averitec_p1_coordinate_worker_v1",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model-root",
            MINILM_ALIAS,
        ]
        if self.paths.minilm_model_root in command:
            raise AveritecP1RuntimeError(
                "absolute MiniLM path escaped into worker argv"
            )
        environment = _child_environment(
            paths=self.paths,
            scratch=scratch,
            physical_gpu=PHYSICAL_GPU0,
            pythonpath=self.paths.typed_pythonpath(),
            python=self.paths.typed_python,
        )
        launch = self._run(
            command=command,
            cwd=alias_cwd,
            environment=environment,
            custody=custody,
            timeout_seconds=COORDINATE_TIMEOUT_SECONDS,
        )
        output = coordinate.read_private_output(
            output_path, expected_input=private_input
        )
        output_file_sha = _write_json_once(
            custody / "output.private.json", output, final_mode=0o400
        )
        launch_receipt = self_hashed(
            {
                "attempt_claim_file_sha256": claim_sha,
                "block": block,
                "child_environment_key_count": len(environment),
                "input_sha256": input_sha,
                "lane": "coordinate_gpu0",
                "launch": launch,
                "output_file_sha256": output_file_sha,
                "output_self_sha256": output["self_sha256"],
                "physical_gpu": PHYSICAL_GPU0,
                "schema": LAUNCH_SCHEMA,
                "study_id": core.STUDY_ID,
            }
        )
        _write_json_once(
            custody / "launch.private.json",
            launch_receipt,
            final_mode=0o400,
        )
        shutil.rmtree(scratch)
        return output


class HippoProductionExecutor(_BaseExecutor):
    def __call__(
        self,
        *,
        block: str,
        articles: Sequence[Mapping[str, object]],
        queries: Sequence[tuple[str, str]],
    ) -> controller.HippoResult:
        if block not in {acquisition.A_HOLD, official.CANARY_BLOCK}:
            raise AveritecP1RuntimeError(
                "official HippoRAG lane is not authorized for this block"
            )
        private_input = official.input_payload(
            block=block,
            articles=articles,
            queries=queries,
        )
        custody, scratch = self._roots(lane="official_gpu1", block=block)
        input_sha = official.stable_hash(private_input)
        claim_sha = self._claim(
            custody=custody,
            block=block,
            lane="official_gpu1",
            input_sha256=input_sha,
        )
        input_path = scratch / "input.private.json"
        output_path = scratch / "output.private.json"
        index_root = scratch / "index"
        network_path = custody / "network.private.strace"
        _write_once(
            input_path,
            official.canonical_bytes(private_input),
            final_mode=0o600,
        )
        alias_cwd = scratch / "model_aliases"
        _short_alias(alias_cwd, LLM_ALIAS, self.paths.smollm_model_root)
        _short_alias(alias_cwd, MINILM_ALIAS, self.paths.minilm_model_root)
        worker_command = [
            self.paths.official_python,
            "-S",
            "-B",
            "-m",
            "replication_runtime.averitec_p1_official_v1.worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--index-root",
            str(index_root),
            "--llm-model",
            LLM_ALIAS,
            "--embedding-model",
            MINILM_ALIAS,
            "--hipporag-source-root",
            self.paths.hipporag_source_root,
            "--project-root",
            self.paths.project_root,
        ]
        if any(
            absolute in worker_command
            for absolute in (
                self.paths.smollm_model_root,
                self.paths.minilm_model_root,
            )
        ):
            raise AveritecP1RuntimeError(
                "absolute model path escaped into official worker argv"
            )
        command = [
            self.paths.strace_path,
            "-f",
            "-qq",
            "-e",
            "trace=socket,connect",
            "-o",
            str(network_path),
            *worker_command,
        ]
        environment = _child_environment(
            paths=self.paths,
            scratch=scratch,
            physical_gpu=PHYSICAL_GPU1,
            pythonpath=self.paths.official_pythonpath(),
            python=self.paths.official_python,
        )
        launch = self._run(
            command=command,
            cwd=alias_cwd,
            environment=environment,
            custody=custody,
            timeout_seconds=OFFICIAL_TIMEOUT_SECONDS,
        )
        network = _network_audit(network_path)
        raw = output_path.read_bytes()
        try:
            parsed = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AveritecP1RuntimeError(
                "official output cannot be parsed"
            ) from exc
        if (
            output_path.is_symlink()
            or stat.S_IMODE(output_path.stat().st_mode) != 0o600
            or not isinstance(parsed, dict)
            or raw != official.canonical_bytes(parsed)
        ):
            raise AveritecP1RuntimeError(
                "official output metadata drifted"
            )
        output = official.validate_output(
            parsed, expected_input=private_input
        )
        output_file_sha = _write_json_once(
            custody / "output.private.json", output, final_mode=0o400
        )
        receipt = output["receipt"]
        if not isinstance(receipt, Mapping):
            raise AveritecP1RuntimeError("official receipt disappeared")
        launch_receipt = self_hashed(
            {
                "attempt_claim_file_sha256": claim_sha,
                "block": block,
                "child_environment_key_count": len(environment),
                "input_sha256": input_sha,
                "lane": "official_gpu1",
                "launch": launch,
                "network_syscall_audit": network,
                "output_file_sha256": output_file_sha,
                "output_self_sha256": output["self_sha256"],
                "physical_gpu": PHYSICAL_GPU1,
                "schema": LAUNCH_SCHEMA,
                "study_id": core.STUDY_ID,
            }
        )
        _write_json_once(
            custody / "launch.private.json",
            launch_receipt,
            final_mode=0o400,
        )
        indices = tuple(
            tuple(int(value) for value in row["top5_document_ordinals"])
            for row in output["rows"]  # type: ignore[union-attr]
        )
        result = controller.HippoResult(
            indices=indices,
            receipt_sha256=str(receipt["self_sha256"]),
            build_receipt_sha256=str(receipt["index_tree_sha256"]),
        )
        shutil.rmtree(scratch)
        return result


def _executors(
    *,
    paths: RuntimePaths,
    execution_root: Path,
) -> tuple[CoordinateProductionExecutor, HippoProductionExecutor]:
    paths.validate()
    _private_directory(execution_root, fresh=True)
    _private_directory(execution_root / "custody", fresh=True)
    _private_directory(execution_root / "scratch", fresh=True)
    registry = _ProcessRegistry()
    return (
        CoordinateProductionExecutor(
            paths=paths, execution_root=execution_root, registry=registry
        ),
        HippoProductionExecutor(
            paths=paths, execution_root=execution_root, registry=registry
        ),
    )


def run_source_free_canary(
    *,
    paths: RuntimePaths,
    canary_root: Path,
    execution_binding_sha256: str,
) -> dict[str, object]:
    if _HEX64.fullmatch(execution_binding_sha256) is None:
        raise AveritecP1RuntimeError("canary execution binding drifted")
    if canary_root.exists() or canary_root.is_symlink():
        raise AveritecP1RuntimeError("source-free canary root is not fresh")
    coordinate_executor, hippo_executor = _executors(
        paths=paths, execution_root=canary_root
    )
    documents = [
        f"Synthetic title {ordinal}\n\nSynthetic body {ordinal} establishes "
        f"a distinct relation without benchmark content."
        for ordinal in range(6)
    ]
    queries = [
        ("a" * 64, "Synthetic claim about a cause and effect."),
        ("b" * 64, "Synthetic claim quoting a source with 42 units."),
    ]
    coordinate_input = coordinate.private_input_payload(
        documents=documents, queries=queries
    )
    articles = [
        {
            "body": f"Synthetic body {ordinal} establishes a distinct "
            f"relation without benchmark content.",
            "idx": ordinal,
            "title": f"Synthetic title {ordinal}",
        }
        for ordinal in range(6)
    ]
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="averitec-source-free-two-gpu"
    ) as pool:
        coordinate_future = pool.submit(
            coordinate_executor,
            block=official.CANARY_BLOCK,
            private_input=coordinate_input,
        )
        hippo_future = pool.submit(
            hippo_executor,
            block=official.CANARY_BLOCK,
            articles=articles,
            queries=queries,
        )
        try:
            coordinate_output = coordinate_future.result()
            hippo = hippo_future.result()
        except BaseException:
            coordinate_executor.cancel_all()
            hippo_executor.cancel_all()
            raise
    coordinate.validate_output(
        coordinate_output, expected_input=coordinate_input
    )
    rows = coordinate_output["rows"]
    if not isinstance(rows, list):
        raise AveritecP1RuntimeError("canary coordinate rows disappeared")
    actions = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise AveritecP1RuntimeError("canary coordinate row drifted")
        slate = core.materialize_recipe_actions(
            document_texts=documents,
            variant_scores=row["variant_scores"],  # type: ignore[arg-type]
        )
        actions.append(slate)
    # Exercise exact utility, E1 fit/roundtrip, frozen selection and reference
    # tail without creating a benchmark cohort or effect gate.
    slates = []
    for action_map in actions:
        qrels = (0, 1)
        slates.append(
            core.AFormSlate(
                tuple(
                    core.AFormAction(
                        recipe_id=recipe_id,
                        features=core.compute_action_features(
                            action=action_map[recipe_id],
                            document_texts=documents,
                        ),
                        utility=core.utility(
                            top5_document_ordinals=action_map[
                                recipe_id
                            ].top5_document_ordinals,
                            qrel_document_ordinals=qrels,
                        ),
                    )
                    for recipe_id in core.RECIPE_IDS
                )
            )
        )
    model = core.model_from_payload(core.model_payload(core.fit_e1(slates)))
    selected = [
        core.select_e1(
            model=model, actions=action_map, document_texts=documents
        )
        for action_map in actions
    ]
    synthetic_tail = core.compare(
        [core.utility(top5_document_ordinals=(0, 1, 2, 3, 4), qrel_document_ordinals=(0, 1))] * 4,
        [core.utility(top5_document_ordinals=(1, 2, 3, 4, 5), qrel_document_ordinals=(0, 1))] * 4,
    )
    receipt = self_hashed(
        {
            "API_or_online_evaluator_call_count": 0,
            "benchmark_source_archive_row_or_label_access_count": 0,
            "coordinate_output_self_sha256": coordinate_output["self_sha256"],
            "execution_binding_sha256": execution_binding_sha256,
            "formal_secret_cohort_action_or_score_count": 0,
            "hipporag_build_receipt_sha256": hippo.build_receipt_sha256,
            "hipporag_retrieval_receipt_sha256": hippo.receipt_sha256,
            "max_concurrent_physical_model_lanes": 2,
            "model_inference_process_count": 2,
            "offline_exact_reference_tail": {
                "denominator": synthetic_tail.reference_tail.denominator,
                "numerator": synthetic_tail.reference_tail.numerator,
            },
            "schema": CANARY_SCHEMA,
            "selected_recipe_count": dict(
                sorted({recipe: selected.count(recipe) for recipe in set(selected)}.items())
            ),
            "status": "qualified_source_free_two_gpu_full_path",
            "study_id": core.STUDY_ID,
        }
    )
    _write_json_once(
        canary_root / "source_free_canary.safe.json",
        receipt,
        final_mode=0o400,
    )
    return receipt


def _top_failure(
    *,
    work_root: Path,
    execution_binding_sha256: str,
    stage: str,
    exc: BaseException,
) -> None:
    path = work_root / "formal_terminal.json"
    if path.exists() or path.is_symlink():
        return
    receipt = self_hashed(
        {
            "exception_message_sha256": hashlib.sha256(
                str(exc).encode("utf-8")
            ).hexdigest(),
            "exception_type_sha256": hashlib.sha256(
                type(exc).__qualname__.encode("utf-8")
            ).hexdigest(),
            "execution_binding_sha256": execution_binding_sha256,
            "formal_retry_authorized": False,
            "online_evaluator_fallback_authorized": False,
            "schema": TOP_FAILURE_SCHEMA,
            "stage": stage,
            "status": "implementation_or_infrastructure_invalid",
            "study_id": core.STUDY_ID,
        }
    )
    _write_json_once(path, receipt, final_mode=0o400)


def run_formal(config: FormalConfig) -> dict[str, object]:
    work_root = Path(config.work_root)
    if work_root.exists() or work_root.is_symlink():
        raise AveritecP1RuntimeError("formal work root is not fresh")
    _private_directory(work_root, fresh=True)
    stage = "formal_attempt_before_source_or_model"
    attempt = self_hashed(
        {
            "execution_binding_sha256": config.execution_binding_sha256,
            "retry_replay_or_resample_authorized": False,
            "schema": ATTEMPT_SCHEMA,
            "study_id": core.STUDY_ID,
        }
    )
    _write_json_once(
        work_root / "formal.attempt.json", attempt, final_mode=0o400
    )
    try:
        stage = "secret_creation"
        secret_path = work_root / "selection.secret.bin"
        _write_once(secret_path, os.getrandom(32), final_mode=0o600)

        stage = "one_shot_source_acquisition"
        acquisition_root = work_root / "acquisition"
        acquisition.run_acquisition(
            source_root=Path(config.source_root),
            p0_receipt_path=Path(config.p0_receipt_path),
            secret_path=secret_path,
            attempt_marker_path=work_root / "acquisition.attempt.json",
            output_root=acquisition_root,
            execution_binding_sha256=config.execution_binding_sha256,
        )

        stage = "production_executor_initialization"
        coordinate_executor, hippo_executor = _executors(
            paths=config.runtime,
            execution_root=work_root / "executor",
        )
        stage = "source_blind_formal_controller"
        formal_controller = controller.FormalController(
            acquisition_root=acquisition_root,
            work_root=work_root / "formal_study",
            execution_binding_sha256=config.execution_binding_sha256,
            coordinate_executor=coordinate_executor,
            hippo_executor=hippo_executor,
        )
        terminal = formal_controller.run()
        controller_terminal = (
            work_root / "formal_study" / "formal_terminal.json"
        )
        if _read_canonical(controller_terminal, mode=0o400) != terminal:
            raise AveritecP1RuntimeError(
                "controller safe terminal revalidation failed"
            )
        os.link(controller_terminal, work_root / "formal_terminal.json")
        if stat.S_IMODE(
            (work_root / "formal_terminal.json").stat().st_mode
        ) != 0o400:
            raise AveritecP1RuntimeError(
                "top-level formal terminal mode drifted"
            )
        return terminal
    except BaseException as exc:
        nested = work_root / "formal_study" / "formal_terminal.json"
        top = work_root / "formal_terminal.json"
        if nested.is_file() and not top.exists():
            os.link(nested, top)
        else:
            _top_failure(
                work_root=work_root,
                execution_binding_sha256=config.execution_binding_sha256,
                stage=stage,
                exc=exc,
            )
        raise


def _runtime_paths_from_json(path: Path) -> RuntimePaths:
    payload = _read_canonical(path, mode=0o400)
    if set(payload) != {
        "hipporag_source_root",
        "minilm_model_root",
        "official_base_site_root",
        "official_overlay_root",
        "official_python",
        "p16_site_root",
        "project_root",
        "smollm_model_root",
        "strace_path",
        "typed_python",
        "typed_site_root",
    }:
        raise AveritecP1RuntimeError("runtime path payload drifted")
    paths = RuntimePaths(**{key: str(value) for key, value in payload.items()})
    paths.validate()
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    canary = subparsers.add_parser("source_free_canary")
    canary.add_argument("--runtime-paths", required=True, type=Path)
    canary.add_argument("--canary-root", required=True, type=Path)
    canary.add_argument("--execution-binding-sha256", required=True)
    formal = subparsers.add_parser("formal")
    formal.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.mode == "source_free_canary":
        receipt = run_source_free_canary(
            paths=_runtime_paths_from_json(arguments.runtime_paths),
            canary_root=arguments.canary_root,
            execution_binding_sha256=arguments.execution_binding_sha256,
        )
    else:
        config_payload = _read_canonical(arguments.config, mode=0o400)
        receipt = run_formal(FormalConfig.from_payload(config_payload))
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "self_sha256": receipt["self_sha256"],
                "status": receipt["status"],
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
