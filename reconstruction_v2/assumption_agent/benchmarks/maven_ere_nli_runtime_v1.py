"""Exact-two-worker offline NLI runtime for the frozen MAVEN-ERE study."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from replication_runtime.qasc_nli_v1 import binding as nli_binding
from replication_runtime.qasc_nli_v1.contract import (
    MAXIMUM_RESPONSE_BYTES,
    NLIPair,
    QASCNLIError,
    decode_response,
    encode_request,
)
from replication_runtime.qasc_nli_v1.worker import canonical_canary_pairs


VERSION = "maven_ere_nli_runtime_v1"
DESIGN_RELATIVE = Path("manifests/maven_ere_g8_e1_formal_design_v1.json")
DESIGN_FILE_SHA256 = "e8ae662809ead29f2a5c08fd0ca44970ef8916ccda3741f480b87b571f44ddf4"
DESIGN_SELF_SHA256 = "314a9804d32a3c3fb848e0100bc62bc693a468e8e3ac09c9baf018c7cfeee417"
WORKER_COUNT = 2
TORCH_THREADS_PER_WORKER = 4
CANARY_SCORE_VECTOR_SHA256 = (
    "a06fba0eea950a61c0599b76169f06a2d77360388c1454fdc7acf9a0a4d2467f"
)


class MavenEreNLIRuntimeError(RuntimeError):
    """The frozen design, asset, worker, or wire contract drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MavenEreNLIRuntimeError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_maven_design(project_root: str | Path) -> Mapping[str, object]:
    root = Path(project_root).resolve(strict=True)
    path = root / DESIGN_RELATIVE
    if path.is_symlink() or not path.is_file() or _sha256_file(path) != DESIGN_FILE_SHA256:
        raise MavenEreNLIRuntimeError("MAVEN-ERE design file drifted")
    try:
        design = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreNLIRuntimeError("MAVEN-ERE design is invalid") from exc
    if not isinstance(design, dict):
        raise MavenEreNLIRuntimeError("MAVEN-ERE design must be an object")
    body = dict(design)
    declared = body.pop("study_design_sha256", None)
    runtime = design.get("offline_runtime")
    nli = runtime.get("NLI") if isinstance(runtime, Mapping) else None
    if (
        declared != DESIGN_SELF_SHA256
        or stable_hash(body) != DESIGN_SELF_SHA256
        or not isinstance(nli, Mapping)
        or nli.get("asset_sha256") != nli_binding.ASSET_SELF_SHA256
        or nli.get("asset_file_sha256") != nli_binding.ASSET_FILE_SHA256
        or nli.get("model") != nli_binding.MODEL_ID
        or nli.get("revision") != nli_binding.MODEL_REVISION
        or nli.get("worker_count") != WORKER_COUNT
        or nli.get("worker_torch_threads_each") != TORCH_THREADS_PER_WORKER
    ):
        raise MavenEreNLIRuntimeError("MAVEN-ERE NLI design binding drifted")
    return MappingProxyType(
        {
            "design_file_sha256": DESIGN_FILE_SHA256,
            "design_sha256": DESIGN_SELF_SHA256,
            "NLI_asset_sha256": nli_binding.ASSET_SELF_SHA256,
            "torch_threads_per_worker": TORCH_THREADS_PER_WORKER,
            "worker_count": WORKER_COUNT,
            "status": "verified_maven_exact_two_worker_NLI_design",
        }
    )


def _worker_environment(project: Path) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(project),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }


class _PersistentWorker:
    def __init__(
        self,
        *,
        runtime_python: str | Path,
        asset_manifest_path: Path,
        model_root: Path,
        project_root: Path,
    ) -> None:
        command = [
            str(runtime_python),
            "-B",
            "-m",
            "replication_runtime.qasc_nli_v1.worker",
            "--asset-manifest",
            str(asset_manifest_path),
            "--model-root",
            str(model_root),
            "--serve-jsonl",
        ]
        try:
            self.process = subprocess.Popen(
                command,
                cwd=project_root,
                env=_worker_environment(project_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise MavenEreNLIRuntimeError("NLI worker failed to start") from exc
        if (
            self.process.stdin is None
            or self.process.stdout is None
            or self.process.stderr is None
        ):
            self.process.kill()
            raise MavenEreNLIRuntimeError("NLI worker pipes are unavailable")
        self._lock = threading.Lock()
        self._stderr_tail: deque[bytes] = deque(maxlen=32)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in iter(self.process.stderr.readline, b""):
            self._stderr_tail.append(line[-4096:])

    def score(self, request: bytes, *, expected_count: int) -> tuple[int, ...]:
        with self._lock:
            if self.process.poll() is not None:
                raise MavenEreNLIRuntimeError("NLI worker exited")
            assert self.process.stdin is not None and self.process.stdout is not None
            try:
                self.process.stdin.write(request)
                self.process.stdin.flush()
                raw = self.process.stdout.readline(MAXIMUM_RESPONSE_BYTES + 1)
            except (BrokenPipeError, OSError) as exc:
                raise MavenEreNLIRuntimeError("NLI worker pipe failed") from exc
            if not raw or len(raw) > MAXIMUM_RESPONSE_BYTES:
                raise MavenEreNLIRuntimeError("NLI worker response is invalid")
            try:
                return decode_response(raw, expected_count=expected_count)
            except QASCNLIError as exc:
                raise MavenEreNLIRuntimeError("NLI worker response drifted") from exc

    def close(self) -> None:
        with self._lock:
            if self.process.poll() is None:
                if self.process.stdin is not None:
                    try:
                        self.process.stdin.close()
                    except OSError:
                        pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)


@dataclass(frozen=True)
class MavenEreNLIPoolReceipt:
    design: Mapping[str, object]
    runtime: Mapping[str, object]
    canary: Mapping[str, object]
    receipt_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "canary": dict(self.canary),
            "design": dict(self.design),
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "receipt_sha256": self.receipt_sha256,
            "runtime": dict(self.runtime),
            "schema": "maven_ere_nli_pool_receipt_v1",
            "version": VERSION,
            "worker_count": WORKER_COUNT,
        }


class MavenEreNLIWorkerPool:
    """Concurrency-safe exact-two worker pool over the immutable QASC wire."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        project_root: str | Path,
        runtime_python: str | Path = sys.executable,
    ) -> None:
        root = Path(project_root).resolve(strict=True)
        design = verify_maven_design(root)
        model = Path(model_path).resolve(strict=True)
        try:
            runtime = nli_binding.verify_runtime_asset(root, model)
        except QASCNLIError as exc:
            raise MavenEreNLIRuntimeError("NLI runtime binding failed") from exc
        asset_manifest = root / nli_binding.ASSET_RELATIVE_PATH
        self._state_lock = threading.Lock()
        self._failed = False
        self._closed = False
        self._workers: list[_PersistentWorker] = []
        self._available: queue.Queue[_PersistentWorker] = queue.Queue()
        try:
            for _ in range(WORKER_COUNT):
                worker = _PersistentWorker(
                    runtime_python=runtime_python,
                    asset_manifest_path=asset_manifest,
                    model_root=model,
                    project_root=root,
                )
                self._workers.append(worker)
            request = encode_request(canonical_canary_pairs())
            vectors: list[tuple[int, ...]] = []
            with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
                for _repeat in range(2):
                    futures = [
                        executor.submit(
                            worker.score,
                            request,
                            expected_count=len(canonical_canary_pairs()),
                        )
                        for worker in self._workers
                    ]
                    vectors.extend(future.result() for future in futures)
            hashes = tuple(stable_hash(list(row)) for row in vectors)
            if (
                len(vectors) != 2 * WORKER_COUNT
                or len(set(vectors)) != 1
                or hashes != (CANARY_SCORE_VECTOR_SHA256,) * (2 * WORKER_COUNT)
            ):
                raise MavenEreNLIRuntimeError("NLI startup canary drifted")
            for worker in self._workers:
                self._available.put(worker)
        except BaseException:
            self.close()
            raise
        canary = {
            "integer_score_vector_sha256": CANARY_SCORE_VECTOR_SHA256,
            "pair_count": len(canonical_canary_pairs()),
            "repeat_count_per_worker": 2,
            "repeat_exact": True,
            "worker_count": WORKER_COUNT,
        }
        body = {
            "canary": canary,
            "design": dict(design),
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "runtime": dict(runtime),
            "schema": "maven_ere_nli_pool_receipt_v1",
            "version": VERSION,
            "worker_count": WORKER_COUNT,
        }
        self.receipt = MavenEreNLIPoolReceipt(
            design=MappingProxyType(dict(design)),
            runtime=MappingProxyType(dict(runtime)),
            canary=MappingProxyType(canary),
            receipt_sha256=stable_hash(body),
        )

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    def _require_live(self) -> None:
        with self._state_lock:
            if self._closed:
                raise MavenEreNLIRuntimeError("NLI pool is closed")
            if self._failed:
                raise MavenEreNLIRuntimeError("NLI pool is poisoned")

    def score_pairs(self, pairs: Sequence[Mapping[str, object] | NLIPair]) -> tuple[int, ...]:
        request = encode_request(pairs)
        self._require_live()
        while True:
            self._require_live()
            try:
                worker = self._available.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        return_worker = True
        try:
            return worker.score(request, expected_count=len(pairs))
        except BaseException:
            return_worker = False
            with self._state_lock:
                self._failed = True
            raise
        finally:
            if return_worker:
                self._available.put(worker)

    def score_items(
        self,
        items: Sequence[tuple[str, Sequence[Mapping[str, object] | NLIPair]]],
    ) -> dict[str, tuple[int, ...]]:
        keys: list[str] = []
        batches: list[Sequence[Mapping[str, object] | NLIPair]] = []
        seen: set[str] = set()
        for key, pairs in items:
            if not isinstance(key, str) or not key or key in seen:
                raise MavenEreNLIRuntimeError("item keys must be unique text")
            encode_request(pairs)
            keys.append(key)
            batches.append(pairs)
            seen.add(key)
        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
            futures = [executor.submit(self.score_pairs, pairs) for pairs in batches]
            results = tuple(future.result() for future in futures)
        return {key: row for key, row in zip(keys, results, strict=True)}

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        for worker in self._workers:
            worker.close()

    def __enter__(self) -> "MavenEreNLIWorkerPool":
        self._require_live()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "CANARY_SCORE_VECTOR_SHA256",
    "MavenEreNLIPoolReceipt",
    "MavenEreNLIRuntimeError",
    "MavenEreNLIWorkerPool",
    "TORCH_THREADS_PER_WORKER",
    "VERSION",
    "WORKER_COUNT",
    "stable_hash",
    "verify_maven_design",
]
