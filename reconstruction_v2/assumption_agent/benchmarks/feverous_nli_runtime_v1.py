"""Eight-worker, FEVEROUS-bound wrapper for the frozen offline NLI asset.

The QASC worker protocol and model asset are reusable, but QASC's public pool
also verifies a QASC-specific study design.  This module owns the much smaller
FEVEROUS seam: it verifies the committed FEVEROUS design, starts exactly eight
of the unchanged local workers, addresses every child with the frozen row-free
canary twice, and then exposes only the pair-scoring protocol.

No FEVEROUS row, item key, label, family, evidence id, retrieval result, or
online service can cross the worker boundary.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from replication_runtime.qasc_nli_v1 import binding as qasc_binding
from replication_runtime.qasc_nli_v1.contract import (
    MAXIMUM_RESPONSE_BYTES,
    NLIPair,
    QASCNLIError,
    decode_response,
    encode_request,
)
from replication_runtime.qasc_nli_v1.worker import canonical_canary_pairs


VERSION = "feverous_nli_runtime_v1"
DESIGN_RELATIVE_PATH = Path("manifests/feverous_p6_e2_evaluator_design_v1.json")
DESIGN_SHA256 = "6193646baca9e35820a5d157bc248012fbd478c89a45db7d879295c4d64f0181"
WORKER_COUNT = 8
TORCH_THREADS_PER_WORKER = 4
CANARY_SCORE_VECTOR_SHA256 = (
    "a06fba0eea950a61c0599b76169f06a2d77360388c1454fdc7acf9a0a4d2467f"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FeverousNLIRuntimeError(RuntimeError):
    """The FEVEROUS design, local NLI runtime, or worker pool drifted."""


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
        raise FeverousNLIRuntimeError("receipt is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FeverousNLIRuntimeError("bound file cannot be hashed") from exc
    return digest.hexdigest()


def _project_path(project_root: str | Path, relative: Path) -> Path:
    root = Path(project_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise FeverousNLIRuntimeError("project root is unavailable or a symlink")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise FeverousNLIRuntimeError("bound project file is unavailable")
    return path


def verify_feverous_design(project_root: str | Path) -> Mapping[str, object]:
    """Verify the exact design fields that authorize this local worker pool."""

    path = _project_path(project_root, DESIGN_RELATIVE_PATH)
    try:
        raw = path.read_bytes()
        design = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverousNLIRuntimeError("FEVEROUS design is invalid") from exc
    if not isinstance(design, dict):
        raise FeverousNLIRuntimeError("FEVEROUS design must be an object")
    body = dict(design)
    declared = body.pop("design_sha256", None)
    parallel = design.get("parallel_execution_contract")
    offline = design.get("offline_runtime_bindings")
    nli = offline.get("NLI") if isinstance(offline, Mapping) else None
    if (
        declared != DESIGN_SHA256
        or stable_hash(body) != DESIGN_SHA256
        or _sha256_file(path)
        != "de99832010f8d12c7987482c3be575c2d6683410ca50bb6a9ff784ac9efa378a"
        or not isinstance(parallel, Mapping)
        or parallel.get("NLI_worker_processes") != WORKER_COUNT
        or not isinstance(nli, Mapping)
        or nli.get("asset_sha256") != qasc_binding.ASSET_SELF_SHA256
        or nli.get("model") != qasc_binding.MODEL_ID
        or nli.get("revision") != qasc_binding.MODEL_REVISION
    ):
        raise FeverousNLIRuntimeError("FEVEROUS NLI design binding drifted")
    return MappingProxyType(
        {
            "design_file_sha256": _sha256_file(path),
            "design_sha256": DESIGN_SHA256,
            "NLI_worker_processes": WORKER_COUNT,
            "NLI_asset_sha256": qasc_binding.ASSET_SELF_SHA256,
            "status": "verified_feverous_eight_worker_NLI_design",
        }
    )


def _worker_environment(project_root: Path) -> dict[str, str]:
    # Do not inherit credentials, proxy configuration, or caller-specific
    # capability variables into model workers.  The runtime is local-only and
    # uses absolute paths for every asset.
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": str(Path.home()),
        "HF_HUB_OFFLINE": "1",
        "LANG": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(project_root),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }


class _PersistentNLIWorker:
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
            raise FeverousNLIRuntimeError("persistent NLI worker failed to start") from exc
        if (
            self.process.stdin is None
            or self.process.stdout is None
            or self.process.stderr is None
        ):
            self.process.kill()
            raise FeverousNLIRuntimeError("persistent NLI worker pipes are unavailable")
        self._lock = threading.Lock()
        self._stderr_tail: deque[bytes] = deque(maxlen=32)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in iter(self.process.stderr.readline, b""):
            self._stderr_tail.append(line[-4096:])

    def score(self, request: bytes, *, expected_count: int) -> tuple[int, ...]:
        with self._lock:
            if self.process.poll() is not None:
                raise FeverousNLIRuntimeError("persistent NLI worker exited")
            assert self.process.stdin is not None and self.process.stdout is not None
            try:
                self.process.stdin.write(request)
                self.process.stdin.flush()
                raw = self.process.stdout.readline(MAXIMUM_RESPONSE_BYTES + 1)
            except (BrokenPipeError, OSError) as exc:
                raise FeverousNLIRuntimeError("persistent NLI worker pipe failed") from exc
            if not raw or len(raw) > MAXIMUM_RESPONSE_BYTES:
                raise FeverousNLIRuntimeError(
                    "persistent NLI worker response is missing or oversized"
                )
            try:
                return decode_response(raw, expected_count=expected_count)
            except QASCNLIError as exc:
                raise FeverousNLIRuntimeError("persistent NLI response drifted") from exc

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


def _verify_canary_vectors(
    vectors: Sequence[Sequence[int]], *, worker_count: int
) -> Mapping[str, object]:
    if worker_count != WORKER_COUNT or len(vectors) != 2 * WORKER_COUNT:
        raise FeverousNLIRuntimeError("NLI canary did not address every worker twice")
    normalized = tuple(tuple(row) for row in vectors)
    hashes = tuple(stable_hash(list(row)) for row in normalized)
    if (
        not normalized
        or len(set(normalized)) != 1
        or hashes != (CANARY_SCORE_VECTOR_SHA256,) * (2 * WORKER_COUNT)
    ):
        raise FeverousNLIRuntimeError("NLI worker canary is not repeat-exact")
    pairs = canonical_canary_pairs()
    return MappingProxyType(
        {
            "canary_pair_count": len(pairs),
            "design_sha256": DESIGN_SHA256,
            "integer_score_vector_sha256": CANARY_SCORE_VECTOR_SHA256,
            "per_worker_startup_repeat_exact": True,
            "repeat_count_per_worker": 2,
            "status": "passed_feverous_exact_eight_worker_repeat_canary",
            "torch_threads_per_worker": TORCH_THREADS_PER_WORKER,
            "worker_count": WORKER_COUNT,
        }
    )


@dataclass(frozen=True)
class FeverousNLIPoolReceipt:
    design: Mapping[str, object]
    runtime: Mapping[str, object]
    canary: Mapping[str, object]
    receipt_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_receipt",
            "version": VERSION,
            "design": dict(self.design),
            "runtime": dict(self.runtime),
            "canary": dict(self.canary),
            "worker_count": WORKER_COUNT,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "receipt_sha256": self.receipt_sha256,
        }


class FeverousNLIWorkerPool:
    """Concurrency-safe exact-eight pool over the unchanged QASC worker wire."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        project_root: str | Path | None = None,
        runtime_python: str | Path = sys.executable,
    ) -> None:
        root = (
            Path(__file__).parents[2]
            if project_root is None
            else Path(project_root).absolute()
        )
        design = verify_feverous_design(root)
        model = Path(model_path).absolute()
        try:
            runtime = qasc_binding.verify_runtime_asset(root, model)
        except QASCNLIError as exc:
            raise FeverousNLIRuntimeError("frozen NLI runtime verification failed") from exc
        asset_manifest = root / qasc_binding.ASSET_RELATIVE_PATH
        self._state_lock = threading.Lock()
        self._failed = False
        self._closed = False
        self._workers: list[_PersistentNLIWorker] = []
        self._available: queue.Queue[_PersistentNLIWorker] = queue.Queue()
        try:
            for _ in range(WORKER_COUNT):
                worker = _PersistentNLIWorker(
                    runtime_python=runtime_python,
                    asset_manifest_path=asset_manifest,
                    model_root=model,
                    project_root=root,
                )
                self._workers.append(worker)
            request = encode_request(canonical_canary_pairs())
            with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
                first_futures = tuple(
                    executor.submit(
                        worker.score,
                        request,
                        expected_count=len(canonical_canary_pairs()),
                    )
                    for worker in self._workers
                )
                first = tuple(future.result() for future in first_futures)
                second_futures = tuple(
                    executor.submit(
                        worker.score,
                        request,
                        expected_count=len(canonical_canary_pairs()),
                    )
                    for worker in self._workers
                )
                second = tuple(future.result() for future in second_futures)
            canary = _verify_canary_vectors(
                (*first, *second), worker_count=len(self._workers)
            )
            for worker in self._workers:
                self._available.put(worker)
        except BaseException:
            self.close()
            raise
        body = {
            "schema": f"{VERSION}_receipt",
            "version": VERSION,
            "design": dict(design),
            "runtime": dict(runtime),
            "canary": dict(canary),
            "worker_count": WORKER_COUNT,
            "network_calls": 0,
            "online_evaluator_calls": 0,
        }
        self.receipt = FeverousNLIPoolReceipt(
            design=MappingProxyType(dict(design)),
            runtime=MappingProxyType(dict(runtime)),
            canary=MappingProxyType(dict(canary)),
            receipt_sha256=stable_hash(body),
        )

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    def _require_live(self) -> None:
        with self._state_lock:
            if self._closed:
                raise FeverousNLIRuntimeError("NLI worker pool is closed")
            if self._failed:
                raise FeverousNLIRuntimeError("NLI worker pool is poisoned")

    def score_pairs(
        self, pairs: Sequence[Mapping[str, object] | NLIPair]
    ) -> tuple[int, ...]:
        try:
            request = encode_request(pairs)
        except QASCNLIError as exc:
            raise FeverousNLIRuntimeError("NLI pair request drifted") from exc
        self._require_live()
        while True:
            self._require_live()
            try:
                worker = self._available.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        reusable = True
        try:
            return worker.score(request, expected_count=len(pairs))
        except BaseException:
            reusable = False
            with self._state_lock:
                self._failed = True
            raise
        finally:
            if reusable:
                self._available.put(worker)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        for worker in self._workers:
            worker.close()

    def __enter__(self) -> "FeverousNLIWorkerPool":
        self._require_live()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def verify_pool_receipt(receipt: FeverousNLIPoolReceipt) -> str:
    if not isinstance(receipt, FeverousNLIPoolReceipt):
        raise FeverousNLIRuntimeError("NLI pool receipt has the wrong type")
    payload = receipt.payload()
    declared = payload.pop("receipt_sha256", None)
    if (
        not isinstance(declared, str)
        or _SHA256.fullmatch(declared) is None
        or stable_hash(payload) != declared
        or payload.get("schema") != f"{VERSION}_receipt"
        or payload.get("version") != VERSION
        or payload.get("worker_count") != WORKER_COUNT
        or payload.get("network_calls") != 0
        or payload.get("online_evaluator_calls") != 0
        or payload.get("design", {}).get("design_sha256") != DESIGN_SHA256
        or payload.get("runtime", {}).get("asset_sha256")
        != qasc_binding.ASSET_SELF_SHA256
        or payload.get("canary", {}).get("integer_score_vector_sha256")
        != CANARY_SCORE_VECTOR_SHA256
    ):
        raise FeverousNLIRuntimeError("NLI pool receipt drifted")
    return declared


__all__ = [
    "CANARY_SCORE_VECTOR_SHA256",
    "DESIGN_SHA256",
    "FeverousNLIPoolReceipt",
    "FeverousNLIRuntimeError",
    "FeverousNLIWorkerPool",
    "TORCH_THREADS_PER_WORKER",
    "VERSION",
    "WORKER_COUNT",
    "stable_hash",
    "verify_feverous_design",
    "verify_pool_receipt",
]
