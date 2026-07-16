"""Immutable asset verification and independent-process NLI clients."""

from __future__ import annotations

from collections import deque
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

from .contract import (
    MAXIMUM_RESPONSE_BYTES,
    NLIPair,
    QASCNLIError,
    decode_response,
    encode_request,
)


ASSET_VERSION = "qasc_nli_runtime_asset_v1"
ASSET_SELF_SHA256 = "d64f4403e7603ea71e622e7e7124eae466cbf67bf4c758979b54c4ccf9bb5fe8"
ASSET_FILE_SHA256 = "7abe0922a800739cdea06a269310681d50cb00d73e6d9b995d8a147e45d7c961"
ASSET_GIT_COMMIT = "acce2ebd46c46abfe197aa0241b5748e0ccec2e2"
ASSET_RELATIVE_PATH = Path("manifests/qasc_nli_runtime_asset_v1.json")
DESIGN_SELF_SHA256 = "7c52b7e43d02ffa986683c49ca61863c3f36985b97a1a4677a40b6cddef8c150"
DESIGN_FILE_SHA256 = "fdd1bd1d088cee851a20015227d1f3dea1d086bcaf5c0f435f1bf52e943ab003"
DESIGN_GIT_COMMIT = "ac95a656b7bd1c4c0078f3d8f54a8f5579209aff"
DESIGN_RELATIVE_PATH = Path(
    "manifests/qasc_evaluator_direct_action_coevolution_design_v1.json"
)
FORMAL_WORKER_COUNT = 8
FORMAL_TORCH_THREADS_PER_WORKER = 4
MODEL_ID = "cross-encoder/nli-distilroberta-base"
MODEL_REVISION = "fe43becf0e9bb49299eabd1fe5cc2a74ecf1fcd6"
MODEL_TREE_SHA256 = "a3509209777b85c7bffed215d5e0ad0a41e7211e396fc6e9d1a2bf2e56869f40"
WEIGHTS_SHA256 = "9df3eb5d37118f952f4ba4fb46fde6889e3a9ccedeee0bad09b0110fc64c5c29"
MODEL_ARCHITECTURE = "RobertaForSequenceClassification"
EXPECTED_LABELS = {"0": "contradiction", "1": "entailment", "2": "neutral"}
EXPECTED_EXECUTION = {
    "batch_size": 64,
    "device": "cpu",
    "dtype": "float32",
    "environment": {
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    },
    "eval_mode": True,
    "local_files_only": True,
    "maximum_sequence_length": 256,
    "network_calls": 0,
    "padding": True,
    "torch_deterministic_algorithms": True,
    "torch_inference_mode": True,
    "torch_interop_threads": 1,
    "torch_manual_seed": 0,
    "torch_num_threads": 4,
    "truncation": True,
    "use_safetensors": True,
}
EXPECTED_RUNTIME_VERSIONS = {
    "huggingface_hub": "1.11.0",
    "safetensors": "0.7.0",
    "tokenizers": "0.22.2",
    "torch": "2.8.0+cu128",
    "transformers": "5.10.1",
}
EXPECTED_SCORE_CONTRACT = {
    "contradiction_logit_index": 0,
    "entailment_logit_index": 1,
    "integer_formula": "int(round((entailment_logit - max(contradiction_logit, neutral_logit)) * 1000000))",
    "neutral_logit_index": 2,
    "quantization_scale": 1000000,
    "rounding": "Python_round_ties_to_even",
    "score_direction": "larger_is_more_entailing",
    "unquantized_formula": "entailment_logit - max(contradiction_logit, neutral_logit)",
}
_PACKAGE_TO_MODULE = {
    "huggingface_hub": ("huggingface-hub", "huggingface_hub"),
    "safetensors": ("safetensors", "safetensors"),
    "tokenizers": ("tokenizers", "tokenizers"),
    "torch": ("torch", "torch"),
    "transformers": ("transformers", "transformers"),
}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _reject_symlink_components(path: Path, field: str) -> Path:
    absolute = path.expanduser().absolute()
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise QASCNLIError(f"{field} contains a symlink component")
    return absolute


def _load_asset_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = _reject_symlink_components(Path(path), "asset manifest path")
    if not manifest_path.is_file() or manifest_path.stat().st_size > 128 * 1024:
        raise QASCNLIError("asset manifest is unavailable or oversized")
    raw = manifest_path.read_bytes()
    if _sha256_bytes(raw) != ASSET_FILE_SHA256:
        raise QASCNLIError("committed asset manifest file drifted")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QASCNLIError("asset manifest is invalid") from exc
    if not isinstance(value, dict) or value.get("asset_version") != ASSET_VERSION:
        raise QASCNLIError("asset manifest version mismatch")
    declared = value.get("asset_sha256")
    body = dict(value)
    body.pop("asset_sha256", None)
    if declared != ASSET_SELF_SHA256 or _canonical_hash(body) != declared:
        raise QASCNLIError("asset manifest self-hash mismatch")
    return manifest_path, value


def _verify_manifest_contract(asset: Mapping[str, Any]) -> None:
    model = asset.get("model")
    label = asset.get("label_contract")
    local = asset.get("local_binding")
    if not isinstance(model, Mapping) or not isinstance(label, Mapping) or not isinstance(
        local, Mapping
    ):
        raise QASCNLIError("asset manifest binding is incomplete")
    if (
        model.get("model_id") != MODEL_ID
        or model.get("snapshot_revision") != MODEL_REVISION
        or model.get("architecture") != MODEL_ARCHITECTURE
        or model.get("weight_serialization") != "safetensors"
        or model.get("weights_sha256") != WEIGHTS_SHA256
        or label.get("config_mapping_verified") is not True
        or label.get("id2label") != EXPECTED_LABELS
        or asset.get("license") != "Apache-2.0"
        or asset.get("execution") != EXPECTED_EXECUTION
        or asset.get("runtime_versions") != EXPECTED_RUNTIME_VERSIONS
        or asset.get("score_contract") != EXPECTED_SCORE_CONTRACT
        or local.get("snapshot_tree_sha256") != MODEL_TREE_SHA256
    ):
        raise QASCNLIError("asset manifest normative contract drifted")


def _verify_model_tree(asset: Mapping[str, Any], model_root: str | Path) -> Path:
    root = _reject_symlink_components(Path(model_root), "model root")
    if not root.is_dir():
        raise QASCNLIError("model root is unavailable")
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in (*directories, *files):
            if (base / name).is_symlink():
                raise QASCNLIError("model tree contains a symlink")
    local = asset["local_binding"]
    rows = local.get("snapshot_files")
    if not isinstance(rows, list) or len(rows) != local.get("snapshot_file_count"):
        raise QASCNLIError("snapshot file manifest is malformed")
    expected_paths: list[str] = []
    verified_rows: list[dict[str, object]] = []
    total_size = 0
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size"}:
            raise QASCNLIError("snapshot file row is malformed")
        relative_text = row.get("path")
        relative = Path(str(relative_text))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise QASCNLIError("snapshot file path is unsafe")
        path = root / relative
        size = row.get("size")
        digest = row.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or not path.is_file()
            or path.stat().st_size != size
            or _sha256_file(path) != digest
        ):
            raise QASCNLIError("snapshot file content drifted")
        expected_paths.append(str(relative_text))
        verified_rows.append({"path": str(relative_text), "sha256": digest, "size": size})
        total_size += size
    live_top_level_files = sorted(entry.name for entry in root.iterdir() if entry.is_file())
    if live_top_level_files != sorted(expected_paths):
        raise QASCNLIError("snapshot top-level file set drifted")
    live_top_level_directories = sorted(
        entry.name for entry in root.iterdir() if entry.is_dir()
    )
    if live_top_level_directories not in ([], [".cache"]):
        raise QASCNLIError("snapshot contains an unbound top-level directory")
    if total_size != local.get("snapshot_size_bytes"):
        raise QASCNLIError("snapshot total size drifted")
    if _canonical_hash(verified_rows) != local.get("snapshot_tree_sha256"):
        raise QASCNLIError("snapshot tree hash drifted")
    required_paths = local.get("runtime_required_paths")
    if not isinstance(required_paths, list) or len(required_paths) != local.get(
        "runtime_required_file_count"
    ):
        raise QASCNLIError("runtime-required file set is malformed")
    required_rows = [row for row in verified_rows if row["path"] in required_paths]
    if (
        len(required_rows) != len(required_paths)
        or _canonical_hash(required_rows) != local.get("runtime_required_file_set_sha256")
        or sum(int(row["size"]) for row in required_rows)
        != local.get("runtime_required_size_bytes")
    ):
        raise QASCNLIError("runtime-required file binding drifted")
    return root


def _verify_package_versions(asset: Mapping[str, Any]) -> dict[str, str]:
    declared = asset.get("runtime_versions")
    if declared != EXPECTED_RUNTIME_VERSIONS:
        raise QASCNLIError("declared runtime package versions drifted")
    actual: dict[str, str] = {}
    for key, (distribution, module_name) in _PACKAGE_TO_MODULE.items():
        try:
            distribution_version = metadata.version(distribution)
            module_version = getattr(import_module(module_name), "__version__")
        except (ImportError, AttributeError, metadata.PackageNotFoundError) as exc:
            raise QASCNLIError(f"required runtime package is missing: {distribution}") from exc
        # Torch's wheel metadata intentionally omits its local CUDA build tag;
        # the public runtime asset binds the imported runtime identity instead.
        actual[key] = str(module_version)
        if key != "torch" and distribution_version != actual[key]:
            raise QASCNLIError("runtime module and distribution versions disagree")
    if actual != declared:
        raise QASCNLIError("installed runtime package versions drifted")
    return actual


def verify_runtime_binding(
    *, asset_manifest_path: str | Path, model_root: str | Path
) -> dict[str, object]:
    """Recompute every immutable public and local runtime binding."""

    manifest_path, asset = _load_asset_manifest(asset_manifest_path)
    _verify_manifest_contract(asset)
    verified_root = _verify_model_tree(asset, model_root)
    versions = _verify_package_versions(asset)
    return {
        "asset_file_sha256": ASSET_FILE_SHA256,
        "asset_git_commit": ASSET_GIT_COMMIT,
        "asset_manifest_path": str(manifest_path),
        "asset_sha256": ASSET_SELF_SHA256,
        "model_root": str(verified_root),
        "model_tree_sha256": MODEL_TREE_SHA256,
        "runtime_versions": versions,
        "status": "verified_offline_immutable_runtime",
    }


def verify_runtime_asset(
    project_root: str | Path,
    model_path: str | Path,
) -> dict[str, object]:
    """Convenience entry point bound to the committed project manifest."""

    root = _reject_symlink_components(Path(project_root), "project root")
    if not root.is_dir():
        raise QASCNLIError("project root is unavailable")
    return verify_runtime_binding(
        asset_manifest_path=root / ASSET_RELATIVE_PATH,
        model_root=model_path,
    )


def verify_design_binding(project_root: str | Path) -> dict[str, object]:
    """Bind the exact 8-worker formal profile fixed after the asset freeze."""

    root = _reject_symlink_components(Path(project_root), "project root")
    path = _reject_symlink_components(root / DESIGN_RELATIVE_PATH, "design path")
    if not path.is_file() or _sha256_file(path) != DESIGN_FILE_SHA256:
        raise QASCNLIError("committed QASC design file drifted")
    try:
        design = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QASCNLIError("committed QASC design is invalid") from exc
    if not isinstance(design, dict):
        raise QASCNLIError("committed QASC design must be an object")
    body = dict(design)
    declared = body.pop("design_sha256", None)
    runtime = design.get("runtime_and_preflight")
    nli = design.get("nli_binding")
    if (
        declared != DESIGN_SELF_SHA256
        or _canonical_hash(body) != declared
        or design.get("schema")
        != "qasc_evaluator_direct_action_coevolution_design_v1"
        or not isinstance(runtime, Mapping)
        or runtime.get("formal_default_NLI_workers") != FORMAL_WORKER_COUNT
        or runtime.get("formal_default_torch_threads_per_worker")
        != FORMAL_TORCH_THREADS_PER_WORKER
        or runtime.get("formal_worker_profile_adjustment")
        != "none_if_the_pre_marker_exact_shape_8_worker_repeat_equality_and_capacity_diagnostic_fails_the_study_is_invalidated_before_marker"
        or not isinstance(nli, Mapping)
        or nli.get("asset_sha256") != ASSET_SELF_SHA256
        or nli.get("asset_file_sha256") != ASSET_FILE_SHA256
        or nli.get("maximum_sequence_length") != 256
        or "batch64_torch_threads4_interop1" not in str(nli.get("runtime"))
    ):
        raise QASCNLIError("QASC design NLI worker profile drifted")
    return {
        "design_file_sha256": DESIGN_FILE_SHA256,
        "design_git_commit": DESIGN_GIT_COMMIT,
        "design_sha256": DESIGN_SELF_SHA256,
        "formal_NLI_workers": FORMAL_WORKER_COUNT,
        "torch_threads_per_worker": FORMAL_TORCH_THREADS_PER_WORKER,
        "status": "verified_formal_NLI_worker_profile",
    }


def _project_root_from_manifest(path: str | Path) -> Path:
    manifest = _reject_symlink_components(Path(path), "asset manifest path")
    if manifest.parent.name != "manifests":
        raise QASCNLIError("asset manifest must be in the project manifests directory")
    root = manifest.parent.parent
    if not root.is_dir():
        raise QASCNLIError("project root is unavailable")
    return root


def _worker_environment(project_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(project_root),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def _worker_command(
    *,
    runtime_python: str | Path,
    asset_manifest_path: str | Path,
    model_root: str | Path,
    serve_jsonl: bool,
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> list[str]:
    command = [
        str(runtime_python),
        "-m",
        "replication_runtime.qasc_nli_v1.worker",
        "--asset-manifest",
        str(Path(asset_manifest_path).absolute()),
        "--model-root",
        str(Path(model_root).absolute()),
    ]
    if serve_jsonl:
        command.append("--serve-jsonl")
    else:
        if input_path is None or output_path is None:
            raise AssertionError("one-shot worker paths are required")
        command.extend(["--input", str(input_path), "--output", str(output_path)])
    return command


def score_pairs_in_subprocess(
    pairs: Sequence[Mapping[str, object] | NLIPair],
    *,
    asset_manifest_path: str | Path,
    model_root: str | Path,
    runtime_python: str | Path = sys.executable,
    timeout_seconds: int = 600,
    temporary_root: str | Path | None = None,
) -> tuple[int, ...]:
    """Score one exact batch in a new process; never retry a failed launch."""

    request = encode_request(pairs)
    verify_runtime_binding(
        asset_manifest_path=asset_manifest_path,
        model_root=model_root,
    )
    project_root = _project_root_from_manifest(asset_manifest_path)
    temp_parent = None if temporary_root is None else str(Path(temporary_root).absolute())
    with tempfile.TemporaryDirectory(prefix="qasc-nli-v1-", dir=temp_parent) as folder:
        directory = Path(folder)
        input_path = directory / "request.json"
        output_path = directory / "response.json"
        input_path.write_bytes(request)
        command = _worker_command(
            runtime_python=runtime_python,
            asset_manifest_path=asset_manifest_path,
            model_root=model_root,
            serve_jsonl=False,
            input_path=input_path,
            output_path=output_path,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=_worker_environment(project_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise QASCNLIError("independent NLI worker failed to complete") from exc
        if completed.returncode != 0 or completed.stdout:
            raise QASCNLIError("independent NLI worker failed closed")
        if output_path.is_symlink() or not output_path.is_file():
            raise QASCNLIError("independent NLI worker omitted its response")
        raw = output_path.read_bytes()
    return decode_response(raw, expected_count=len(pairs))


class _PersistentWorker:
    def __init__(
        self,
        *,
        runtime_python: str | Path,
        asset_manifest_path: str | Path,
        model_root: str | Path,
        project_root: Path,
    ) -> None:
        command = _worker_command(
            runtime_python=runtime_python,
            asset_manifest_path=asset_manifest_path,
            model_root=model_root,
            serve_jsonl=True,
        )
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
            raise QASCNLIError("persistent NLI worker could not start") from exc
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            self.process.kill()
            raise QASCNLIError("persistent NLI worker pipes are unavailable")
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
                raise QASCNLIError("persistent NLI worker exited")
            assert self.process.stdin is not None and self.process.stdout is not None
            try:
                self.process.stdin.write(request)
                self.process.stdin.flush()
                raw = self.process.stdout.readline(MAXIMUM_RESPONSE_BYTES + 1)
            except (BrokenPipeError, OSError) as exc:
                raise QASCNLIError("persistent NLI worker pipe failed") from exc
            if not raw or len(raw) > MAXIMUM_RESPONSE_BYTES:
                raise QASCNLIError("persistent NLI worker response is missing or oversized")
            return decode_response(raw, expected_count=expected_count)

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


class NLIWorkerPool:
    """Concurrency-safe, fail-closed pool of independent local NLI workers.

    The study design at commit ``ac95a656`` binds exactly eight workers.  Each
    worker independently enforces the asset's frozen four Torch threads.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        workers: int = FORMAL_WORKER_COUNT,
        project_root: str | Path | None = None,
        runtime_python: str | Path = sys.executable,
    ) -> None:
        if (
            isinstance(workers, bool)
            or not isinstance(workers, int)
            or workers != FORMAL_WORKER_COUNT
        ):
            raise QASCNLIError("formal QASC design requires exactly 8 NLI workers")
        root = (
            Path(__file__).parents[2]
            if project_root is None
            else _reject_symlink_components(Path(project_root), "project root")
        )
        verify_design_binding(root)
        asset_manifest_path = root / ASSET_RELATIVE_PATH
        verify_runtime_asset(root, model_path)
        project_root = _project_root_from_manifest(asset_manifest_path)
        self._state_lock = threading.Lock()
        self._failed = False
        self._closed = False
        self._workers: list[_PersistentWorker] = []
        self._available: queue.Queue[_PersistentWorker] = queue.Queue()
        try:
            for _ in range(workers):
                worker = _PersistentWorker(
                    runtime_python=runtime_python,
                    asset_manifest_path=asset_manifest_path,
                    model_root=model_path,
                    project_root=project_root,
                )
                self._workers.append(worker)
                self._available.put(worker)
        except BaseException:
            self.close()
            raise

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    def _require_live(self) -> None:
        with self._state_lock:
            if self._closed:
                raise QASCNLIError("NLI worker pool is closed")
            if self._failed:
                raise QASCNLIError("NLI worker pool is poisoned by a prior failure")

    def _poison(self) -> None:
        with self._state_lock:
            self._failed = True

    def score_pairs(
        self, pairs: Sequence[Mapping[str, object] | NLIPair]
    ) -> tuple[int, ...]:
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
            result = worker.score(request, expected_count=len(pairs))
        except BaseException:
            return_worker = False
            self._poison()
            raise
        finally:
            if return_worker:
                self._available.put(worker)
        return result

    def score_batches(
        self,
        batches: Sequence[Sequence[Mapping[str, object] | NLIPair]],
    ) -> tuple[tuple[int, ...], ...]:
        if not batches:
            return ()
        self._require_live()
        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            futures = [executor.submit(self.score_pairs, batch) for batch in batches]
            return tuple(future.result() for future in futures)

    def score_items(
        self,
        items: Sequence[
            tuple[
                str,
                Sequence[
                    Mapping[str, object]
                    | NLIPair
                    | tuple[str, str]
                ],
            ]
        ],
    ) -> dict[str, tuple[int, ...]]:
        """Score item batches in parallel while keeping item keys local.

        Tuple pairs are interpreted as ``(premise, hypothesis)``.  Item keys
        are used only to reconstruct the caller's exact input order and never
        cross a worker process boundary.
        """

        keys: list[str] = []
        batches: list[list[Mapping[str, object] | NLIPair]] = []
        seen: set[str] = set()
        for row in items:
            if not isinstance(row, tuple) or len(row) != 2:
                raise QASCNLIError("each score_items row must be (item_key, pairs)")
            item_key, raw_pairs = row
            if not isinstance(item_key, str) or not item_key or item_key in seen:
                raise QASCNLIError("item keys must be non-empty and unique")
            if isinstance(raw_pairs, (str, bytes)) or not isinstance(raw_pairs, Sequence):
                raise QASCNLIError("item NLI pairs must be a sequence")
            normalized: list[Mapping[str, object] | NLIPair] = []
            for pair in raw_pairs:
                if isinstance(pair, tuple):
                    if len(pair) != 2:
                        raise QASCNLIError("tuple NLI pairs must have length two")
                    normalized.append({"premise": pair[0], "hypothesis": pair[1]})
                else:
                    normalized.append(pair)
            # Validate before any work is submitted and canonicalize mappings.
            request = encode_request(normalized)
            from .contract import decode_request

            validated = decode_request(request)
            keys.append(item_key)
            batches.append(list(validated))
            seen.add(item_key)
        results = self.score_batches(batches)
        return {key: result for key, result in zip(keys, results)}

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        for worker in self._workers:
            worker.close()

    def __enter__(self) -> "NLIWorkerPool":
        self._require_live()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def run_canary(
    project_root: str | Path,
    model_path: str | Path,
    *,
    workers: int = FORMAL_WORKER_COUNT,
    runtime_python: str | Path = sys.executable,
) -> dict[str, object]:
    """Run the design-bound pre-marker canary on all eight child workers."""

    if workers != FORMAL_WORKER_COUNT or isinstance(workers, bool):
        raise QASCNLIError("formal QASC canary requires exactly 8 NLI workers")
    verify_design_binding(project_root)
    from .worker import canonical_canary_pairs

    pairs = canonical_canary_pairs()
    request = encode_request(pairs)
    expected_hash = "a06fba0eea950a61c0599b76169f06a2d77360388c1454fdc7acf9a0a4d2467f"
    with NLIWorkerPool(
        model_path,
        workers=workers,
        project_root=project_root,
        runtime_python=runtime_python,
    ) as pool:
        # Address each child directly so successful completion proves that all
        # eight independently loaded, verified, and repeated the startup canary.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(worker.score, request, expected_count=len(pairs))
                for worker in pool._workers
            ]
            vectors = tuple(future.result() for future in futures)
    hashes = tuple(_canonical_hash(list(vector)) for vector in vectors)
    if len(set(vectors)) != 1 or hashes != (expected_hash,) * workers:
        raise QASCNLIError("8-worker NLI capacity canary was not exactly equal")
    return {
        "canary_pair_count": len(pairs),
        "design_sha256": DESIGN_SELF_SHA256,
        "integer_score_vector_sha256": expected_hash,
        "model_tree_sha256": MODEL_TREE_SHA256,
        "per_worker_startup_repeat_exact": True,
        "status": "passed_exact_shape_8_worker_repeat_equality_and_capacity",
        "torch_threads_per_worker": FORMAL_TORCH_THREADS_PER_WORKER,
        "worker_count": workers,
    }
