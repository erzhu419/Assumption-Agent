"""Sole local-runtime factory for the formal FEVEROUS P6/E2 lifecycle.

The factory binds the already frozen Qasper MiniLM, MultiHopRAG NER, exact-eight
FEVEROUS NLI pool, and FEVEROUS official HippoRAG adapter.  Configuration
preflight is verification-only: it hashes manifests, model trees, runtimes, and
the Hippo transport capability without loading a model or performing inference.
Model startup canaries run only when the context-managed semantic bundle opens.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

from assumption_agent.benchmarks.feverous_nli_runtime_v1 import (
    FeverousNLIPoolReceipt,
    FeverousNLIWorkerPool,
    WORKER_COUNT as NLI_WORKER_COUNT,
    verify_feverous_design,
    verify_pool_receipt,
)
from assumption_agent.benchmarks.feverous_offline_semantic_tensor_v1 import (
    BoundMiniLMBackend,
    BoundNERBackend,
    BoundNLIBackend,
    make_verified_backend_binding,
)
from replication_runtime.feverous_official_hipporag_v1.adapter import (
    _preflight_systemd_transport as verify_hippo_transport,
    build_feverous_official_hipporag_global_index_v1,
    retrieve_feverous_official_hipporag_global_index_v1,
)
from replication_runtime.feverous_official_hipporag_v1.contract import (
    CORPUS_SIZE,
    RetrievalBatch,
    validate_corpus as validate_hippo_corpus,
)
from replication_runtime.multihoprag_ner_v1 import binding as ner_binding
from replication_runtime.multihoprag_ner_v1.binding import (
    verify_runtime_binding as verify_ner_runtime_binding,
)
from replication_runtime.multihoprag_ner_v1.contract import (
    MAXIMUM_RESPONSE_BYTES,
    EntitySpan,
    decode_response,
    encode_request,
    synthetic_canary_inputs,
    validate_inputs as validate_ner_inputs,
)
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)
from replication_runtime.qasc_nli_v1 import binding as nli_binding
from replication_runtime.qasper_minilm_v1.binding import (
    OfflineMiniLMEncoder,
    verify_runtime_binding as verify_minilm_runtime_binding,
)


VERSION = "feverous_local_runtime_v1"
PREFLIGHT_SCHEMA = f"{VERSION}_preflight"
BUNDLE_SCHEMA = f"{VERSION}_semantic_bundle"

BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
LOCAL_ITEM_WORKER_CAP = 64
HIPPORAG_QUERY_BATCH_CAP = 8
DEFAULT_NER_BATCH_SIZE = 32
NER_PROCESS_COUNT = 1

FORMAL_ROOT_RELATIVE = Path("artifacts/feverous_p6_e2_formal_v2")
HIPPORAG_STAGE_RELATIVE = FORMAL_ROOT_RELATIVE / "official_hipporag_stage"
HIPPORAG_WORK_RELATIVE = FORMAL_ROOT_RELATIVE / "hipporag_query_work"
NER_PRIVATE_RELATIVE = FORMAL_ROOT_RELATIVE / "ner_private"
NER_PYCACHE_RELATIVE = NER_PRIVATE_RELATIVE / "pycache"


class FeverousLocalRuntimeError(RuntimeError):
    """A frozen path, runtime, subprocess, canary, or cleanup drifted."""


@dataclass(frozen=True)
class FormalRuntimeConfig:
    """Canonical path-only configuration for one formal FEVEROUS lifecycle."""

    project: Path
    local_runtime_python: Path
    minilm_asset_manifest: Path
    minilm_model_root: Path
    ner_asset_manifest: Path
    ner_model_root: Path
    nli_asset_manifest: Path
    nli_model_root: Path
    hippo_runtime_python: Path
    hippo_llm_model: Path
    hippo_embedding_model: Path
    hippo_base_binding_receipt: Path
    hippo_attestation_receipt: Path
    hippo_stage_root: Path
    hippo_work_root: Path
    ner_pycache_root: Path
    local_item_worker_cap: int = LOCAL_ITEM_WORKER_CAP
    hippo_query_batch_cap: int = HIPPORAG_QUERY_BATCH_CAP
    ner_batch_size: int = DEFAULT_NER_BATCH_SIZE
    ner_process_count: int = NER_PROCESS_COUNT
    nli_worker_count: int = NLI_WORKER_COUNT


def _canonical_project(project: str | Path) -> Path:
    try:
        root = Path(project).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FeverousLocalRuntimeError("project root is unavailable") from exc
    if not root.is_dir():
        raise FeverousLocalRuntimeError("project root is not a directory")
    return root


def _assert_safe_project_directory_chain(
    *, project: Path, path: Path, field: str
) -> None:
    try:
        relative = path.absolute().relative_to(project)
    except ValueError as exc:
        raise FeverousLocalRuntimeError(f"{field} escaped the project root") from exc
    cursor = project
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise FeverousLocalRuntimeError(f"{field} contains a symlink component")
        if cursor.exists() and not cursor.is_dir():
            raise FeverousLocalRuntimeError(f"{field} contains a nondirectory component")


def default_formal_runtime_config(project: str | Path) -> FormalRuntimeConfig:
    """Return the only authorized path configuration; perform no verification."""

    root = _canonical_project(project)
    home = Path.home().absolute()
    try:
        local_python = Path(sys.executable).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FeverousLocalRuntimeError("local runtime Python is unavailable") from exc
    return FormalRuntimeConfig(
        project=root,
        local_runtime_python=local_python,
        minilm_asset_manifest=root / "manifests/qasper_minilm_runtime_asset_v1.json",
        minilm_model_root=root / "artifacts/qasper_minilm_runtime_v1/model",
        ner_asset_manifest=root / "manifests/multihoprag_ner_runtime_asset_v1.json",
        ner_model_root=root / "artifacts/multihoprag_ner_runtime_v1/model",
        nli_asset_manifest=root / "manifests/qasc_nli_runtime_asset_v1.json",
        nli_model_root=root / "artifacts/qasc_nli_runtime_v3/model",
        hippo_runtime_python=home / ".hr5/venv/bin/python",
        hippo_llm_model=home / ".hr5/models/smollm2-135m-instruct",
        hippo_embedding_model=(
            home
            / ".cache/huggingface/hub"
            / "models--sentence-transformers--all-MiniLM-L6-v2"
            / "snapshots"
            / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        ),
        hippo_base_binding_receipt=(
            root / "manifests/musique_official_hipporag_retrieve_only_binding_v1.json"
        ),
        hippo_attestation_receipt=(
            root / "manifests/musique_official_hipporag_runtime_attestation_v3.json"
        ),
        hippo_stage_root=root / HIPPORAG_STAGE_RELATIVE,
        hippo_work_root=root / HIPPORAG_WORK_RELATIVE,
        ner_pycache_root=root / NER_PYCACHE_RELATIVE,
    )


def _receipt(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise FeverousLocalRuntimeError(f"{field} verifier returned no receipt")
    return dict(value)


def preflight_formal_runtime_config(
    config: FormalRuntimeConfig,
) -> dict[str, Any]:
    """Verify every frozen runtime and transport without loading any model."""

    if not isinstance(config, FormalRuntimeConfig):
        raise FeverousLocalRuntimeError("formal runtime config type drifted")
    project = _canonical_project(config.project)
    if config != default_formal_runtime_config(project):
        raise FeverousLocalRuntimeError("formal runtime config is not canonical")
    try:
        minilm = _receipt(
            verify_minilm_runtime_binding(
                asset_manifest_path=config.minilm_asset_manifest,
                model_root=config.minilm_model_root,
            ),
            field="MiniLM runtime",
        )
        ner = _receipt(
            verify_ner_runtime_binding(
                asset_manifest_path=config.ner_asset_manifest,
                model_root=config.ner_model_root,
            ),
            field="NER runtime",
        )
        nli_design = _receipt(
            verify_feverous_design(project),
            field="FEVEROUS NLI design",
        )
        nli = _receipt(
            nli_binding.verify_runtime_binding(
                asset_manifest_path=config.nli_asset_manifest,
                model_root=config.nli_model_root,
            ),
            field="NLI runtime",
        )
        hippo = _receipt(
            verify_formal_runtime_attestation_v3(
                project_root=project,
                attestation_receipt_path=config.hippo_attestation_receipt,
                base_binding_receipt_path=config.hippo_base_binding_receipt,
                runtime_python=config.hippo_runtime_python,
                local_llm_model=config.hippo_llm_model,
                local_embedding_model=config.hippo_embedding_model,
            ),
            field="HippoRAG runtime",
        )
        verify_hippo_transport()
    except FeverousLocalRuntimeError:
        raise
    except Exception as exc:
        raise FeverousLocalRuntimeError("local runtime preflight failed") from exc
    return {
        "schema": PREFLIGHT_SCHEMA,
        "version": VERSION,
        "minilm_runtime_binding": minilm,
        "ner_runtime_binding": ner,
        "nli_design_binding": nli_design,
        "nli_runtime_binding": nli,
        "hipporag_runtime_attestation": hippo,
        "hipporag_transport": {
            "IPAddressDeny": "any",
            "RestrictAddressFamilies": "AF_UNIX",
            "status": "verified_systemd_network_isolation_capability",
        },
        "model_inference_calls": 0,
        "benchmark_source_or_private_pack_reads": 0,
        "external_network_calls": 0,
    }


@dataclass(frozen=True)
class OfficialHippoGateway:
    """Path-bound exact-text build-once/reopen FEVEROUS gateway."""

    runtime_python: Path
    local_llm_model: Path
    local_embedding_model: Path
    base_binding_receipt_path: Path
    attestation_receipt_path: Path
    stage_root: Path
    work_root: Path

    def build(self, units: Sequence[Mapping[str, object]]) -> Mapping[str, Any]:
        validated = validate_hippo_corpus(units)
        canonical_units = tuple(
            {"idx": row.idx, "text": row.text} for row in validated
        )
        if len(canonical_units) != CORPUS_SIZE or any(
            str(raw["text"]).encode("utf-8")
            != validated[index].text.encode("utf-8")
            for index, raw in enumerate(canonical_units)
        ):
            raise FeverousLocalRuntimeError(
                "HippoRAG corpus is not exact FEVEROUS linearized text"
            )
        return build_feverous_official_hipporag_global_index_v1(
            units=canonical_units,
            runtime_python=self.runtime_python,
            local_llm_model=self.local_llm_model,
            local_embedding_model=self.local_embedding_model,
            base_binding_receipt_path=self.base_binding_receipt_path,
            attestation_receipt_path=self.attestation_receipt_path,
            stage_root=self.stage_root,
        )

    def retrieve(self, *, block: str, queries: Sequence[str]) -> RetrievalBatch:
        if block not in BLOCK_ORDER:
            raise FeverousLocalRuntimeError("HippoRAG block is invalid")
        return retrieve_feverous_official_hipporag_global_index_v1(
            queries=queries,
            runtime_python=self.runtime_python,
            local_llm_model=self.local_llm_model,
            local_embedding_model=self.local_embedding_model,
            base_binding_receipt_path=self.base_binding_receipt_path,
            attestation_receipt_path=self.attestation_receipt_path,
            stage_root=self.stage_root,
            work_root=self.work_root / block,
        )


def _canonical_hash(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousLocalRuntimeError("canary payload is not canonical JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def _entity_payload(rows: Sequence[Sequence[EntitySpan]]) -> list[list[dict[str, object]]]:
    return [[span.as_payload() for span in row] for row in rows]


def _ner_worker_environment(
    *, project: Path, runtime_python: Path, pycache_root: Path
) -> dict[str, str]:
    """Return a credential- and proxy-free exact worker environment."""

    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "PATH": f"{runtime_python.parent}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(project),
        "PYTHONPYCACHEPREFIX": str(pycache_root),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }


class OfflineNERJSONLClient:
    """One locked persistent NER worker with a two-pass startup canary."""

    def __init__(
        self,
        *,
        project_root: Path,
        runtime_python: Path,
        asset_manifest_path: Path,
        model_root: Path,
        pycache_root: Path,
    ) -> None:
        if NER_PROCESS_COUNT != 1:
            raise FeverousLocalRuntimeError("NER process contract drifted")
        project = _canonical_project(project_root)
        canonical = default_formal_runtime_config(project)
        if (
            runtime_python != canonical.local_runtime_python
            or asset_manifest_path != canonical.ner_asset_manifest
            or model_root != canonical.ner_model_root
            or pycache_root != canonical.ner_pycache_root
        ):
            raise FeverousLocalRuntimeError("NER runtime paths are not canonical")
        _assert_safe_project_directory_chain(
            project=project,
            path=pycache_root,
            field="NER private pycache root",
        )
        try:
            pycache_root.mkdir(mode=0o700, parents=True)
        except OSError as exc:
            raise FeverousLocalRuntimeError(
                "NER private pycache root creation failed"
            ) from exc
        _assert_safe_project_directory_chain(
            project=project,
            path=pycache_root,
            field="NER private pycache root",
        )
        if not pycache_root.is_dir():
            raise FeverousLocalRuntimeError("NER private pycache root is unsafe")
        try:
            binding = verify_ner_runtime_binding(
                asset_manifest_path=asset_manifest_path,
                model_root=model_root,
            )
        except Exception as exc:
            raise FeverousLocalRuntimeError("NER runtime binding failed") from exc
        self.runtime_binding = _receipt(binding, field="NER runtime")
        if (
            self.runtime_binding.get("canary_output_sha256")
            != ner_binding.CANARY_OUTPUT_SHA256
        ):
            raise FeverousLocalRuntimeError("NER canary trust root drifted")
        self._pipe_lock = threading.Lock()
        self._closed = False
        self._stderr_tail: deque[bytes] = deque(maxlen=32)
        environment = _ner_worker_environment(
            project=project,
            runtime_python=runtime_python,
            pycache_root=pycache_root,
        )
        try:
            self._process = subprocess.Popen(
                [
                    str(runtime_python),
                    "-B",
                    "-m",
                    "replication_runtime.multihoprag_ner_v1.worker",
                    "--asset-manifest",
                    str(asset_manifest_path),
                    "--model-root",
                    str(model_root),
                    "--serve-jsonl",
                ],
                cwd=project,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise FeverousLocalRuntimeError("NER worker launch failed") from exc
        if (
            self._process.stdin is None
            or self._process.stdout is None
            or self._process.stderr is None
        ):
            self._process.kill()
            self._process.wait(timeout=30)
            raise FeverousLocalRuntimeError("NER worker pipes are unavailable")
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
        )
        self._stderr_thread.start()
        try:
            canary_inputs = synthetic_canary_inputs()
            first = self._roundtrip(canary_inputs)
            second = self._roundtrip(canary_inputs)
            expected_hash = str(self.runtime_binding["canary_output_sha256"])
            hashes = (
                _canonical_hash(_entity_payload(first)),
                _canonical_hash(_entity_payload(second)),
            )
            if first != second or hashes != (expected_hash, expected_hash):
                raise FeverousLocalRuntimeError(
                    "NER worker startup canary is not repeat-exact"
                )
            self.canary_receipt: dict[str, object] = {
                "input_count": len(canary_inputs),
                "multihoprag_rows_or_archives_accessed": False,
                "output_sha256": expected_hash,
                "repeat_count": 2,
                "repeat_exact": True,
                "status": "passed_exact_row_free_synthetic_canary",
                "worker_serve_loop_reached": True,
            }
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def _drain_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in iter(self._process.stderr.readline, b""):
            self._stderr_tail.append(line[-4096:])

    def _roundtrip(
        self, values: Sequence[Mapping[str, object]]
    ) -> tuple[tuple[EntitySpan, ...], ...]:
        canonical = validate_ner_inputs(values)
        request = encode_request(values)
        with self._pipe_lock:
            if self._closed or self._process.poll() is not None:
                raise FeverousLocalRuntimeError("NER worker is unavailable")
            assert self._process.stdin is not None and self._process.stdout is not None
            try:
                self._process.stdin.write(request)
                self._process.stdin.flush()
                response = self._process.stdout.readline(MAXIMUM_RESPONSE_BYTES + 1)
            except (BrokenPipeError, OSError) as exc:
                raise FeverousLocalRuntimeError("NER worker pipe failed") from exc
            if not response or len(response) > MAXIMUM_RESPONSE_BYTES:
                raise FeverousLocalRuntimeError(
                    "NER worker response is missing or oversized"
                )
            try:
                return decode_response(
                    response,
                    canonical_texts=tuple(row.text for row in canonical),
                )
            except Exception as exc:
                raise FeverousLocalRuntimeError("NER worker response drifted") from exc

    def extract_inputs(
        self, values: Sequence[Mapping[str, object]]
    ) -> tuple[tuple[EntitySpan, ...], ...]:
        return self._roundtrip(values)

    def close(self) -> None:
        with self._pipe_lock:
            if self._closed:
                return
            self._closed = True
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except OSError:
                    pass
        try:
            returncode = self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=30)
            raise FeverousLocalRuntimeError("NER worker did not terminate") from None
        if returncode != 0:
            stderr = b"".join(self._stderr_tail)
            raise FeverousLocalRuntimeError(
                "NER worker failed; "
                f"stderr_sha256={hashlib.sha256(stderr).hexdigest()}"
            )

    def __enter__(self) -> "OfflineNERJSONLClient":
        if self._closed:
            raise FeverousLocalRuntimeError("NER worker is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _hippo_gateway(config: FormalRuntimeConfig) -> OfficialHippoGateway:
    return OfficialHippoGateway(
        runtime_python=config.hippo_runtime_python,
        local_llm_model=config.hippo_llm_model,
        local_embedding_model=config.hippo_embedding_model,
        base_binding_receipt_path=config.hippo_base_binding_receipt,
        attestation_receipt_path=config.hippo_attestation_receipt,
        stage_root=config.hippo_stage_root,
        work_root=config.hippo_work_root,
    )


class SemanticRuntimeBundle(AbstractContextManager["SemanticRuntimeBundle"]):
    """Context owning all semantic subprocesses and verified Bound backends."""

    def __init__(self, config: FormalRuntimeConfig) -> None:
        self.config = config
        self.preflight_receipt: Mapping[str, object] | None = None
        self.minilm: BoundMiniLMBackend | None = None
        self.ner: BoundNERBackend | None = None
        self.nli: BoundNLIBackend | None = None
        self.hippo: OfficialHippoGateway | None = None
        self._stack: ExitStack | None = None
        self._entered = False
        self._closed = False

    def __enter__(self) -> "SemanticRuntimeBundle":
        if self._entered or self._closed:
            raise FeverousLocalRuntimeError("semantic runtime bundle is not fresh")
        self._entered = True
        stack = ExitStack()
        try:
            self.preflight_receipt = preflight_formal_runtime_config(self.config)
            minilm_runtime = OfflineMiniLMEncoder(
                asset_manifest_path=self.config.minilm_asset_manifest,
                model_root=self.config.minilm_model_root,
            )
            minilm_binding = make_verified_backend_binding(
                role="MiniLM",
                runtime_receipt=minilm_runtime.runtime_receipt,
                canary_receipt=minilm_runtime.canary_receipt,
            )
            ner_runtime = stack.enter_context(
                OfflineNERJSONLClient(
                    project_root=self.config.project,
                    runtime_python=self.config.local_runtime_python,
                    asset_manifest_path=self.config.ner_asset_manifest,
                    model_root=self.config.ner_model_root,
                    pycache_root=self.config.ner_pycache_root,
                )
            )
            ner_backend_binding = make_verified_backend_binding(
                role="NER",
                runtime_receipt=ner_runtime.runtime_binding,
                canary_receipt=ner_runtime.canary_receipt,
            )
            nli_runtime = stack.enter_context(
                FeverousNLIWorkerPool(
                    self.config.nli_model_root,
                    project_root=self.config.project,
                    runtime_python=self.config.local_runtime_python,
                )
            )
            if not isinstance(nli_runtime.receipt, FeverousNLIPoolReceipt):
                raise FeverousLocalRuntimeError("NLI pool receipt type drifted")
            verify_pool_receipt(nli_runtime.receipt)
            nli_backend_binding = make_verified_backend_binding(
                role="NLI",
                runtime_receipt=nli_runtime.receipt.runtime,
                canary_receipt=nli_runtime.receipt.canary,
            )
            self.minilm = BoundMiniLMBackend(
                runtime=minilm_runtime,
                binding=minilm_binding,
            )
            self.ner = BoundNERBackend(
                runtime=ner_runtime,
                binding=ner_backend_binding,
            )
            self.nli = BoundNLIBackend(
                runtime=nli_runtime,
                binding=nli_backend_binding,
            )
            self.hippo = _hippo_gateway(self.config)
            self._stack = stack
            return self
        except BaseException:
            self._closed = True
            stack.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stack is not None:
            self._stack.close()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def receipt(self) -> dict[str, object]:
        if (
            not self._entered
            or self._closed
            or self.minilm is None
            or self.ner is None
            or self.nli is None
        ):
            raise FeverousLocalRuntimeError("semantic runtime bundle is unavailable")
        return {
            "schema": BUNDLE_SCHEMA,
            "version": VERSION,
            "minilm_binding": self.minilm.binding.payload(),
            "ner_binding": self.ner.binding.payload(),
            "nli_binding": self.nli.binding.payload(),
            "ner_process_count": NER_PROCESS_COUNT,
            "nli_worker_count": NLI_WORKER_COUNT,
            "network_calls": 0,
        }


@dataclass(frozen=True)
class FeverousLocalRuntimeFactory:
    """The formal controller's sole constructor for every local runtime."""

    def preflight(self, config: FormalRuntimeConfig) -> Mapping[str, object]:
        return preflight_formal_runtime_config(config)

    def create_hippo(self, config: FormalRuntimeConfig) -> OfficialHippoGateway:
        project = _canonical_project(config.project)
        if config != default_formal_runtime_config(project):
            raise FeverousLocalRuntimeError("formal runtime config is not canonical")
        return _hippo_gateway(config)

    def create_semantic_runtime_bundle(
        self, config: FormalRuntimeConfig
    ) -> SemanticRuntimeBundle:
        project = _canonical_project(config.project)
        if config != default_formal_runtime_config(project):
            raise FeverousLocalRuntimeError("formal runtime config is not canonical")
        return SemanticRuntimeBundle(config)

    def create_semantic_runtime_context(
        self, config: FormalRuntimeConfig
    ) -> SemanticRuntimeBundle:
        """Compatibility spelling for controllers that name the context seam."""

        return self.create_semantic_runtime_bundle(config)


DefaultLocalRuntimeFactory = FeverousLocalRuntimeFactory
DEFAULT_RUNTIME_FACTORY = FeverousLocalRuntimeFactory()


__all__ = [
    "BLOCK_ORDER",
    "BUNDLE_SCHEMA",
    "DEFAULT_NER_BATCH_SIZE",
    "DEFAULT_RUNTIME_FACTORY",
    "DefaultLocalRuntimeFactory",
    "FORMAL_ROOT_RELATIVE",
    "FormalRuntimeConfig",
    "FeverousLocalRuntimeError",
    "FeverousLocalRuntimeFactory",
    "HIPPORAG_QUERY_BATCH_CAP",
    "HIPPORAG_STAGE_RELATIVE",
    "HIPPORAG_WORK_RELATIVE",
    "LOCAL_ITEM_WORKER_CAP",
    "NER_PROCESS_COUNT",
    "NER_PYCACHE_RELATIVE",
    "OfflineNERJSONLClient",
    "OfficialHippoGateway",
    "PREFLIGHT_SCHEMA",
    "SemanticRuntimeBundle",
    "VERSION",
    "default_formal_runtime_config",
    "preflight_formal_runtime_config",
]
