"""Path-bound offline MiniLM and official HippoRAG runtime for HybridQA v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import hybridqa_direct_acquisition_v2 as acquisition
from replication_runtime.hybridqa_official_hipporag_v1.adapter import (
    build_official_hipporag_global_index_v1,
    retrieve_official_hipporag_global_index_v1,
)
from replication_runtime.hybridqa_official_hipporag_v1.contract import RetrievalBatch
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)
from replication_runtime.qasper_minilm_v1.binding import (
    OfflineMiniLMEncoder,
    verify_runtime_binding as verify_minilm_runtime_binding,
)


VERSION = "hybridqa_local_runtime_v2"
HIPPORAG_STAGE_RELATIVE = (
    acquisition.FORMAL_ROOT_RELATIVE / "official_hipporag_stage"
)
HIPPORAG_WORK_RELATIVE = (
    acquisition.FORMAL_ROOT_RELATIVE / "official_hipporag_work"
)


class HybridQaLocalRuntimeError(RuntimeError):
    """A local path, frozen asset, attestation or gateway contract drifted."""


@dataclass(frozen=True)
class FormalRuntimeConfig:
    project: Path
    hippo_runtime_python: Path
    hippo_llm_model: Path
    hippo_embedding_model: Path
    hippo_base_binding_receipt: Path
    hippo_attestation_receipt: Path
    hippo_stage_root: Path
    hippo_work_root: Path
    minilm_asset_manifest: Path
    minilm_model_root: Path


def _canonical_project(project: str | Path) -> Path:
    try:
        root = Path(project).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HybridQaLocalRuntimeError("project root is unavailable") from exc
    if root.is_symlink() or not root.is_dir():
        raise HybridQaLocalRuntimeError("project root is unsafe")
    return root


def default_formal_runtime_config(project: str | Path) -> FormalRuntimeConfig:
    root = _canonical_project(project)
    home = Path.home()
    return FormalRuntimeConfig(
        project=root,
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
        minilm_asset_manifest=root / "manifests/qasper_minilm_runtime_asset_v1.json",
        minilm_model_root=root / "artifacts/qasper_minilm_runtime_v1/model",
    )


def preflight_formal_runtime_config(config: FormalRuntimeConfig) -> dict[str, Any]:
    """Hash/attest both local runtimes without loading a model or source pack."""

    if not isinstance(config, FormalRuntimeConfig):
        raise HybridQaLocalRuntimeError("runtime config type drifted")
    project = _canonical_project(config.project)
    if config != default_formal_runtime_config(project):
        raise HybridQaLocalRuntimeError("runtime config is not canonical")
    for path in (config.hippo_stage_root, config.hippo_work_root):
        try:
            path.absolute().relative_to(project)
        except ValueError as exc:
            raise HybridQaLocalRuntimeError("runtime output escaped project") from exc
        if path.exists() or path.is_symlink():
            raise HybridQaLocalRuntimeError("formal runtime output already exists")
    try:
        minilm = verify_minilm_runtime_binding(
            asset_manifest_path=config.minilm_asset_manifest,
            model_root=config.minilm_model_root,
        )
        hippo = verify_formal_runtime_attestation_v3(
            project_root=project,
            attestation_receipt_path=config.hippo_attestation_receipt,
            base_binding_receipt_path=config.hippo_base_binding_receipt,
            runtime_python=config.hippo_runtime_python,
            local_llm_model=config.hippo_llm_model,
            local_embedding_model=config.hippo_embedding_model,
        )
    except Exception as exc:
        raise HybridQaLocalRuntimeError("offline runtime preflight failed") from exc
    if not isinstance(minilm, Mapping) or not isinstance(hippo, Mapping):
        raise HybridQaLocalRuntimeError("offline runtime preflight receipt drifted")
    return {
        "schema": f"{VERSION}_preflight",
        "version": VERSION,
        "minilm_runtime_binding": dict(minilm),
        "official_hipporag_runtime_attestation": dict(hippo),
        "model_inference_calls": 0,
        "benchmark_source_or_private_pack_reads": 0,
        "external_network_calls": 0,
    }


@dataclass(frozen=True)
class OfficialHippoGateway:
    config: FormalRuntimeConfig

    def build(self, articles: Sequence[Mapping[str, object]]) -> Mapping[str, Any]:
        return build_official_hipporag_global_index_v1(
            articles=articles,
            runtime_python=self.config.hippo_runtime_python,
            local_llm_model=self.config.hippo_llm_model,
            local_embedding_model=self.config.hippo_embedding_model,
            base_binding_receipt_path=self.config.hippo_base_binding_receipt,
            attestation_receipt_path=self.config.hippo_attestation_receipt,
            stage_root=self.config.hippo_stage_root,
        )

    def retrieve(self, *, block: str, queries: Sequence[str]) -> RetrievalBatch:
        if block not in {"A_form_F_search_A_hold", "M_search"}:
            raise HybridQaLocalRuntimeError("HippoRAG retrieval stage is invalid")
        try:
            self.config.hippo_work_root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise HybridQaLocalRuntimeError(
                "HippoRAG work root cannot be created"
            ) from exc
        try:
            metadata = self.config.hippo_work_root.lstat()
        except OSError as exc:
            raise HybridQaLocalRuntimeError(
                "HippoRAG work root cannot be inspected"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise HybridQaLocalRuntimeError("HippoRAG work root is unsafe")
        return retrieve_official_hipporag_global_index_v1(
            queries=queries,
            runtime_python=self.config.hippo_runtime_python,
            local_llm_model=self.config.hippo_llm_model,
            local_embedding_model=self.config.hippo_embedding_model,
            base_binding_receipt_path=self.config.hippo_base_binding_receipt,
            attestation_receipt_path=self.config.hippo_attestation_receipt,
            stage_root=self.config.hippo_stage_root,
            work_root=self.config.hippo_work_root / block,
        )


@dataclass(frozen=True)
class RuntimeBundle:
    encoder: OfflineMiniLMEncoder
    hippo: OfficialHippoGateway


def open_runtime(config: FormalRuntimeConfig) -> RuntimeBundle:
    if config != default_formal_runtime_config(config.project):
        raise HybridQaLocalRuntimeError("runtime config drifted before model load")
    try:
        encoder = OfflineMiniLMEncoder(
            asset_manifest_path=config.minilm_asset_manifest,
            model_root=config.minilm_model_root,
        )
    except Exception as exc:
        raise HybridQaLocalRuntimeError("offline MiniLM load/canary failed") from exc
    return RuntimeBundle(encoder=encoder, hippo=OfficialHippoGateway(config))


__all__ = [
    "FormalRuntimeConfig",
    "HIPPORAG_STAGE_RELATIVE",
    "HIPPORAG_WORK_RELATIVE",
    "HybridQaLocalRuntimeError",
    "OfficialHippoGateway",
    "RuntimeBundle",
    "VERSION",
    "default_formal_runtime_config",
    "open_runtime",
    "preflight_formal_runtime_config",
]
