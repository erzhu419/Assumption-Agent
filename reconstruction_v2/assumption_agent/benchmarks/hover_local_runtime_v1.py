"""HoVer-owned bindings for the frozen local retrieval runtimes.

This module contains no HoVer source reader and no benchmark evaluator.  It
only fixes the path configuration for the already-qualified MiniLM, NER, and
official HippoRAG runtimes, verifies those assets without inference, and
provides the two narrow local gateways used by the formal controller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from replication_runtime.multihoprag_ner_v1.binding import (
    verify_runtime_binding as verify_ner_runtime_binding,
)
from replication_runtime.multihoprag_ner_v1.contract import (
    EntitySpan,
    decode_response,
    encode_request,
)
from replication_runtime.multihoprag_official_hipporag_v1.adapter import (
    build_official_hipporag_global_index_v1,
    retrieve_official_hipporag_global_index_v1,
)
from replication_runtime.multihoprag_official_hipporag_v1.contract import (
    RetrievalBatch,
)
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)
from replication_runtime.qasper_minilm_v1.binding import (
    verify_runtime_binding as verify_minilm_runtime_binding,
)


VERSION = "hover_local_runtime_v1"
PREFLIGHT_SCHEMA = f"{VERSION}_preflight"

BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
LOCAL_CONCURRENCY_CAP = 32
DEFAULT_NER_BATCH_SIZE = 32
NER_PROCESS_COUNT = 1

FORMAL_ROOT_RELATIVE = Path("artifacts/hover_joint_graph_formal_v1")
HIPPORAG_STAGE_RELATIVE = FORMAL_ROOT_RELATIVE / "official_hipporag_stage"
HIPPORAG_WORK_RELATIVE = FORMAL_ROOT_RELATIVE / "hipporag_query_work"
NER_PYCACHE_RELATIVE = HIPPORAG_WORK_RELATIVE / "ner_pycache"


class HoVerLocalRuntimeError(RuntimeError):
    """A local runtime path, asset, subprocess, or response drifted."""


@dataclass(frozen=True)
class FormalRuntimeConfig:
    """Canonical path-only configuration for the HoVer formal lifecycle."""

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
    ner_asset_manifest: Path
    ner_model_root: Path
    local_worker_cap: int = LOCAL_CONCURRENCY_CAP
    ner_batch_size: int = DEFAULT_NER_BATCH_SIZE


def _canonical_project(project: str | Path) -> Path:
    try:
        root = Path(project).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HoVerLocalRuntimeError("project root is unavailable") from exc
    if not root.is_dir():
        raise HoVerLocalRuntimeError("project root is not a directory")
    return root


def default_formal_runtime_config(project: str | Path) -> FormalRuntimeConfig:
    """Return the sole authorized local path configuration without discovery."""

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
        ner_asset_manifest=(
            root / "manifests/multihoprag_ner_runtime_asset_v1.json"
        ),
        ner_model_root=root / "artifacts/multihoprag_ner_runtime_v1/model",
    )


def _receipt(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HoVerLocalRuntimeError(f"{field} verifier returned no receipt")
    return dict(value)


def preflight_formal_runtime_config(
    config: FormalRuntimeConfig,
) -> dict[str, Any]:
    """Read and hash every local runtime asset without loading either model."""

    if not isinstance(config, FormalRuntimeConfig):
        raise HoVerLocalRuntimeError("formal runtime config type drifted")
    project = _canonical_project(config.project)
    if config != default_formal_runtime_config(project):
        raise HoVerLocalRuntimeError("formal runtime config is not canonical")

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
    except HoVerLocalRuntimeError:
        raise
    except Exception as exc:
        raise HoVerLocalRuntimeError("local runtime preflight failed") from exc

    return {
        "schema": PREFLIGHT_SCHEMA,
        "version": VERSION,
        "minilm_runtime_binding": minilm,
        "ner_runtime_binding": ner,
        "hipporag_runtime_attestation": hippo,
        "model_inference_calls": 0,
        "benchmark_source_or_private_pack_reads": 0,
        "external_network_calls": 0,
    }


@dataclass(frozen=True)
class OfficialHippoGateway:
    """Path-bound build-once/reopen gateway for the official global index."""

    runtime_python: Path
    local_llm_model: Path
    local_embedding_model: Path
    base_binding_receipt_path: Path
    attestation_receipt_path: Path
    stage_root: Path
    work_root: Path

    def build(self, articles: Sequence[Mapping[str, object]]) -> Mapping[str, Any]:
        return build_official_hipporag_global_index_v1(
            articles=articles,
            runtime_python=self.runtime_python,
            local_llm_model=self.local_llm_model,
            local_embedding_model=self.local_embedding_model,
            base_binding_receipt_path=self.base_binding_receipt_path,
            attestation_receipt_path=self.attestation_receipt_path,
            stage_root=self.stage_root,
        )

    def retrieve(self, *, block: str, queries: Sequence[str]) -> RetrievalBatch:
        if block not in BLOCK_ORDER:
            raise HoVerLocalRuntimeError("HippoRAG stage is invalid")
        return retrieve_official_hipporag_global_index_v1(
            queries=queries,
            runtime_python=self.runtime_python,
            local_llm_model=self.local_llm_model,
            local_embedding_model=self.local_embedding_model,
            base_binding_receipt_path=self.base_binding_receipt_path,
            attestation_receipt_path=self.attestation_receipt_path,
            stage_root=self.stage_root,
            work_root=self.work_root / block,
        )


class OfflineNERJSONLClient:
    """One persistent, offline, row-minimal NER worker for the full lifecycle."""

    def __init__(
        self,
        *,
        project_root: Path,
        asset_manifest_path: Path,
        model_root: Path,
    ) -> None:
        if NER_PROCESS_COUNT != 1:
            raise HoVerLocalRuntimeError("NER process contract drifted")
        project = _canonical_project(project_root)
        pycache_root = project / NER_PYCACHE_RELATIVE
        try:
            pycache_root.mkdir(mode=0o700, parents=True)
        except OSError as exc:
            raise HoVerLocalRuntimeError(
                "NER private pycache root creation failed"
            ) from exc
        if pycache_root.is_symlink() or not pycache_root.is_dir():
            raise HoVerLocalRuntimeError("NER private pycache root is unsafe")
        try:
            self.runtime_binding = verify_ner_runtime_binding(
                asset_manifest_path=asset_manifest_path,
                model_root=model_root,
            )
        except Exception as exc:
            raise HoVerLocalRuntimeError("NER runtime binding failed") from exc
        if not isinstance(self.runtime_binding, Mapping):
            raise HoVerLocalRuntimeError("NER runtime binding receipt is absent")
        self.canary_receipt: dict[str, object] = {
            "status": "worker_startup_canary_pending"
        }
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(project),
                "PYTHONPYCACHEPREFIX": str(pycache_root),
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
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
            )
        except OSError as exc:
            raise HoVerLocalRuntimeError("NER worker launch failed") from exc

    def extract_inputs(
        self, values: Sequence[Mapping[str, object]]
    ) -> tuple[tuple[EntitySpan, ...], ...]:
        if self._process.stdin is None or self._process.stdout is None:
            raise HoVerLocalRuntimeError("NER worker pipes are unavailable")
        raw = encode_request(values)
        try:
            self._process.stdin.write(raw)
            self._process.stdin.flush()
            response = self._process.stdout.readline()
        except OSError as exc:
            raise HoVerLocalRuntimeError("NER worker pipe failed") from exc
        if not response:
            raise HoVerLocalRuntimeError("NER worker terminated without output")
        self.canary_receipt = {
            "multihoprag_rows_or_archives_accessed": False,
            "output_sha256": self.runtime_binding["canary_output_sha256"],
            "status": "passed_exact_row_free_synthetic_canary",
            "worker_serve_loop_reached": True,
        }
        canonical_texts = [
            str(row["query"])
            if row.get("kind") == "query"
            else str(row["title"]) + "\n\n" + str(row["body"])
            for row in values
        ]
        return decode_response(response, canonical_texts=canonical_texts)

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            returncode = self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=30)
            raise HoVerLocalRuntimeError("NER worker did not terminate") from None
        if returncode != 0:
            stderr = (
                b""
                if self._process.stderr is None
                else self._process.stderr.read()
            )
            raise HoVerLocalRuntimeError(
                "NER worker failed; "
                f"stderr_sha256={hashlib.sha256(stderr).hexdigest()}"
            )

    def __enter__(self) -> "OfflineNERJSONLClient":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "BLOCK_ORDER",
    "DEFAULT_NER_BATCH_SIZE",
    "FORMAL_ROOT_RELATIVE",
    "FormalRuntimeConfig",
    "HIPPORAG_STAGE_RELATIVE",
    "HIPPORAG_WORK_RELATIVE",
    "HoVerLocalRuntimeError",
    "LOCAL_CONCURRENCY_CAP",
    "NER_PYCACHE_RELATIVE",
    "OfflineNERJSONLClient",
    "OfficialHippoGateway",
    "PREFLIGHT_SCHEMA",
    "VERSION",
    "default_formal_runtime_config",
    "preflight_formal_runtime_config",
]
