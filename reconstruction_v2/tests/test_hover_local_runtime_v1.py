from __future__ import annotations

from dataclasses import replace
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from assumption_agent.benchmarks import hover_local_runtime_v1 as runtime
from replication_runtime.multihoprag_ner_v1.contract import (
    encode_request,
    encode_response,
)
from replication_runtime.multihoprag_official_hipporag_v1.contract import (
    RetrievalBatch,
)


def test_default_config_and_preflight_are_canonical_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def minilm(**kwargs: object) -> Mapping[str, object]:
        calls.append(("minilm", dict(kwargs)))
        return {"status": "synthetic_minilm_verified"}

    def ner(**kwargs: object) -> Mapping[str, object]:
        calls.append(("ner", dict(kwargs)))
        return {"status": "synthetic_ner_verified"}

    def hippo(**kwargs: object) -> Mapping[str, object]:
        calls.append(("hippo", dict(kwargs)))
        return {"status": "synthetic_hippo_verified"}

    monkeypatch.setattr(runtime, "verify_minilm_runtime_binding", minilm)
    monkeypatch.setattr(runtime, "verify_ner_runtime_binding", ner)
    monkeypatch.setattr(runtime, "verify_formal_runtime_attestation_v3", hippo)

    config = runtime.default_formal_runtime_config(tmp_path)
    assert config.project == tmp_path.resolve()
    assert config.local_worker_cap == 32
    assert config.ner_batch_size == 32
    assert config.hippo_stage_root == (
        tmp_path.resolve()
        / "artifacts/hover_joint_graph_formal_v1/official_hipporag_stage"
    )
    assert config.hippo_work_root == (
        tmp_path.resolve()
        / "artifacts/hover_joint_graph_formal_v1/hipporag_query_work"
    )

    receipt = runtime.preflight_formal_runtime_config(config)
    assert receipt == {
        "schema": runtime.PREFLIGHT_SCHEMA,
        "version": runtime.VERSION,
        "minilm_runtime_binding": {"status": "synthetic_minilm_verified"},
        "ner_runtime_binding": {"status": "synthetic_ner_verified"},
        "hipporag_runtime_attestation": {
            "status": "synthetic_hippo_verified"
        },
        "model_inference_calls": 0,
        "benchmark_source_or_private_pack_reads": 0,
        "external_network_calls": 0,
    }
    assert [name for name, _arguments in calls] == ["minilm", "ner", "hippo"]
    assert calls[0][1] == {
        "asset_manifest_path": config.minilm_asset_manifest,
        "model_root": config.minilm_model_root,
    }
    assert calls[1][1] == {
        "asset_manifest_path": config.ner_asset_manifest,
        "model_root": config.ner_model_root,
    }
    assert calls[2][1] == {
        "project_root": config.project,
        "attestation_receipt_path": config.hippo_attestation_receipt,
        "base_binding_receipt_path": config.hippo_base_binding_receipt,
        "runtime_python": config.hippo_runtime_python,
        "local_llm_model": config.hippo_llm_model,
        "local_embedding_model": config.hippo_embedding_model,
    }

    calls.clear()
    with pytest.raises(runtime.HoVerLocalRuntimeError, match="not canonical"):
        runtime.preflight_formal_runtime_config(
            replace(config, local_worker_cap=config.local_worker_cap - 1)
        )
    assert calls == []


def test_official_hippo_gateway_delegates_only_fixed_hover_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.default_formal_runtime_config(tmp_path)
    gateway = runtime.OfficialHippoGateway(
        runtime_python=config.hippo_runtime_python,
        local_llm_model=config.hippo_llm_model,
        local_embedding_model=config.hippo_embedding_model,
        base_binding_receipt_path=config.hippo_base_binding_receipt,
        attestation_receipt_path=config.hippo_attestation_receipt,
        stage_root=config.hippo_stage_root,
        work_root=config.hippo_work_root,
    )
    observed: dict[str, dict[str, object]] = {}

    def build(**kwargs: object) -> Mapping[str, object]:
        observed["build"] = dict(kwargs)
        return {"status": "synthetic_build"}

    expected_batch = RetrievalBatch(
        indices=((0, 1, 2, 3, 4),),
        receipt={"status": "synthetic_retrieve"},
    )

    def retrieve(**kwargs: object) -> RetrievalBatch:
        observed["retrieve"] = dict(kwargs)
        return expected_batch

    monkeypatch.setattr(runtime, "build_official_hipporag_global_index_v1", build)
    monkeypatch.setattr(
        runtime, "retrieve_official_hipporag_global_index_v1", retrieve
    )

    articles: Sequence[Mapping[str, object]] = (
        {"idx": 0, "title": "Synthetic", "body": "Offline body."},
    )
    assert gateway.build(articles) == {"status": "synthetic_build"}
    assert (
        gateway.retrieve(block="F_search", queries=("Synthetic claim",))
        is expected_batch
    )
    assert observed["build"] == {
        "articles": articles,
        "runtime_python": config.hippo_runtime_python,
        "local_llm_model": config.hippo_llm_model,
        "local_embedding_model": config.hippo_embedding_model,
        "base_binding_receipt_path": config.hippo_base_binding_receipt,
        "attestation_receipt_path": config.hippo_attestation_receipt,
        "stage_root": config.hippo_stage_root,
    }
    assert observed["retrieve"]["work_root"] == (
        config.hippo_work_root / "F_search"
    )
    with pytest.raises(runtime.HoVerLocalRuntimeError, match="stage is invalid"):
        gateway.retrieve(block="legacy_block", queries=("Synthetic claim",))


class _InputPipe:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.flush_count = 0
        self.closed = False

    def write(self, raw: bytes) -> int:
        self.writes.append(raw)
        return len(raw)

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


class _SyntheticProcess:
    def __init__(self, response: bytes) -> None:
        self.stdin = _InputPipe()
        self.stdout = io.BytesIO(response)
        self.stderr = io.BytesIO()
        self.wait_timeouts: list[int] = []
        self.killed = False

    def wait(self, *, timeout: int) -> int:
        self.wait_timeouts.append(timeout)
        return 0

    def kill(self) -> None:
        self.killed = True


def test_ner_client_uses_one_offline_row_minimal_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = {
        "status": "synthetic_runtime_verified",
        "canary_output_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        runtime, "verify_ner_runtime_binding", lambda **_kwargs: binding
    )
    process = _SyntheticProcess(encode_response(((),)))
    launch: dict[str, Any] = {}

    def popen(command: Sequence[str], **kwargs: object) -> _SyntheticProcess:
        launch["command"] = list(command)
        launch.update(kwargs)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    asset = tmp_path / "synthetic-asset.json"
    model = tmp_path / "synthetic-model"
    values = ({"kind": "query", "query": "Synthetic offline query?"},)

    client = runtime.OfflineNERJSONLClient(
        project_root=tmp_path,
        asset_manifest_path=asset,
        model_root=model,
    )
    assert launch["command"] == [
        runtime.sys.executable,
        "-B",
        "-m",
        "replication_runtime.multihoprag_ner_v1.worker",
        "--asset-manifest",
        str(asset),
        "--model-root",
        str(model),
        "--serve-jsonl",
    ]
    assert launch["cwd"] == tmp_path.resolve()
    environment = launch["env"]
    assert isinstance(environment, dict)
    assert environment["PYTHONPATH"] == str(tmp_path.resolve())
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""

    assert client.extract_inputs(values) == ((),)
    assert process.stdin.writes == [encode_request(values)]
    assert process.stdin.flush_count == 1
    assert client.canary_receipt == {
        "multihoprag_rows_or_archives_accessed": False,
        "output_sha256": "a" * 64,
        "status": "passed_exact_row_free_synthetic_canary",
        "worker_serve_loop_reached": True,
    }
    client.close()
    assert process.stdin.closed is True
    assert process.wait_timeouts == [30]
    assert process.killed is False


def test_runtime_source_has_no_legacy_runner_or_acquisition_dependency() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "multihoprag_joint_graph_formal_runner_v1" not in source
    assert "multihoprag_direct_acquisition_v1" not in source
