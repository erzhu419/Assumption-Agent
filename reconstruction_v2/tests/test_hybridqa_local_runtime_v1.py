from __future__ import annotations

from pathlib import Path

import pytest

from assumption_agent.benchmarks import hybridqa_local_runtime_v1 as runtime


def test_default_config_is_exact_and_project_scoped(tmp_path: Path) -> None:
    config = runtime.default_formal_runtime_config(tmp_path)
    assert config.project == tmp_path.resolve()
    assert config.hippo_stage_root == (
        tmp_path.resolve() / runtime.HIPPORAG_STAGE_RELATIVE
    )
    assert config.hippo_work_root == (
        tmp_path.resolve() / runtime.HIPPORAG_WORK_RELATIVE
    )
    assert config.minilm_model_root == (
        tmp_path.resolve() / "artifacts/qasper_minilm_runtime_v1/model"
    )


def test_preflight_is_model_free_and_binds_both_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.default_formal_runtime_config(tmp_path)
    calls: list[str] = []

    def minilm(**_kwargs):
        calls.append("minilm")
        return {"status": "synthetic_minilm"}

    def hippo(**_kwargs):
        calls.append("hippo")
        return {"status": "synthetic_hippo"}

    monkeypatch.setattr(runtime, "verify_minilm_runtime_binding", minilm)
    monkeypatch.setattr(runtime, "verify_formal_runtime_attestation_v3", hippo)
    receipt = runtime.preflight_formal_runtime_config(config)
    assert calls == ["minilm", "hippo"]
    assert receipt["model_inference_calls"] == 0
    assert receipt["benchmark_source_or_private_pack_reads"] == 0
    assert receipt["external_network_calls"] == 0


def test_preflight_rejects_existing_formal_runtime_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.default_formal_runtime_config(tmp_path)
    config.hippo_stage_root.mkdir(parents=True)
    monkeypatch.setattr(
        runtime,
        "verify_minilm_runtime_binding",
        lambda **_kwargs: {"status": "unused"},
    )
    monkeypatch.setattr(
        runtime,
        "verify_formal_runtime_attestation_v3",
        lambda **_kwargs: {"status": "unused"},
    )
    with pytest.raises(runtime.HybridQaLocalRuntimeError, match="already exists"):
        runtime.preflight_formal_runtime_config(config)


def test_gateway_stage_names_are_closed(tmp_path: Path) -> None:
    gateway = runtime.OfficialHippoGateway(
        runtime.default_formal_runtime_config(tmp_path)
    )
    with pytest.raises(runtime.HybridQaLocalRuntimeError, match="stage is invalid"):
        gateway.retrieve(block="A_form", queries=["query"])
