from __future__ import annotations

from pathlib import Path
import stat
import tempfile
from typing import Iterator

import pytest

from assumption_agent.benchmarks import hybridqa_local_runtime_v2 as runtime
from replication_runtime.hybridqa_official_hipporag_v1.contract import RetrievalBatch


@pytest.fixture
def private_project_root() -> Iterator[Path]:
    linux_tmp = Path("/tmp")
    parent = str(linux_tmp) if linux_tmp.is_dir() else None
    with tempfile.TemporaryDirectory(prefix="hybridqa-runtime-", dir=parent) as value:
        yield Path(value)


def test_default_config_is_exact_and_project_scoped(tmp_path: Path) -> None:
    config = runtime.default_formal_runtime_config(tmp_path)
    assert config.project == tmp_path.resolve()
    assert runtime.HIPPORAG_STAGE_RELATIVE == (
        runtime.acquisition.FORMAL_ROOT_RELATIVE / "official_hipporag_stage"
    )
    assert runtime.HIPPORAG_WORK_RELATIVE == (
        runtime.acquisition.FORMAL_ROOT_RELATIVE / "official_hipporag_work"
    )
    assert runtime.acquisition.FORMAL_ROOT_RELATIVE == Path(
        "artifacts/hybridqa_p6_e2_formal_v2"
    )
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
    assert receipt["schema"] == "hybridqa_local_runtime_v2_preflight"
    assert receipt["version"] == runtime.VERSION
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
    assert not gateway.config.hippo_work_root.exists()


def test_gateway_creates_one_shared_work_root_then_reuses_it_by_fixed_block(
    private_project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.default_formal_runtime_config(private_project_root)
    config.hippo_work_root.parent.mkdir(parents=True)
    gateway = runtime.OfficialHippoGateway(config)
    expected = RetrievalBatch(
        indices=((0, 1, 2, 3, 4),),
        receipt={"status": "synthetic_retrieve"},
    )
    calls: list[dict[str, object]] = []

    def retrieve(**kwargs: object) -> RetrievalBatch:
        calls.append(dict(kwargs))
        return expected

    monkeypatch.setattr(
        runtime, "retrieve_official_hipporag_global_index_v1", retrieve
    )

    assert gateway.retrieve(
        block="A_form_F_search_A_hold", queries=("first query",)
    ) is expected
    assert config.hippo_work_root.is_dir()
    assert stat.S_IMODE(config.hippo_work_root.lstat().st_mode) == 0o700
    assert gateway.retrieve(block="M_search", queries=("second query",)) is expected
    assert [call["work_root"] for call in calls] == [
        config.hippo_work_root / "A_form_F_search_A_hold",
        config.hippo_work_root / "M_search",
    ]
    assert [call["queries"] for call in calls] == [
        ("first query",),
        ("second query",),
    ]
    assert all(call["stage_root"] == config.hippo_stage_root for call in calls)
    assert not (config.hippo_work_root / "A_form_F_search_A_hold").exists()
    assert not (config.hippo_work_root / "M_search").exists()


def test_gateway_rejects_non_directory_work_root_before_runtime_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.default_formal_runtime_config(tmp_path)
    config.hippo_work_root.parent.mkdir(parents=True)
    config.hippo_work_root.write_text("unsafe", encoding="ascii")
    gateway = runtime.OfficialHippoGateway(config)
    calls = 0

    def retrieve(**_kwargs: object) -> RetrievalBatch:
        nonlocal calls
        calls += 1
        raise AssertionError("official runtime must not be called")

    monkeypatch.setattr(
        runtime, "retrieve_official_hipporag_global_index_v1", retrieve
    )
    with pytest.raises(runtime.HybridQaLocalRuntimeError, match="work root is unsafe"):
        gateway.retrieve(
            block="A_form_F_search_A_hold", queries=("synthetic query",)
        )
    assert calls == 0


def test_gateway_rejects_reusable_work_root_with_broad_permissions(
    private_project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.default_formal_runtime_config(private_project_root)
    config.hippo_work_root.mkdir(parents=True, mode=0o700)
    config.hippo_work_root.chmod(0o750)
    gateway = runtime.OfficialHippoGateway(config)
    calls = 0

    def retrieve(**_kwargs: object) -> RetrievalBatch:
        nonlocal calls
        calls += 1
        raise AssertionError("official runtime must not be called")

    monkeypatch.setattr(
        runtime, "retrieve_official_hipporag_global_index_v1", retrieve
    )
    with pytest.raises(runtime.HybridQaLocalRuntimeError, match="work root is unsafe"):
        gateway.retrieve(block="M_search", queries=("synthetic query",))
    assert calls == 0
