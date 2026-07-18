from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks import hover_joint_graph_formal_controller_v1 as controller
from assumption_agent.benchmarks import hover_joint_graph_formal_runner_v1 as runner


@dataclass(frozen=True)
class FakePrepared:
    corpus_token: str = "prepared_once"


@dataclass(frozen=True)
class FakeStage:
    block: str


def _artifact(kind: str, block: str | None) -> controller.LifecycleArtifact:
    payload = {"kind": kind, "block": block, "synthetic": True}
    return controller.LifecycleArtifact(
        kind=kind,
        block=block,
        receipt_sha256=controller.stable_hash(payload),
        payload=payload,
    )


class FakeAcquisition:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def verify_prerequisites(
        self, *, project: Path
    ) -> controller.PrerequisiteBinding:
        assert project.is_dir()
        self.events.append("verify")
        return controller.PrerequisiteBinding("a" * 64, "b" * 64)

    def load_corpus_view(self, *, project: Path) -> Mapping[str, Any]:
        self.events.append("load_corpus")
        return {"synthetic_corpus": True}

    def load_block_view(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]:
        self.events.append(f"load_view:{expected_block}")
        return {"block": expected_block, "late_utility_fields_included": False}

    def load_block_labels(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]:
        assert expected_block != "F_search"
        self.events.append(f"load_labels:{expected_block}")
        return {"block": expected_block, "late": True}

    def archive_stage(
        self, *, project: Path, prepared: object, stage: object
    ) -> controller.LifecycleArtifact:
        assert prepared == FakePrepared()
        assert isinstance(stage, FakeStage)
        self.events.append(f"archive:{stage.block}")
        return _artifact("stage_archive", stage.block)

    def seal_stage(
        self,
        *,
        project: Path,
        block: str,
        archive: controller.LifecycleArtifact,
    ) -> controller.LifecycleArtifact:
        archive.validate(kind="stage_archive", block=block)
        self.events.append(f"seal:{block}")
        return _artifact("action_seal", block)

    def freeze_f_policies(
        self,
        *,
        project: Path,
        policies: controller.PolicyPair,
        archive: controller.LifecycleArtifact,
    ) -> controller.LifecycleArtifact:
        assert policies.identifiable
        archive.validate(kind="stage_archive", block="F_search")
        self.events.append("freeze:F_search")
        return _artifact("policy_freeze", "F_search")

    def authorize_promotion(
        self,
        *,
        project: Path,
        outcome: controller.AHoldOutcome,
        policy_freeze: controller.LifecycleArtifact,
        archive: controller.LifecycleArtifact,
        seal: controller.LifecycleArtifact,
    ) -> controller.LifecycleArtifact:
        assert outcome.promoted
        policy_freeze.validate(kind="policy_freeze", block="F_search")
        archive.validate(kind="stage_archive", block="A_hold")
        seal.validate(kind="action_seal", block="A_hold")
        self.events.append("authorize_promotion")
        return _artifact("promotion_authorization", "A_hold")


class FakeNERContext(AbstractContextManager[object]):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self) -> object:
        self.events.append("ner_enter")
        return object()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.events.append("ner_exit")


class FakeRuntimeFactory:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def create_encoder(self, config: controller.FormalRuntimeConfig) -> object:
        self.events.append("create_encoder")
        return object()

    def create_hippo(self, config: controller.FormalRuntimeConfig) -> object:
        self.events.append("create_hippo")
        return object()

    def create_ner_context(
        self, config: controller.FormalRuntimeConfig
    ) -> AbstractContextManager[object]:
        self.events.append("create_ner")
        return FakeNERContext(self.events)


class FakeCore:
    def __init__(
        self,
        events: list[str],
        *,
        identifiable: bool = True,
        promoted: bool = True,
        fail_block: str | None = None,
    ) -> None:
        self.events = events
        self.identifiable = identifiable
        self.promoted = promoted
        self.fail_block = fail_block
        self.prepare_count = 0

    def prepare(
        self,
        *,
        corpus_view: Mapping[str, Any],
        encoder: object,
        ner: object,
        hippo: object,
        config: controller.FormalRuntimeConfig,
    ) -> FakePrepared:
        assert corpus_view == {"synthetic_corpus": True}
        self.events.append("prepare")
        self.prepare_count += 1
        return FakePrepared()

    def execute(
        self,
        *,
        block: str,
        view: Mapping[str, Any],
        prepared: object,
        encoder: object,
        ner: object,
        hippo: object,
        config: controller.FormalRuntimeConfig,
    ) -> FakeStage:
        assert view["block"] == block
        self.events.append(f"execute:{block}")
        if block == self.fail_block:
            raise RuntimeError("synthetic terminal failure")
        return FakeStage(block)

    def descriptive(
        self, *, stage: object, labels: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        assert stage == FakeStage("A_form") and labels["block"] == "A_form"
        self.events.append("descriptive:A_form")
        return {"synthetic_descriptive": True, "external_network_calls": 0}

    def select_f_policies(self, *, stage: object) -> controller.PolicyPair:
        assert stage == FakeStage("F_search")
        self.events.append("select:F_search")
        return controller.PolicyPair(
            e0=controller.PolicyHandle("E0", "P0_IND_SUM", "c" * 64, object()),
            e1=controller.PolicyHandle(
                "E1", "P2_ENTITY_BRIDGE", "d" * 64, object()
            ),
            identifiable=self.identifiable,
        )

    def assess_a_hold(
        self,
        *,
        stage: object,
        labels: Mapping[str, Any],
        f_stage: object,
        policies: controller.PolicyPair,
    ) -> controller.AHoldOutcome:
        assert stage == FakeStage("A_hold")
        assert f_stage == FakeStage("F_search")
        assert labels["block"] == "A_hold" and policies.identifiable
        self.events.append("assess:A_hold")
        return controller.AHoldOutcome(
            primary_passed=True,
            promoted=self.promoted,
            report={
                "promoted": self.promoted,
                "delta_total": [1 if self.promoted else 0, 1],
                "signflip_p": [1, 16],
            },
        )

    def assess_m_search(
        self,
        *,
        stage: object,
        labels: Mapping[str, Any],
        f_stage: object,
        policies: controller.PolicyPair,
    ) -> controller.MSearchOutcome:
        assert stage == FakeStage("M_search")
        assert f_stage == FakeStage("F_search")
        assert labels["block"] == "M_search" and policies.identifiable
        self.events.append("assess:M_search")
        return controller.MSearchOutcome(
            l5_passed=True,
            report={"l5_passed": True, "delta_total": [1, 1], "p": [1, 16]},
        )


def _config(project: Path) -> controller.FormalRuntimeConfig:
    path = project / "unused"
    return controller.FormalRuntimeConfig(
        project=project,
        hippo_runtime_python=path / "python",
        hippo_llm_model=path / "llm",
        hippo_embedding_model=path / "embedding",
        hippo_base_binding_receipt=path / "base.json",
        hippo_attestation_receipt=path / "attestation.json",
        hippo_stage_root=path / "stage",
        hippo_work_root=path / "work",
        minilm_asset_manifest=path / "minilm.json",
        minilm_model_root=path / "minilm",
        ner_asset_manifest=path / "ner.json",
        ner_model_root=path / "ner",
        local_worker_cap=runner.LOCAL_CONCURRENCY_CAP,
        ner_batch_size=runner.DEFAULT_NER_BATCH_SIZE,
    )


def _synthetic_root(tmp_path: Path) -> controller.FormalRuntimeConfig:
    (tmp_path / controller.SYNTHETIC_SENTINEL).write_text(
        controller.SYNTHETIC_SENTINEL_CONTENT, encoding="ascii"
    )
    return _config(tmp_path)


def _run(
    tmp_path: Path,
    *,
    identifiable: bool = True,
    promoted: bool = True,
    fail_block: str | None = None,
) -> tuple[dict[str, Any], list[str], FakeCore]:
    events: list[str] = []
    core = FakeCore(
        events,
        identifiable=identifiable,
        promoted=promoted,
        fail_block=fail_block,
    )
    result = controller.run_synthetic_lifecycle(
        _synthetic_root(tmp_path),
        acquisition_adapter=FakeAcquisition(events),
        runtime_factory=FakeRuntimeFactory(events),
        core=core,
    )
    return result, events, core


def _assert_before(events: list[str], left: str, right: str) -> None:
    assert events.index(left) < events.index(right)


def test_promotion_path_is_authorized_incrementally_and_builds_corpus_once(
    tmp_path: Path,
) -> None:
    result, events, core = _run(tmp_path)
    assert result["status"] == "formal_M_search_complete"
    assert result["L5_passed"] is True
    assert result["external_network_calls"] == 0
    assert result["online_evaluator_calls"] == 0
    assert core.prepare_count == 1
    assert [row for row in events if row.startswith("execute:")] == [
        "execute:A_form",
        "execute:F_search",
        "execute:A_hold",
        "execute:M_search",
    ]
    assert not any(row == "load_labels:F_search" for row in events)
    _assert_before(events, "seal:A_form", "load_labels:A_form")
    _assert_before(events, "freeze:F_search", "load_view:A_hold")
    _assert_before(events, "seal:A_hold", "load_labels:A_hold")
    _assert_before(events, "authorize_promotion", "load_view:M_search")
    _assert_before(events, "seal:M_search", "load_labels:M_search")
    controller.verify_self_hash(result, "result_sha256")

    root = tmp_path / controller.SYNTHETIC_ROOT_RELATIVE
    stored = json.loads((root / "result.json").read_text(encoding="ascii"))
    marker = json.loads(
        (root / "runner.one_shot_marker.json").read_text(encoding="ascii")
    )
    descriptive = json.loads(
        (root / "A_form.descriptive.json").read_text(encoding="ascii")
    )
    assert stored == result
    controller.verify_self_hash(marker, "marker_sha256")
    controller.verify_self_hash(descriptive, "descriptive_receipt_sha256")
    assert not (root / "runner.terminal_failure.json").exists()


def test_unidentifiable_f_is_terminal_without_policy_freeze_a_hold_or_m(
    tmp_path: Path,
) -> None:
    result, events, _core = _run(tmp_path, identifiable=False)
    assert result["status"] == (
        "valid_F_search_nonidentifiable_A_hold_and_M_unopened"
    )
    assert result["F_search_policy_freeze_created"] is False
    assert result["A_hold_view_or_labels_opened"] is False
    assert result["M_search_view_or_labels_opened"] is False
    assert "freeze:F_search" not in events
    assert not any("A_hold" in row or "M_search" in row for row in events)
    controller.verify_self_hash(result, "result_sha256")


def test_nonpromotion_is_terminal_and_keeps_m_view_and_labels_unopened(
    tmp_path: Path,
) -> None:
    result, events, _core = _run(tmp_path, promoted=False)
    assert result["status"] == "valid_A_hold_nonpromotion_M_unopened"
    assert result["A_hold_primary_passed"] is True
    assert result["M_search_view_or_labels_opened"] is False
    assert "authorize_promotion" not in events
    assert not any("M_search" in row for row in events)
    _assert_before(events, "seal:A_hold", "load_labels:A_hold")
    controller.verify_self_hash(result, "result_sha256")


def test_failure_after_marker_burns_lifecycle_and_retry_cannot_reopen_views(
    tmp_path: Path,
) -> None:
    config = _synthetic_root(tmp_path)
    events: list[str] = []
    acquisition_adapter = FakeAcquisition(events)
    runtime_factory = FakeRuntimeFactory(events)
    core = FakeCore(events, fail_block="A_hold")
    with pytest.raises(RuntimeError, match="synthetic terminal failure"):
        controller.run_synthetic_lifecycle(
            config,
            acquisition_adapter=acquisition_adapter,
            runtime_factory=runtime_factory,
            core=core,
        )
    root = tmp_path / controller.SYNTHETIC_ROOT_RELATIVE
    assert (root / "runner.one_shot_marker.json").is_file()
    failure = json.loads(
        (root / "runner.terminal_failure.json").read_text(encoding="ascii")
    )
    assert failure["status"] == "terminal_cohort_burned_no_replay"
    assert failure["failure_stage"] == (
        "A_hold_claim_view_and_gold_free_execution"
    )
    controller.verify_self_hash(failure, "failure_sha256")
    assert not (root / "result.json").exists()

    before_retry = len(events)
    with pytest.raises(
        controller.HoVerFormalControllerError, match="exclusive output"
    ):
        controller.run_synthetic_lifecycle(
            config,
            acquisition_adapter=acquisition_adapter,
            runtime_factory=runtime_factory,
            core=FakeCore(events),
        )
    assert events[before_retry:] == ["verify"]


def test_synthetic_entrypoint_requires_sentinel_before_any_capability_access(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    with pytest.raises(
        controller.HoVerFormalControllerError, match="sentinel"
    ):
        controller.run_synthetic_lifecycle(
            _config(tmp_path),
            acquisition_adapter=FakeAcquisition(events),
            runtime_factory=FakeRuntimeFactory(events),
            core=FakeCore(events),
        )
    assert events == []
