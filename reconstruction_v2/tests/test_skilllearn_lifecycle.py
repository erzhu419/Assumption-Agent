from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import pytest

from assumption_agent.archive import PolicyArchive
from assumption_agent.benchmarks import (
    SkillLearnBenchAdapter,
    SkillLearnBackendPool,
    SkillLearnEvolutionHarness,
    SkillLearnPrebuiltImageCache,
    SkillLearnProgramCompiler,
    SkillLearnProviderCircuit,
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrainingEvidenceReplayCache,
    TrialVariant,
)
from assumption_agent.evaluation import PromotionGate, PromotionGateSpec
from assumption_agent.evolution import CounterfactualEvidenceReplayCache
from assumption_agent.events import MemoryEventSink
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnSubprocessBackend,
    _cleanup_ephemeral_codex_home,
)
from assumption_agent.benchmarks.skilllearn_experiment import _run_paired_arms
from assumption_agent.models import HypothesisProgram, HypothesisStatus, SplitName
from assumption_agent.proposer import StructuredHypothesisProposer
from assumption_agent.splits import (
    SplitAccessGuard,
    build_family_out_manifest,
    build_instance_holdout_manifest,
)
from assumption_agent.validation import (
    EvaluatorEpochCheck,
    RecursiveValidationEngine,
    RuntimeCandidateKindCheck,
    RuntimeActionCheck,
    SchemaCheck,
    TrainingSupportCheck,
    TriggerVocabularyCheck,
)


BENCH_ROOT = (
    Path(__file__).resolve().parents[1]
    / "reference"
    / "self_evo_continual_20260707"
    / "repos"
    / "SkillLearnBench"
)


class QueueProposalModel:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[Mapping[str, Any]] = []

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.requests.append(payload)
        return self.responses.pop(0)


class FakeSkillLearnBackend:
    agent_id = "codex"
    model = "gpt-5.3-codex-spark"
    max_steps = 20

    def __init__(
        self,
        *,
        invalid_candidate_item: str | None = None,
        invalid_training_item: str | None = None,
        invalid_candidate_once: bool = False,
        invalid_training_once: bool = False,
    ) -> None:
        self.invalid_candidate_item = invalid_candidate_item
        self.invalid_training_item = invalid_training_item
        self.invalid_candidate_once = invalid_candidate_once
        self.invalid_training_once = invalid_training_once
        self._invalidated_candidate = False
        self._invalidated_training = False
        self.calls: list[tuple[str, str, bool]] = []
        self.request_hashes: list[str] = []

    def run(self, request, *, skill_source_dir, trace_id):
        has_skill = skill_source_dir is not None and skill_source_dir.is_dir()
        self.calls.append((request.item_id, request.variant.value, has_skill))
        self.request_hashes.append(request.request_hash)
        invalid_candidate = (
            request.variant is TrialVariant.POLICY_ON
            and request.item_id == self.invalid_candidate_item
            and (not self.invalid_candidate_once or not self._invalidated_candidate)
        )
        invalid_training = (
            request.split is SplitName.TRAIN
            and request.item_id == self.invalid_training_item
            and (not self.invalid_training_once or not self._invalidated_training)
        )
        invalid = invalid_candidate or invalid_training
        self._invalidated_candidate = self._invalidated_candidate or invalid_candidate
        self._invalidated_training = self._invalidated_training or invalid_training
        success = has_skill and not invalid
        return SkillLearnTrialObservation(
            request=request,
            success=success,
            score=float(success),
            metrics={
                "task_success": float(success),
                "trajectory_key_point_recall": 0.2 if not success else 1.0,
                "evaluation_valid": float(not invalid),
            },
            total_tokens=100,
            steps=10,
            duration_seconds=0.1,
            provider_fingerprint="provider-fixed",
            fairness_fingerprint="budget-fixed",
            error_type="endpoint_error" if invalid else None,
            upstream_result_hash=f"result-{request.request_hash}",
        )


class AlwaysFailSkillLearnBackend(FakeSkillLearnBackend):
    def run(self, request, *, skill_source_dir, trace_id):
        observation = super().run(
            request,
            skill_source_dir=skill_source_dir,
            trace_id=trace_id,
        )
        metrics = dict(observation.metrics)
        metrics["task_success"] = 0.0
        return replace(
            observation,
            success=False,
            score=0.0,
            metrics=metrics,
        )


class RecordingSubprocess:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.agent_stdout = ""

    def run(self, args, *positional, **kwargs):
        command = list(args)
        self.commands.append(command)
        self.kwargs.append(dict(kwargs))
        stdout = self.agent_stdout if "codex exec" in " ".join(command) else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


class FakeDockerSubprocess:
    def __init__(self) -> None:
        self.images: dict[str, dict[str, Any]] = {}
        self.volumes: dict[str, dict[str, Any]] = {}
        self.commands: list[list[str]] = []
        self.base_contexts: list[Path] = []
        self.skill_stubs_present: list[bool] = []

    def run(self, args, *positional, **kwargs):
        command = list(args)
        self.commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            tag = command[3]
            image = self.images.get(tag)
            if image is None:
                return SimpleNamespace(returncode=1, stdout="", stderr="not found")
            return SimpleNamespace(returncode=0, stdout=json.dumps([image]), stderr="")
        if command[:3] == ["docker", "volume", "inspect"]:
            volume = self.volumes.get(command[3])
            if volume is None:
                return SimpleNamespace(returncode=1, stdout="", stderr="not found")
            return SimpleNamespace(returncode=0, stdout=json.dumps([volume]), stderr="")
        if command[:3] == ["docker", "volume", "create"]:
            name = command[-1]
            key_label = command[command.index("--label") + 1]
            self.volumes[name] = {
                "Name": name,
                "Labels": {key_label.split("=", 1)[0]: key_label.split("=", 1)[1]},
            }
            return SimpleNamespace(returncode=0, stdout=name, stderr="")
        if command[:3] == ["docker", "volume", "rm"]:
            self.volumes.pop(command[-1], None)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:2] == ["docker", "run"]:
            stdout = "codex-cli 0.144.1\n" if command[-1] == "codex --version" else ""
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        if command[:2] == ["docker", "build"] and command[-1] != "-":
            self.base_contexts.append(Path(command[-1]))
            self.skill_stubs_present.append(
                (Path(command[-1]) / "skills" / "tool").is_dir()
            )
            tag = command[command.index("-t") + 1]
            key_label = next(
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--label"
                and command[index + 1].startswith("org.assumption-agent.prebuild.key=")
            )
            cache_key = key_label.split("=", 1)[1]
            self.images[tag] = {
                "Id": f"sha256:{cache_key}",
                "Config": {
                    "Labels": {"org.assumption-agent.prebuild.key": cache_key}
                },
            }
            return SimpleNamespace(returncode=0, stdout="base built", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class ConcurrentFakeBackend(FakeSkillLearnBackend):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def run(self, request, *, skill_source_dir, trace_id):
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(0.02)
            return super().run(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
        finally:
            with self._lock:
                self.active -= 1


def test_real_manifest_lifecycle_promotes_and_seals_test(tmp_path: Path) -> None:
    harness, backend, model, archive, guard, sink = _harness(tmp_path)
    train_ids = (
        "organize-messy-files-1",
        "organize-messy-files-2",
        "offer-letter-generator-1",
        "offer-letter-generator-2",
    )
    validation_ids = ("organize-messy-files-5", "offer-letter-generator-5")

    result = harness.run_generation(
        train_item_ids=train_ids,
        validation_item_ids=validation_ids,
        trace_id="real-manifest-generation",
    )

    assert result.evolution is not None
    assert result.evolution.promoted is True
    assert len(result.residuals) == 4
    assert all(row.context["task_instruction"] for row in result.residuals)
    assert all(row.split.value == "train" for row in result.residuals)
    assert guard.test_accessed is False
    assert len(backend.calls) == 8
    assert archive.incumbent_id == result.evolution.archive_node.id
    promoted = archive.hypotheses[result.evolution.accepted_hypothesis_id]
    assert promoted.status is HypothesisStatus.PROMOTED
    assert any(row["event"] == "skilllearn_counterfactual_pair_completed" for row in sink.events)

    serialized_requests = json.dumps(model.requests, sort_keys=True)
    for test_id in guard.manifest.test_ids:
        assert test_id not in serialized_requests
    assert "task_instruction" in serialized_requests
    assert "correct_answer" not in serialized_requests
    assert not _contains_forbidden_answer_key(model.requests)

    second = harness.run_generation(
        train_item_ids=train_ids,
        validation_item_ids=validation_ids,
        trace_id="incumbent-training-replay",
    )
    assert second.evolution is None
    assert second.reason == "no_valid_failed_training_rows"
    assert second.baseline_hypothesis_ids == (promoted.id,)
    assert len(model.requests) == 1
    assert all(variant == "policy_off" and has_skill for _, variant, has_skill in backend.calls[-4:])

    calls_before_test = len(backend.calls)
    with pytest.raises(PermissionError, match="archive must be frozen"):
        harness.run_sealed_test(
            promoted,
            test_item_ids=("organize-messy-files-4",),
        )
    assert len(backend.calls) == calls_before_test
    assert guard.test_accessed is False

    guard.freeze_archive()
    pairs = harness.run_sealed_test(
        promoted,
        test_item_ids=("organize-messy-files-4", "offer-letter-generator-3"),
    )
    assert len(pairs) == 2
    assert guard.test_accessed is True
    assert all(pair.candidate_outcome.success for pair in pairs)


def test_root_proposal_failure_preserves_a_terminal_generation_report(
    tmp_path: Path,
) -> None:
    harness, backend, model, _, guard, sink = _harness(tmp_path)
    model.responses.clear()

    result = harness.run_generation(
        train_item_ids=(
            "organize-messy-files-1",
            "offer-letter-generator-1",
        ),
        validation_item_ids=(
            "organize-messy-files-5",
            "offer-letter-generator-5",
        ),
        trace_id="root-proposal-model-failure",
    )

    assert result.evolution is None
    assert result.reason == "proposal_model_failed"
    assert result.to_dict()["proposal_model_failure_count"] == 1
    assert len(backend.calls) == 2
    assert guard.test_accessed is False
    assert any(
        row["event"] == "skilllearn_generation_stopped_after_proposal_model_failure"
        and row["payload"]["performance_claim_eligible"] is False
        for row in sink.events
    )
    assert any(
        row["event"] == "skilllearn_evolution_generation_completed"
        and row["payload"]["reason"] == "proposal_model_failed"
        for row in sink.events
    )


def test_repair_model_failure_blocks_generation_promotion(
    tmp_path: Path,
) -> None:
    bad = _program_dict()
    bad["id"] = "hyp-needs-repair"
    for action in bad["action_graph"]:
        if action.get("target") == "skilllearn_challenger":
            action["target"] = "missing_lane"
    good = _program_dict()
    good["id"] = "hyp-statically-valid"
    harness, backend, _, archive, guard, sink = _harness(
        tmp_path,
        proposal_rows=[bad, good],
    )

    result = harness.run_generation(
        train_item_ids=(
            "organize-messy-files-1",
            "offer-letter-generator-1",
        ),
        validation_item_ids=(
            "organize-messy-files-5",
            "offer-letter-generator-5",
        ),
        trace_id="repair-failure-blocks-generation",
    )

    assert result.evolution is not None
    assert result.reason == "proposal_model_failed"
    assert result.evolution.repair_model_failure_count == 1
    assert result.evolution.promotion_decision is None
    assert result.to_dict()["proposal_model_failure_count"] == 1
    assert archive.incumbent_id is None
    assert len(backend.calls) == 2
    assert guard.test_accessed is False
    assert any(
        row["event"] == "evolution_generation_blocked_by_repair_model_failure"
        and row["payload"]["counterfactual_validation_executed"] is False
        for row in sink.events
    )


def test_invalid_external_pair_blocks_promotion(tmp_path: Path) -> None:
    harness, _, _, _, _, _ = _harness(
        tmp_path,
        invalid_candidate_item="offer-letter-generator-5",
    )

    result = harness.run_generation(
        train_item_ids=(
            "organize-messy-files-1",
            "organize-messy-files-2",
            "offer-letter-generator-1",
            "offer-letter-generator-2",
        ),
        validation_item_ids=("organize-messy-files-5", "offer-letter-generator-5"),
    )

    assert result.evolution is not None
    assert result.evolution.promoted is False
    assert "invalid_counterfactual_pairs" in result.evolution.promotion_decision.blockers


def test_invalid_counterfactual_bundle_does_not_enter_replay_cache(
    tmp_path: Path,
) -> None:
    target = "offer-letter-generator-5"
    harness, backend, _, _, _, sink = _harness(
        tmp_path,
        invalid_candidate_item=target,
    )
    cache = CounterfactualEvidenceReplayCache(event_sink=sink)
    program = HypothesisProgram.from_dict(_program_dict())
    tasks = harness.tasks((target,))

    first = cache.run_or_replay(
        runner=harness.counterfactual_runner,
        tasks=tasks,
        program=program,
        baseline_programs=(),
        split=SplitName.VALIDATION,
        trace_id="invalid-cache-source",
    )
    second = cache.run_or_replay(
        runner=harness.counterfactual_runner,
        tasks=tasks,
        program=program,
        baseline_programs=(),
        split=SplitName.VALIDATION,
        trace_id="invalid-cache-target",
    )

    assert first[0].candidate_outcome.metrics["evaluation_valid"] == 0.0
    assert second[0].candidate_outcome.metrics["evaluation_valid"] == 0.0
    assert len(backend.calls) == 4
    assert not any(
        row["event"] == "counterfactual_evidence_replayed"
        for row in sink.events
    )
    assert sum(
        row["event"] == "counterfactual_evidence_not_recorded_invalid"
        for row in sink.events
    ) == 2


def test_invalid_counterfactual_arm_is_cleanly_retried_before_promotion(
    tmp_path: Path,
) -> None:
    backend = FakeSkillLearnBackend(
        invalid_candidate_item="offer-letter-generator-5",
        invalid_candidate_once=True,
    )
    harness, _, _, _, _, sink = _harness(
        tmp_path,
        backend_override=backend,
        invalid_trial_max_attempts=2,
    )

    result = harness.run_generation(
        train_item_ids=(
            "organize-messy-files-1",
            "organize-messy-files-2",
            "offer-letter-generator-1",
            "offer-letter-generator-2",
        ),
        validation_item_ids=(
            "organize-messy-files-5",
            "offer-letter-generator-5",
        ),
        trace_id="counterfactual-clean-retry",
    )

    assert result.evolution is not None
    assert result.evolution.promoted is True
    assert result.evolution.promotion_decision.summary.invalid_pair_count == 0
    assert len(backend.calls) == 9
    assert any(
        row["event"] == "skilllearn_invalid_trial_clean_replacement"
        and row["payload"]["same_request_key"] is True
        for row in sink.events
    )


def test_invalid_training_evidence_blocks_proposal(tmp_path: Path) -> None:
    invalid_item = "organize-messy-files-1"
    harness, backend, model, _, _, sink = _harness(
        tmp_path,
        invalid_training_item=invalid_item,
    )

    with pytest.raises(RuntimeError, match="training evidence contains invalid"):
        harness.run_generation(
            train_item_ids=(
                invalid_item,
                "organize-messy-files-2",
                "offer-letter-generator-1",
                "offer-letter-generator-2",
            ),
            validation_item_ids=(
                "organize-messy-files-5",
                "offer-letter-generator-5",
            ),
        )

    assert len(backend.calls) == 4
    assert model.requests == []
    assert any(
        row["event"] == "skilllearn_training_evidence_blocked"
        and row["payload"]["invalid_observation_count"] == 1
        and row["payload"]["proposal_blocked"] is True
        for row in sink.events
    )


def test_invalid_training_trial_is_cleanly_retried_with_same_request_key(
    tmp_path: Path,
) -> None:
    invalid_item = "organize-messy-files-1"
    backend = FakeSkillLearnBackend(
        invalid_training_item=invalid_item,
        invalid_training_once=True,
    )
    harness, _, _, _, _, sink = _harness(
        tmp_path,
        backend_override=backend,
        invalid_trial_max_attempts=2,
    )

    observations = harness.collect_training_observations(
        train_item_ids=(invalid_item, "organize-messy-files-2"),
        trace_id="training-clean-retry",
    )

    assert all(row.valid for row in observations)
    assert len(backend.calls) == 3
    assert backend.request_hashes[0] == backend.request_hashes[1]
    replacement = next(
        row
        for row in sink.events
        if row["event"] == "skilllearn_invalid_trial_clean_replacement"
    )
    assert replacement["payload"]["same_request_key"] is True
    assert replacement["payload"]["attempt"] == 2


def test_training_evidence_replay_avoids_identical_generation_resampling(
    tmp_path: Path,
) -> None:
    harness, backend, _, _, _, sink = _harness(tmp_path)
    cache = TrainingEvidenceReplayCache(event_sink=sink)
    train_ids = (
        "organize-messy-files-1",
        "organize-messy-files-2",
        "offer-letter-generator-1",
        "offer-letter-generator-2",
    )

    first = harness.collect_training_observations(
        train_item_ids=train_ids,
        training_replay_cache=cache,
        trace_id="training-replay-source",
    )
    second = harness.collect_training_observations(
        train_item_ids=train_ids,
        training_replay_cache=cache,
        trace_id="training-replay-target",
    )

    assert second == first
    assert len(backend.calls) == len(train_ids)
    replay = next(
        row for row in sink.events if row["event"] == "training_evidence_replayed"
    )
    assert replay["payload"]["behavior_identical"] is True
    assert replay["payload"]["new_training_executions"] == 0


def test_paired_ablation_shares_first_train_checkpoint_and_root(tmp_path: Path) -> None:
    recursive, recursive_backend, recursive_model, _, _, recursive_sink = _harness(
        tmp_path / "recursive"
    )
    no_recursive, no_recursive_backend, no_recursive_model, _, _, _ = _harness(
        tmp_path / "no-recursive"
    )
    no_recursive.validator.proposer = None
    train_ids = (
        "organize-messy-files-1",
        "organize-messy-files-2",
        "offer-letter-generator-1",
        "offer-letter-generator-2",
    )
    validation_ids = ("organize-messy-files-5", "offer-letter-generator-5")

    paired = _run_paired_arms(
        recursive_harness=recursive,
        no_recursive_harness=no_recursive,
        train_ids=train_ids,
        validation_ids=validation_ids,
        manifest_hash=recursive.manifest.manifest_hash,
        max_generations=1,
        max_consecutive_non_promotions=1,
    )

    recursive_first = paired["recursive_generations"][0]
    no_recursive_first = paired["no_recursive_generations"][0]
    assert len(paired["checkpoint_hash"]) == 64
    assert recursive_first.evolution.root_hypothesis_id == no_recursive_first.evolution.root_hypothesis_id
    assert recursive_first.train_observations == no_recursive_first.train_observations
    assert len(recursive_model.requests) == 1
    assert len(no_recursive_model.requests) == 0
    assert len(recursive_backend.calls) == 8
    assert len(no_recursive_backend.calls) == 0
    assert (
        recursive_first.evolution.promotion_decision.summary
        == no_recursive_first.evolution.promotion_decision.summary
    )
    replay_event = next(
        row
        for row in recursive_sink.events
        if row["event"] == "counterfactual_evidence_replayed"
    )
    assert replay_event["payload"]["pair_count"] == len(validation_ids)
    assert replay_event["payload"]["new_counterfactual_executions"] == 0


def test_paired_ablation_replays_later_root_when_arm_state_is_identical(
    tmp_path: Path,
) -> None:
    first = _program_dict()
    first["id"] = "generation-one-root"
    second = _program_dict()
    second["id"] = "generation-two-root"
    second["action_graph"][1]["value"] += " Then verify the final artifact exists."
    recursive_backend = AlwaysFailSkillLearnBackend()
    no_recursive_backend = AlwaysFailSkillLearnBackend()
    recursive, _, recursive_model, _, _, recursive_sink = _harness(
        tmp_path / "recursive",
        backend_override=recursive_backend,
        proposal_rows=[first],
    )
    no_recursive, _, no_recursive_model, _, _, _ = _harness(
        tmp_path / "no-recursive",
        backend_override=no_recursive_backend,
    )
    recursive_model.responses.append({"hypotheses": [second]})
    no_recursive.proposer = recursive.proposer
    no_recursive.kernel.proposer = recursive.proposer
    no_recursive.validator.proposer = None
    train_ids = (
        "organize-messy-files-1",
        "organize-messy-files-2",
        "offer-letter-generator-1",
        "offer-letter-generator-2",
    )
    validation_ids = ("organize-messy-files-5", "offer-letter-generator-5")

    paired = _run_paired_arms(
        recursive_harness=recursive,
        no_recursive_harness=no_recursive,
        train_ids=train_ids,
        validation_ids=validation_ids,
        manifest_hash=recursive.manifest.manifest_hash,
        max_generations=2,
        max_consecutive_non_promotions=2,
    )

    recursive_second = paired["recursive_generations"][1]
    no_recursive_second = paired["no_recursive_generations"][1]
    assert recursive_second.evolution.root_hypothesis_id == "generation-two-root"
    assert (
        no_recursive_second.evolution.root_hypothesis_id
        == recursive_second.evolution.root_hypothesis_id
    )
    assert len(recursive_model.requests) == 2
    assert len(no_recursive_model.requests) == 0
    assert len(recursive_backend.calls) == 12
    assert len(no_recursive_backend.calls) == 0
    proposal_replay = next(
        row
        for row in recursive_sink.events
        if row["event"] == "root_proposal_evidence_replayed"
        and row["payload"]["target_trace_id"].endswith("no-recursive-g2")
    )
    assert proposal_replay["payload"]["new_proposal_model_executions"] == 0
    assert proposal_replay["payload"]["request_identical"] is True
    assert sum(
        row["event"] == "training_evidence_replayed"
        for row in recursive_sink.events
    ) == 2
    assert sum(
        row["event"] == "counterfactual_evidence_replayed"
        for row in recursive_sink.events
    ) == 2


def test_train_only_candidate_selection_checks_all_roots_and_trigger_vocabulary(
    tmp_path: Path,
) -> None:
    invalid = _program_dict()
    invalid["id"] = "invalid-context-trigger"
    invalid["kind"] = "evaluator"
    invalid["trigger"] = {
        "all_of": [
            {"key": "task_instruction", "op": "contains", "value": "secret context"}
        ],
        "any_of": [],
        "none_of": [],
    }
    invalid["verifier"]["repair_on_failure"] = False
    selected = _program_dict()
    selected["id"] = "selected-runtime-trigger"
    broader = _program_dict()
    broader["id"] = "more-complex-runtime-trigger"
    broader["trigger"]["all_of"].append(
        {"key": "family", "op": "in", "value": [
            "organize-messy-files",
            "offer-letter-generator",
        ]}
    )
    harness, backend, model, archive, _, sink = _harness(
        tmp_path,
        proposal_rows=[invalid, selected, broader],
    )
    harness.validator.proposer = None

    result = harness.run_generation(
        train_item_ids=(
            "organize-messy-files-1",
            "organize-messy-files-2",
            "offer-letter-generator-1",
            "offer-letter-generator-2",
        ),
        validation_item_ids=("organize-messy-files-5", "offer-letter-generator-5"),
        trace_id="train-only-root-selection",
    )

    assert result.evolution is not None
    assert result.evolution.root_hypothesis_id == "selected-runtime-trigger"
    assert result.evolution.proposal_candidate_count == 3
    assert result.evolution.static_accepted_candidate_count == 2
    assert result.evolution.static_validation_node_count == 3
    assert result.evolution.static_validation_max_recursion_depth == 0
    assert result.evolution.repaired_candidate_count == 0
    assert archive.hypotheses["invalid-context-trigger"].status is HypothesisStatus.REJECTED
    assert archive.hypotheses["more-complex-runtime-trigger"].status is HypothesisStatus.SHADOW
    assert len(backend.calls) == 8
    assert len(model.requests) == 1
    trigger_contract = model.requests[0]["capabilities"]["runtime_trigger_contract"]
    assert "benchmark" in trigger_contract["allowed_feature_catalog"]
    assert "task_instruction" in trigger_contract["forbidden_context_only_keys"]
    validation_events = [
        row for row in sink.events if row["event"] == "hypothesis_validation_node_evaluated"
    ]
    assert any(
        any(
            check["check"] == "trigger_vocabulary" and not check["passed"]
            for check in row["payload"]["check_results"]
        )
        for row in validation_events
    )
    assert any(
        any(
            check["check"] == "runtime_candidate_kind" and not check["passed"]
            for check in row["payload"]["check_results"]
        )
        for row in validation_events
    )
    selection = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_training_candidate_selection_completed"
    )
    assert selection["payload"]["selection_uses_validation_outcomes"] is False


def test_family_out_compiler_targets_unseen_validation_families(tmp_path: Path) -> None:
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    items = adapter.discover()
    manifest = build_family_out_manifest(
        items,
        benchmark="skilllearnbench",
        seed="skilllearnbench-v2-family-out",
    )
    validation_by_family: dict[str, str] = {}
    for item_id in manifest.validation_ids:
        validation_by_family.setdefault(manifest.family_by_id[item_id], item_id)
    target_ids = tuple(validation_by_family.values())
    program = HypothesisProgram.from_dict(_program_dict(status="promoted"))

    compiled = SkillLearnProgramCompiler().compile(
        programs=(program,),
        items=items,
        split_manifest=manifest,
        output_root=tmp_path,
        target_item_ids=target_ids,
        target_split="validation",
    )

    assert compiled.family_count == len(validation_by_family)
    assert len(compiled.skill_paths) == len(target_ids)
    assert all(compiled.source_for(item_id) is not None for item_id in target_ids)
    assert all("items" in path.parts for path in compiled.skill_paths)


def test_subscription_trial_auth_is_ephemeral_and_not_passed_as_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"tokens":{"access_token":"fake-secret"}}\n', encoding="utf-8")
    monkeypatch.setenv("ASSUMPTION_V2_CODEX_AUTH_PATH", str(auth_path))
    monkeypatch.setenv("OPENAI_API_KEY", "api-key-must-not-enter-subscription-agent")
    agent = {
        "env": ["OPENAI_API_KEY"],
        "setup": "auth_json",
        "trajectory_env": {"CODEX_HOME": "/logs/agent"},
    }
    delegate = RecordingSubprocess()
    runner = ModuleType("fake_skilllearn_runner")
    runner.subprocess = delegate
    runner.get_agent = lambda agent_id: agent
    sink = MemoryEventSink()
    backend = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        provider_mode="codex_subscription",
        record_upstream=False,
        event_sink=sink,
    )

    with backend._provider_runtime(
        runner,
        agent_runtime_volume="assumption-v2-agent-test",
        trace_id="subscription-isolation-test",
    ):
        assert agent["env"] == []
        assert agent["setup"] is None
        assert agent["trajectory_env"]["CODEX_HOME"] == "/root/.codex"
        assert agent["trajectory_env"]["PATH"].startswith(
            "/opt/assumption-v2-agent/bin:"
        )
        tests_dir = tmp_path / "verifier-tests"
        tests_dir.mkdir()
        (tests_dir / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        runner.subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "trial",
                "-v",
                f"{tests_dir}:/tests:ro",
                "image",
                "sleep",
                "3600",
            ]
        )
        command = delegate.commands[-1]
        assert not any(":/tests" in value for value in command)
        assert command[-3:] == [
            "sh",
            "-c",
            "trap 'exit 0' TERM INT; while :; do sleep 86400; done",
        ]
        assert any(
            row["event"] == "skilllearn_fixed_container_lifetime_removed"
            for row in sink.events
        )
        mounts = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "-v"
        ]
        mount = next(value for value in mounts if value.endswith(":/root/.codex"))
        host_home = Path(mount.split(":", 1)[0])
        assert (
            "assumption-v2-agent-test:/opt/assumption-v2-agent:ro" in mounts
        )
        assert (host_home / "auth.json").is_file()
        assert "fake-secret" not in repr(command)
        assert "api-key-must-not-enter-subscription-agent" not in repr(command)
        runner.subprocess.run(
            ["docker", "exec", "trial", "sh", "-c", "npm install"],
            timeout=300,
        )
        assert delegate.kwargs[-1]["timeout"] == 300
        runner.subprocess.run(
            ["docker", "exec", "trial", "sh", "-c", "codex exec --help"],
            timeout=1800,
        )
        assert "timeout" not in delegate.kwargs[-1]
        assert not any(command[:2] == ["docker", "cp"] for command in delegate.commands)
        runner.subprocess.run(
            ["docker", "exec", "trial", "bash", "/tests/test.sh"],
            timeout=1800,
        )
        assert delegate.commands[-3] == [
            "docker",
            "exec",
            "trial",
            "mkdir",
            "-p",
            "/tests",
        ]
        assert delegate.commands[-2] == [
            "docker",
            "cp",
            f"{tests_dir.resolve()}/.",
            "trial:/tests",
        ]
        assert delegate.commands[-1] == [
            "docker",
            "exec",
            "trial",
            "bash",
            "/tests/test.sh",
        ]
        assert "timeout" not in delegate.kwargs[-1]
        assert sum(
            row["event"] == "skilllearn_fixed_wall_timeout_removed"
            for row in sink.events
        ) == 2
        assert any(
            row["event"] == "skilllearn_verifier_mount_withheld"
            for row in sink.events
        )
        assert any(
            row["event"] == "skilllearn_verifier_materialized_post_agent"
            for row in sink.events
        )
        delegate.agent_stdout = json.dumps(
            {
                "type": "turn.failed",
                "error": {"message": "You've hit your usage limit for the model."},
            }
        )
        with pytest.raises(RuntimeError, match="subscription_usage_limit"):
            runner.subprocess.run(
                ["docker", "exec", "trial", "sh", "-c", "codex exec --help"],
                timeout=1800,
            )
        assert any(
            row["event"] == "skilllearn_agent_terminal_error_detected"
            and row["payload"]["error_type"] == "subscription_usage_limit"
            for row in sink.events
        )

    assert not host_home.exists()
    assert agent["env"] == ["OPENAI_API_KEY"]
    assert agent["setup"] == "auth_json"
    assert runner.subprocess is delegate
    assert any(
        row["event"] == "skilllearn_ephemeral_auth_cleanup_completed"
        for row in sink.events
    )


def test_openai_compatible_trial_compiles_sanitized_codex_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "provider-secret-must-not-enter-command-or-events"
    monkeypatch.setenv("ASSUMPTION_V2_API_KEY", secret)
    monkeypatch.setenv("ASSUMPTION_V2_API_BASE", "https://ruoli.dev")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-be-replaced")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    agent = {
        "env": ["OPENAI_API_KEY"],
        "setup": "auth_json",
        "run": (
            "codex exec --dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check --json --model {model} -- "
            '"$(cat {instruction_file})"'
        ),
        "trajectory_env": {"CODEX_HOME": "/logs/agent"},
    }
    original_agent = dict(agent)
    original_agent["env"] = list(agent["env"])
    original_agent["trajectory_env"] = dict(agent["trajectory_env"])
    delegate = RecordingSubprocess()
    runner = ModuleType("fake_openai_compatible_runner")
    runner.subprocess = delegate
    runner.get_agent = lambda agent_id: agent
    sink = MemoryEventSink()
    backend = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        model="gpt-5.4-mini",
        provider_mode="openai_compatible",
        record_upstream=False,
        event_sink=sink,
    )

    with backend._provider_runtime(runner, trace_id="openai-provider-test"):
        assert agent["setup"] is None
        assert agent["env"] == ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
        assert os.environ["OPENAI_API_KEY"] == secret
        assert os.environ["OPENAI_BASE_URL"] == "https://ruoli.dev/v1"
        run_template = agent["run"]
        assert "--ignore-user-config" in run_template
        assert "--ephemeral" in run_template
        assert 'model_provider="assumption_v2_openai_compatible"' in run_template
        assert "https://ruoli.dev/v1" in run_template
        assert "wire_api" in run_template and "responses" in run_template
        assert "supports_websockets=false" in run_template
        assert "requires_openai_auth=false" in run_template
        assert "api.openai.com" not in run_template
        assert secret not in run_template
        prepared = next(
            row
            for row in sink.events
            if row["event"] == "skilllearn_openai_compatible_provider_prepared"
        )
        assert prepared["payload"]["config_version"] == (
            "codex_custom_responses_provider_v1"
        )
        assert prepared["payload"]["endpoint_origin"] == "https://ruoli.dev"
        assert prepared["payload"]["wire_api"] == "responses"
        assert prepared["payload"]["secret_value_persisted"] is False
        assert secret not in repr(prepared)
        delegate.agent_stdout = json.dumps(
            {
                "type": "turn.failed",
                "error": {"message": "You've hit your usage limit for the model."},
            }
        )
        with pytest.raises(RuntimeError, match="provider_usage_limit"):
            runner.subprocess.run(
                ["docker", "exec", "trial", "sh", "-c", "codex exec --help"],
                timeout=1800,
            )
        assert any(
            row["event"] == "skilllearn_agent_terminal_error_detected"
            and row["payload"]["error_type"] == "provider_usage_limit"
            for row in sink.events
        )

    assert agent == original_agent
    assert runner.subprocess is delegate


def test_openai_compatible_terminal_errors_use_provider_labels() -> None:
    backend = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        model="gpt-5.4-mini",
        provider_mode="openai_compatible",
        record_upstream=False,
    )
    request = SkillLearnTrialRequest(
        item_id="item",
        family="family",
        split=SplitName.VALIDATION,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="epoch",
        pair_id="pair",
        repeat=0,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash="manifest",
    )
    result = {
        "passed": False,
        "reward": 0.0,
        "agent_stdout": json.dumps(
            {
                "type": "turn.failed",
                "error": {"message": "Incorrect API key provided"},
            }
        ),
    }

    observation = backend._sanitize_result(
        request,
        result=result,
        return_code=1,
        duration_seconds=1.0,
    )

    assert observation.error_type == "provider_authentication_failed"
    assert observation.valid is False
    assert backend.provider_circuit.open(observation.error_type) is True


@pytest.mark.parametrize(
    ("result", "expected_error"),
    [
        ({"agent_timed_out": True, "verifier_exit": 0}, "agent_timeout"),
        ({"agent_timed_out": False, "verifier_exit": -1}, "verifier_timeout"),
        (
            {
                "agent_timed_out": False,
                "verifier_exit": 0,
                "agent_stdout": json.dumps(
                    {
                        "type": "error",
                        "message": "You've hit your usage limit for the model.",
                    }
                ),
            },
            "subscription_usage_limit",
        ),
    ],
)
def test_upstream_trial_timeouts_are_invalid_observations(
    result: Mapping[str, Any],
    expected_error: str,
) -> None:
    backend = SkillLearnSubprocessBackend(BENCH_ROOT, record_upstream=False)
    request = SkillLearnTrialRequest(
        item_id="item",
        family="family",
        split=SplitName.VALIDATION,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="epoch",
        pair_id="pair",
        repeat=0,
        agent_id="codex",
        model="gpt-5.3-codex-spark",
        max_steps=100,
        manifest_hash="manifest",
    )

    observation = backend._sanitize_result(
        request,
        result={"passed": False, "reward": 0.0, **result},
        return_code=0,
        duration_seconds=1800.0,
    )

    assert observation.error_type == expected_error
    assert observation.valid is False
    assert observation.metrics["evaluation_valid"] == 0.0


def test_open_provider_circuit_skips_model_execution() -> None:
    circuit = SkillLearnProviderCircuit()
    assert circuit.open("subscription_usage_limit") is True
    backend = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        record_upstream=False,
        provider_circuit=circuit,
    )
    request = SkillLearnTrialRequest(
        item_id="item",
        family="family",
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="epoch",
        pair_id="pair",
        repeat=0,
        agent_id="codex",
        model="gpt-5.3-codex-spark",
        max_steps=100,
        manifest_hash="manifest",
    )

    observation = backend.run(request, skill_source_dir=None, trace_id="circuit-open")

    assert observation.valid is False
    assert observation.error_type == "provider_circuit_open_subscription_usage_limit"


def test_ephemeral_auth_cleanup_retries_transient_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_home = tmp_path / "secret-home"
    secret_home.mkdir()
    (secret_home / "auth.json").write_text("secret", encoding="utf-8")
    real_rmtree = shutil.rmtree
    attempts = 0

    def flaky_rmtree(path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(39, "Directory not empty")
        return real_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)
    sink = MemoryEventSink()

    _cleanup_ephemeral_codex_home(
        secret_home,
        event_sink=sink,
        trace_id="cleanup-retry",
    )

    assert attempts == 2
    assert not secret_home.exists()
    assert sink.events[-1]["payload"]["attempt_count"] == 2


def test_loaded_runners_isolate_agent_registry_across_parallel_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"tokens":{"access_token":"fake-secret"}}\n', encoding="utf-8")
    monkeypatch.setenv("ASSUMPTION_V2_CODEX_AUTH_PATH", str(auth_path))
    first = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        provider_mode="codex_subscription",
        record_upstream=False,
    )
    second = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        provider_mode="codex_subscription",
        record_upstream=False,
    )
    first_runner = first._load_runner()
    second_runner = second._load_runner()
    assert first_runner.get_agent("codex") is not second_runner.get_agent("codex")

    both_entered = threading.Barrier(2)
    first_exited = threading.Event()
    errors: list[BaseException] = []

    def run_first() -> None:
        try:
            with first._provider_runtime(first_runner):
                both_entered.wait(timeout=5)
            first_exited.set()
        except BaseException as exc:
            errors.append(exc)
            first_exited.set()

    def run_second() -> None:
        try:
            with second._provider_runtime(second_runner):
                both_entered.wait(timeout=5)
                assert first_exited.wait(timeout=5)
                agent = second_runner.get_agent("codex")
                assert agent["env"] == []
                assert agent["setup"] is None
                assert agent["trajectory_env"]["CODEX_HOME"] == "/root/.codex"
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_first), threading.Thread(target=run_second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []


def test_prebuilt_cache_is_keyed_by_exact_non_oracle_environment(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    environment = benchmark / "tasks" / "family" / "item-1" / "environment"
    (benchmark / "core").mkdir(parents=True)
    (benchmark / "core" / "eval_runner.py").write_text("# frozen runner\n", encoding="utf-8")
    (environment / "skills" / "oracle").mkdir(parents=True)
    (environment / "Dockerfile").write_text(
        "FROM scratch\nCOPY skills/tool /root/tool\n",
        encoding="utf-8",
    )
    (environment / "payload.txt").write_text("version-one\n", encoding="utf-8")
    (environment / "skills" / "oracle" / "SKILL.md").write_text(
        "oracle secret\n",
        encoding="utf-8",
    )
    docker = FakeDockerSubprocess()
    runner = ModuleType("prebuilt_runner")
    runner.subprocess = docker
    runner.get_agent = lambda agent_id: {
        "runtime_deps": "RUN-DEPS",
        "install": "npm install -g @openai/codex",
    }
    build_index = 0

    def prepare(source: Path, skill_mode: str, skill_source_dir) -> Path:
        nonlocal build_index
        assert skill_mode == "no_skill"
        assert skill_source_dir is None
        build_index += 1
        build_root = tmp_path / f"build-{build_index}"
        build_env = build_root / "environment"
        shutil.copytree(source, build_env, ignore=shutil.ignore_patterns("skills"))
        (build_env / "skills").mkdir()
        return build_env

    runner._prepare_build_env = prepare
    runner._parse_skill_copies = lambda dockerfile: [
        ("skills/tool", "/root/tool")
    ]
    sink = MemoryEventSink()
    cache = SkillLearnPrebuiltImageCache(benchmark, event_sink=sink)

    first = cache.ensure(
        family="family",
        item_id="item-1",
        agent_id="codex",
        runner=runner,
        trace_id="prebuild-first",
    )
    second = cache.ensure(
        family="family",
        item_id="item-1",
        agent_id="codex",
        runner=runner,
        trace_id="prebuild-second",
    )

    assert first.reused is False
    assert second.reused is True
    assert first.cache_key == second.cache_key
    assert docker.skill_stubs_present == [True]
    assert sum(command[:2] == ["docker", "build"] for command in docker.commands) == 1

    (environment / "skills" / "oracle" / "SKILL.md").write_text(
        "changed oracle content\n",
        encoding="utf-8",
    )
    oracle_changed = SkillLearnPrebuiltImageCache(benchmark).ensure(
        family="family",
        item_id="item-1",
        agent_id="codex",
        runner=runner,
        trace_id="prebuild-oracle-change",
    )
    assert oracle_changed.cache_key == first.cache_key
    assert oracle_changed.reused is True

    (environment / "payload.txt").write_text("version-two\n", encoding="utf-8")
    payload_changed = SkillLearnPrebuiltImageCache(benchmark).ensure(
        family="family",
        item_id="item-1",
        agent_id="codex",
        runner=runner,
        trace_id="prebuild-payload-change",
    )
    assert payload_changed.cache_key != first.cache_key
    assert payload_changed.reused is False
    assert any(row["event"] == "skilllearn_prebuilt_image_built" for row in sink.events)


def test_training_parallelism_preserves_manifest_order(tmp_path: Path) -> None:
    backend = ConcurrentFakeBackend()
    harness, _, _, _, _, _ = _harness(
        tmp_path,
        backend_override=backend,
        parallel_workers=2,
    )
    train_ids = (
        "organize-messy-files-1",
        "organize-messy-files-2",
        "offer-letter-generator-1",
        "offer-letter-generator-2",
    )

    observations = harness.collect_training_observations(
        train_item_ids=train_ids,
        trace_id="parallel-training",
    )

    assert tuple(row.request.item_id for row in observations) == train_ids
    assert backend.maximum_active == 2


def _harness(
    tmp_path: Path,
    *,
    invalid_candidate_item: str | None = None,
    invalid_training_item: str | None = None,
    backend_override=None,
    parallel_workers: int = 1,
    invalid_trial_max_attempts: int = 1,
    proposal_rows: list[dict[str, Any]] | None = None,
):
    sink = MemoryEventSink()
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    manifest = build_instance_holdout_manifest(
        adapter.discover(),
        benchmark="skilllearnbench",
        seed="skilllearnbench-v2-instance-holdout",
    )
    guard = SplitAccessGuard(manifest, event_sink=sink)
    model = QueueProposalModel(
        [{"hypotheses": proposal_rows or [_program_dict()]}]
    )
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    validator = RecursiveValidationEngine(
        [
            SchemaCheck(),
            RuntimeCandidateKindCheck(),
            TriggerVocabularyCheck(),
            TrainingSupportCheck(min_support=2),
            RuntimeActionCheck(),
            EvaluatorEpochCheck(),
        ],
        proposer=proposer,
        event_sink=sink,
    )
    backend = backend_override or FakeSkillLearnBackend(
        invalid_candidate_item=invalid_candidate_item,
        invalid_training_item=invalid_training_item,
    )
    archive = PolicyArchive(event_sink=sink)
    harness = SkillLearnEvolutionHarness(
        adapter=adapter,
        manifest=manifest,
        guard=guard,
        backend=backend,
        proposer=proposer,
        validator=validator,
        promotion_gate=PromotionGate(
            PromotionGateSpec(
                minimum_pairs=2,
                confidence=0.9,
                minimum_net_gain_count=1,
                minimum_activation_rate=1.0,
            ),
            event_sink=sink,
        ),
        archive=archive,
        evaluator_epoch="skilllearn-eval-epoch-0",
        output_root=tmp_path / "compiled",
        parallel_workers=parallel_workers,
        invalid_trial_max_attempts=invalid_trial_max_attempts,
        event_sink=sink,
    )
    return harness, backend, model, archive, guard, sink


def _program_dict(*, status: str = "candidate") -> dict[str, Any]:
    return {
        "id": "hyp-skilllearn-explicit-completion",
        "kind": "policy",
        "statement": "Skill-dependent tasks benefit from an explicit procedure and completion audit.",
        "trigger": {
            "all_of": [
                {"key": "benchmark", "op": "eq", "value": "skilllearnbench"},
            ]
        },
        "anti_trigger": {
            "any_of": [
                {
                    "key": "has_container_environment",
                    "op": "eq",
                    "value": False,
                }
            ]
        },
        "action_graph": [
            {
                "id": "enable-skill",
                "operation": "enable_lane",
                "target": "skilllearn_challenger",
            },
            {
                "id": "execute",
                "operation": "execute_step",
                "target": "task_procedure",
                "value": "Translate every explicit task requirement into an ordered execution checklist.",
                "depends_on": ["enable-skill"],
            },
            {
                "id": "audit",
                "operation": "check_condition",
                "target": "all_explicit_requirements_satisfied",
                "value": "Audit every requested output and constraint before completion.",
                "depends_on": ["execute"],
            },
        ],
        "expected_effect": {
            "metric": "task_success",
            "minimum_delta": 0.1,
            "maximum_harm_rate": 0.0,
            "maximum_cost_ratio": 1.1,
        },
        "verifier": {
            "checks": ["schema", "training_support", "runtime_action", "paired_validation"],
            "required_evidence": ["policy_off_outcome", "policy_on_outcome"],
            "anchor_id": "skilllearn_external_task_verifier",
            "repair_on_failure": True,
            "max_repair_depth": 2,
        },
        "evaluator_epoch": "skilllearn-eval-epoch-0",
        "fallback": "preserve_baseline",
        "status": status,
    }


def _contains_forbidden_answer_key(value: Any) -> bool:
    forbidden = {"gold", "gold_label", "correct_answer", "_answer"}
    if isinstance(value, Mapping):
        return bool(forbidden & set(value)) or any(
            _contains_forbidden_answer_key(child) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_answer_key(child) for child in value)
    return False
