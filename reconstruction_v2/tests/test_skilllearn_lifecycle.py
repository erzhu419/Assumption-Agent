from __future__ import annotations

import hashlib
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
    BaselineArmEvidenceReplayCache,
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
from assumption_agent.evaluation import (
    CANDIDATE_MAY_ONLY_TIGHTEN,
    PROSPECTIVE_ABSTENTION_PAIRED_GUARD,
    PromotionGate,
    PromotionGateSpec,
)
from assumption_agent.evolution import (
    CANDIDATE_BUNDLE_POLICY_VERSION,
    COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSION,
    COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION,
    CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
    PROPOSAL_FORMATION_POLICY_VERSION,
    PROPOSAL_FORMATION_POLICY_V2,
    PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
    TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
    CounterfactualEvidenceReplayCache,
    EvolutionKernel,
    _allowlisted_profile_primitives,
    _training_candidate_metrics,
    _training_candidate_score,
    _training_family_coverage_target,
)
from assumption_agent.events import MemoryEventSink
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
    CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
    INVALID_TRIAL_RETRY_POLICY_VERSION,
    LEGACY_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
    MODEL_INFERENCE_CONCURRENCY_POLICY_VERSION,
    SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
    TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
    SkillLearnModelInferenceLimiter,
    SkillLearnPrebuiltImage,
    SkillLearnSubprocessBackend,
    _ContainerNetworkBudgetMonitor,
    _classify_training_failure,
    _DockerVerifierIsolationSubprocessProxy,
    _extract_train_action_trace_profile,
    _inspect_codex_tool_policy,
    _inspect_verifier_execution_receipt,
    _parse_docker_byte_size,
    _parse_docker_net_io,
    _run_invalid_only_trial,
    _fairness_fingerprint,
    _provider_fingerprint,
)
from assumption_agent.benchmarks.codex_execution_policy import (
    LEGACY_CODEX_AGENT_EXECUTION_POLICY,
    LOW_REASONING_LOCAL_COMPACTION_POLICY,
    MODEL_ONLY_ACTION_BUDGET_POLICY,
)
from assumption_agent.benchmarks.docker_egress import DockerEgressPolicy
from assumption_agent.benchmarks.offline_verifier import (
    COMMON_PY38_VERIFIER_PROFILE,
    OFFLINE_VERIFIER_MOUNT,
    POSTER_VERIFIER_PROFILE,
    WEIGHTED_GDP_VERIFIER_PROFILE,
    OfflineVerifierRuntime,
)
from assumption_agent.benchmarks.skilllearn_experiment import (
    _advance_arm,
    _run_paired_arms,
)
from assumption_agent.benchmarks.paper_protocol import (
    COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION,
)
from assumption_agent.benchmarks.skilllearn_compiler import (
    NO_SKILL_TREATMENT_HASH,
    SKILL_ACTION_LOWERING_VERSION,
    skilllearn_program_set_treatment_hash,
    skilllearn_program_treatment_hash,
)
from assumption_agent.models import (
    HypothesisProgram,
    HypothesisStatus,
    ResidualExample,
    SplitName,
    stable_hash,
)
from assumption_agent.proposer import (
    PROPOSAL_DIVERSITY_POLICY_VERSION,
    REPAIR_REQUEST_SCOPE_POLICY_VERSION,
    StructuredHypothesisProposer,
    TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY,
    TRAIN_ACTION_DESIGN_POLICY_VERSION,
    train_action_quality_contract,
)
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
    ValidationContext,
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
        self.requests: list[Any] = []
        self.request_hashes: list[str] = []

    def run(self, request, *, skill_source_dir, trace_id):
        has_skill = skill_source_dir is not None and skill_source_dir.is_dir()
        self.calls.append((request.item_id, request.variant.value, has_skill))
        self.requests.append(request)
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


class ValidationBaselineSequenceBackend(FakeSkillLearnBackend):
    """Return a frozen sequence of validation policy-off error types."""

    def __init__(self, errors: tuple[str | None, ...]) -> None:
        super().__init__()
        self.errors = errors
        self.validation_baseline_call_count = 0

    def run(self, request, *, skill_source_dir, trace_id):
        observation = super().run(
            request,
            skill_source_dir=skill_source_dir,
            trace_id=trace_id,
        )
        if not (
            request.split is SplitName.VALIDATION
            and request.variant is TrialVariant.POLICY_OFF
        ):
            return observation
        index = self.validation_baseline_call_count
        self.validation_baseline_call_count += 1
        error_type = self.errors[index] if index < len(self.errors) else None
        if error_type is None:
            return observation
        metrics = dict(observation.metrics)
        metrics.update({"evaluation_valid": 0.0, "task_success": 0.0})
        return replace(
            observation,
            success=False,
            score=0.0,
            metrics=metrics,
            error_type=error_type,
        )


class SelectiveTrainingSuccessBackend(FakeSkillLearnBackend):
    def __init__(self, successful_training_items: set[str]) -> None:
        super().__init__()
        self.successful_training_items = set(successful_training_items)

    def run(self, request, *, skill_source_dir, trace_id):
        observation = super().run(
            request,
            skill_source_dir=skill_source_dir,
            trace_id=trace_id,
        )
        if (
            request.split is SplitName.TRAIN
            and request.variant is TrialVariant.POLICY_OFF
            and request.item_id in self.successful_training_items
        ):
            metrics = dict(observation.metrics)
            metrics.update(
                {
                    "task_success": 1.0,
                    "trajectory_key_point_recall": 1.0,
                    "evaluation_valid": 1.0,
                }
            )
            return replace(
                observation,
                success=True,
                score=1.0,
                metrics=metrics,
                error_type=None,
            )
        return observation


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


class FakeNetworkStatsSubprocess:
    def __init__(self, net_io: str) -> None:
        self.net_io = net_io
        self.commands: list[list[str]] = []

    def run(self, args, *positional, **kwargs):
        command = list(args)
        self.commands.append(command)
        if command[:2] == ["docker", "stats"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"NetIO": self.net_io}) + "\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class ConcurrentFakeBackend(FakeSkillLearnBackend):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0
        self.active_by_item: dict[str, int] = {}
        self.maximum_active_by_item: dict[str, int] = {}

    def run(self, request, *, skill_source_dir, trace_id):
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            item_id = request.item_id
            self.active_by_item[item_id] = self.active_by_item.get(item_id, 0) + 1
            self.maximum_active_by_item[item_id] = max(
                self.maximum_active_by_item.get(item_id, 0),
                self.active_by_item[item_id],
            )
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
                self.active_by_item[item_id] -= 1


class BlockingAgentSubprocess:
    def __init__(self, *, fail_first: bool = False) -> None:
        self._lock = threading.Lock()
        self.active_agents = 0
        self.maximum_active_agents = 0
        self.agent_entered = threading.Event()
        self.release_agents = threading.Event()
        self.verifier_entered = threading.Event()
        self.fail_first = fail_first
        self.agent_calls = 0

    def run(self, args, *positional, **kwargs):
        command = list(args)
        command_text = " ".join(str(value) for value in command)
        if "codex exec" in command_text:
            with self._lock:
                self.agent_calls += 1
                call_index = self.agent_calls
                self.active_agents += 1
                self.maximum_active_agents = max(
                    self.maximum_active_agents,
                    self.active_agents,
                )
            self.agent_entered.set()
            try:
                if self.fail_first and call_index == 1:
                    raise RuntimeError("synthetic agent failure")
                assert self.release_agents.wait(timeout=2.0)
            finally:
                with self._lock:
                    self.active_agents -= 1
        if "/tests/test.sh" in command_text:
            self.verifier_entered.set()
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_actionable_train_failure_feedback_removes_completion_check_bias() -> None:
    request = SkillLearnTrialRequest(
        item_id="financial-analysis-1",
        family="financial-analysis",
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="skilllearn-eval-test",
        pair_id="train",
        repeat=1,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash="manifest",
    )
    observation = SkillLearnTrialObservation(
        request=request,
        success=False,
        score=0.0,
        metrics={"task_success": 0.0, "evaluation_valid": 1.0},
        total_tokens=100,
        steps=10,
        duration_seconds=1.0,
        provider_fingerprint="provider",
        fairness_fingerprint="budget",
    )

    _, legacy_feedback = _classify_training_failure(observation)
    _, actionable_feedback = _classify_training_failure(
        observation,
        actionable_feedback=True,
    )

    assert ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION.endswith(
        "_v2"
    )
    assert "completion check" in legacy_feedback[-1]
    assert "concrete reusable corrective operator" in actionable_feedback[-1]
    assert "do not default to a generic completeness check" in (
        actionable_feedback[-1]
    )


def test_network_budget_parser_and_monitor_kill_over_limit() -> None:
    assert _parse_docker_byte_size("1.5MB") == 1_500_000
    assert _parse_docker_byte_size("2MiB") == 2 * 1024 * 1024
    assert _parse_docker_net_io("20MiB / 13MiB") == (
        20 * 1024 * 1024,
        13 * 1024 * 1024,
    )
    delegate = FakeNetworkStatsSubprocess("20MiB / 13MiB")
    sink = MemoryEventSink()
    monitor = _ContainerNetworkBudgetMonitor(
        delegate,
        container_name="trial",
        byte_limit=32 * 1024 * 1024,
        event_sink=sink,
        trace_id="network-budget",
    )

    monitor._sample()

    assert monitor.exceeded is True
    assert ["docker", "kill", "trial"] in delegate.commands
    exceeded = next(
        row
        for row in sink.events
        if row["event"] == "skilllearn_trial_network_budget_exceeded"
    )
    assert exceeded["payload"]["observed_bytes"] == 33 * 1024 * 1024


def test_network_budget_violation_is_never_retried() -> None:
    request = SkillLearnTrialRequest(
        item_id="anthropic-poster-design-4",
        family="anthropic-poster-design",
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="epoch-offline",
        pair_id="poster-network-budget",
        repeat=0,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=20,
        manifest_hash="manifest-offline",
    )
    calls: list[int] = []

    def run_once(attempt: int) -> SkillLearnTrialObservation:
        calls.append(attempt)
        return SkillLearnTrialObservation(
            request=request,
            success=False,
            score=0.0,
            metrics={"evaluation_valid": 0.0},
            total_tokens=0,
            steps=0,
            duration_seconds=1.0,
            provider_fingerprint="provider-fixed",
            fairness_fingerprint="budget-fixed",
            error_type="trial_network_byte_limit_exceeded",
        )

    sink = MemoryEventSink()
    observation = _run_invalid_only_trial(
        request=request,
        run_once=run_once,
        maximum_attempts=3,
        backoff_seconds=0.0,
        retry_semaphore=threading.Semaphore(1),
        event_sink=sink,
        trace_id="network-budget-no-retry",
    )

    assert observation.error_type == "trial_network_byte_limit_exceeded"
    assert calls == [1]
    suppressed = next(
        row
        for row in sink.events
        if row["event"] == "skilllearn_invalid_trial_retry_suppressed"
    )
    assert suppressed["payload"]["suppression_reason"] == "hard_network_budget_exceeded"
    assert suppressed["payload"]["hard_budget_violation"] is True


def test_verifier_receipt_requires_a_real_ctrf_test_run(tmp_path: Path) -> None:
    test_script = tmp_path / "tests" / "test.sh"
    verifier_dir = tmp_path / "trial" / "verifier"
    test_script.parent.mkdir(parents=True)
    verifier_dir.mkdir(parents=True)
    test_script.write_text(
        "pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n",
        encoding="utf-8",
    )
    (verifier_dir / "reward.txt").write_text("0\n", encoding="utf-8")
    result = {"verifier_exit": 0, "reward": 0}

    missing = _inspect_verifier_execution_receipt(
        test_script=test_script,
        verifier_dir=verifier_dir,
        result=result,
    )

    assert missing.valid is False
    assert missing.error_type == "verifier_execution_ctrf_missing"

    (verifier_dir / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "tool": {"name": "pytest", "version": "8.4.1"},
                    "summary": {
                        "tests": 2,
                        "passed": 1,
                        "failed": 1,
                        "skipped": 0,
                        "pending": 0,
                        "other": 0,
                    },
                    "tests": [
                        {"name": "test_ok", "status": "passed"},
                        {"name": "test_bad", "status": "failed"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    complete = _inspect_verifier_execution_receipt(
        test_script=test_script,
        verifier_dir=verifier_dir,
        result=result,
    )

    assert complete.valid is True
    assert complete.reward == 0
    assert complete.test_count == 2
    assert complete.receipt_hash


def test_verifier_receipt_binds_the_executed_offline_profile(tmp_path: Path) -> None:
    test_script = tmp_path / "tests" / "test.sh"
    verifier_dir = tmp_path / "trial" / "verifier"
    test_script.parent.mkdir(parents=True)
    verifier_dir.mkdir(parents=True)
    test_script.write_text(
        "python3 /tests/test_outputs.py\n",
        encoding="utf-8",
    )
    (verifier_dir / "reward.txt").write_text("0\n", encoding="utf-8")
    result = {"verifier_exit": 0, "reward": 0}

    missing = _inspect_verifier_execution_receipt(
        test_script=test_script,
        verifier_dir=verifier_dir,
        result=result,
        offline_verifier_profile=COMMON_PY38_VERIFIER_PROFILE,
    )
    assert missing.valid is False
    assert missing.error_type == "verifier_execution_ctrf_missing"

    (verifier_dir / "ctrf.json").write_text("{", encoding="utf-8")
    malformed = _inspect_verifier_execution_receipt(
        test_script=test_script,
        verifier_dir=verifier_dir,
        result=result,
        offline_verifier_profile=COMMON_PY38_VERIFIER_PROFILE,
    )
    assert malformed.valid is False
    assert malformed.error_type == "verifier_execution_ctrf_malformed"

    (verifier_dir / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "summary": {
                        "tests": 2,
                        "passed": 1,
                        "failed": 1,
                        "skipped": 0,
                        "pending": 0,
                        "other": 0,
                    },
                    "tests": [
                        {"name": "test_ok", "status": "passed"},
                        {"name": "test_bad", "status": "failed"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    receipt = _inspect_verifier_execution_receipt(
        test_script=test_script,
        verifier_dir=verifier_dir,
        result=result,
        offline_verifier_profile=COMMON_PY38_VERIFIER_PROFILE,
    )
    changed_command_profile = replace(
        COMMON_PY38_VERIFIER_PROFILE,
        artifact_command="printf audited > /logs/verifier/audited.txt",
    )
    changed_command_receipt = _inspect_verifier_execution_receipt(
        test_script=test_script,
        verifier_dir=verifier_dir,
        result=result,
        offline_verifier_profile=changed_command_profile,
    )

    assert receipt.valid is True
    assert receipt.evidence_kind == "pytest_ctrf"
    assert receipt.reward == 0
    assert receipt.test_count == 2
    assert changed_command_profile.verifier_command != (
        COMMON_PY38_VERIFIER_PROFILE.verifier_command
    )
    assert changed_command_receipt.valid is True
    assert changed_command_receipt.receipt_hash != receipt.receipt_hash


def test_trial_audit_binds_the_runtime_profile_not_a_catalog_lookup(
    tmp_path: Path,
) -> None:
    family = "temperature-simulation"
    item_id = "temperature-simulation-3"
    benchmark_root = tmp_path / "benchmark"
    trials_dir = tmp_path / "trials"
    test_script = (
        benchmark_root / "tasks" / family / item_id / "tests" / "test.sh"
    )
    test_script.parent.mkdir(parents=True)
    test_script.write_text("python3 /tests/test_outputs.py\n", encoding="utf-8")
    request = SkillLearnTrialRequest(
        item_id=item_id,
        family=family,
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="epoch-offline",
        pair_id="runtime-profile",
        repeat=0,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash="manifest-offline",
    )
    trial_dir = (
        trials_dir / "no_skill" / family / item_id / request.trial_id
    )
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (trial_dir / "agent").mkdir()
    (trial_dir / "agent" / "codex.txt").write_text("", encoding="utf-8")
    (verifier_dir / "reward.txt").write_text("0\n", encoding="utf-8")
    (verifier_dir / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "summary": {
                        "tests": 1,
                        "passed": 0,
                        "failed": 1,
                        "skipped": 0,
                        "pending": 0,
                        "other": 0,
                    },
                    "tests": [{"name": "test_failure", "status": "failed"}],
                }
            }
        ),
        encoding="utf-8",
    )
    result = {"verifier_exit": 0, "reward": 0}
    runtime_profile = replace(
        COMMON_PY38_VERIFIER_PROFILE,
        profile_id="runtime-injected-py38-v1",
    )
    backend = SkillLearnSubprocessBackend(
        benchmark_root,
        trials_dir=trials_dir,
        provider_mode="openai_compatible",
        event_sink=MemoryEventSink(),
    )

    audited = backend._audit_trial_artifacts(
        runner=SimpleNamespace(TRIALS_DIR=str(trials_dir)),
        request=request,
        skill_config="no_skill",
        result=result,
        offline_verifier_profile=runtime_profile,
        trace_id="runtime-profile-audit",
    )
    runtime_receipt = _inspect_verifier_execution_receipt(
        test_script=test_script,
        verifier_dir=verifier_dir,
        result=result,
        offline_verifier_profile=runtime_profile,
    )
    catalog_receipt = _inspect_verifier_execution_receipt(
        test_script=test_script,
        verifier_dir=verifier_dir,
        result=result,
        offline_verifier_profile=COMMON_PY38_VERIFIER_PROFILE,
    )

    assert audited["verifier_receipt_valid"] is True
    assert audited["verifier_receipt_test_count"] == 1
    assert audited["verifier_receipt_hash"] == runtime_receipt.receipt_hash
    assert audited["verifier_receipt_hash"] != catalog_receipt.receipt_hash


def test_verifier_receipt_structurally_binds_semantic_prelude(tmp_path: Path) -> None:
    test_script = tmp_path / "tests" / "test.sh"
    verifier_dir = tmp_path / "trial" / "verifier"
    test_script.parent.mkdir(parents=True)
    verifier_dir.mkdir(parents=True)
    test_script.write_text(
        "pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n",
        encoding="utf-8",
    )
    (verifier_dir / "reward.txt").write_text("0\n", encoding="utf-8")
    (verifier_dir / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "summary": {
                        "tests": 1,
                        "passed": 0,
                        "failed": 1,
                        "skipped": 0,
                        "pending": 0,
                        "other": 0,
                    },
                    "tests": [{"name": "test_failure", "status": "failed"}],
                }
            }
        ),
        encoding="utf-8",
    )
    result = {"verifier_exit": 0, "reward": 0}

    missing = _inspect_verifier_execution_receipt(
        test_script=test_script,
        verifier_dir=verifier_dir,
        result=result,
        offline_verifier_profile=WEIGHTED_GDP_VERIFIER_PROFILE,
    )

    assert missing.valid is False
    assert missing.error_type == "semantic_prelude_receipt_missing"

    (verifier_dir / "semantic_prelude.json").write_text(
        json.dumps(
            {
                "prelude_id": "weighted_gdp_ssconvert_v1",
                "exit_code": 0,
            }
        ),
        encoding="utf-8",
    )
    (verifier_dir / "semantic_prelude_details.txt").write_text(
        "tool=ssconvert\ncommand_exit=0\nsheet_count=3\n",
        encoding="utf-8",
    )
    succeeded = _inspect_verifier_execution_receipt(
        test_script=test_script,
        verifier_dir=verifier_dir,
        result=result,
        offline_verifier_profile=WEIGHTED_GDP_VERIFIER_PROFILE,
    )

    assert succeeded.valid is True
    assert succeeded.semantic_prelude_valid is True
    assert succeeded.semantic_prelude_succeeded is True
    assert succeeded.semantic_prelude_exit_code == 0
    assert succeeded.semantic_prelude_details["sheet_count"] == "3"

    (verifier_dir / "semantic_prelude.json").write_text(
        json.dumps(
            {
                "prelude_id": "weighted_gdp_ssconvert_v1",
                "exit_code": 3,
            }
        ),
        encoding="utf-8",
    )
    (verifier_dir / "semantic_prelude_details.txt").write_text(
        "tool=ssconvert\ncommand_exit=3\nsheet_count=0\n",
        encoding="utf-8",
    )
    task_failure = _inspect_verifier_execution_receipt(
        test_script=test_script,
        verifier_dir=verifier_dir,
        result=result,
        offline_verifier_profile=WEIGHTED_GDP_VERIFIER_PROFILE,
    )

    assert task_failure.valid is True
    assert task_failure.error_type is None
    assert task_failure.semantic_prelude_valid is True
    assert task_failure.semantic_prelude_succeeded is False
    assert task_failure.reward == 0


def test_codex_tool_audit_rejects_remote_tools_and_runtime_installs(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "codex.txt"
    trace.write_text(
        "Reading additional input from stdin...\n"
        + json.dumps(
            {
                "type": "item.completed",
                "item": {"id": "search-1", "type": "web_search"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "command-1",
                    "type": "command_execution",
                    "command": "/bin/bash -lc 'python3 -m pip install Pillow'",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = _inspect_codex_tool_policy(trace)

    assert audit.valid is False
    assert audit.error_type == "model_remote_tool_policy_violation"
    assert audit.remote_tool_call_count == 1
    assert audit.runtime_install_command_count == 1
    assert audit.trace_hash


def test_train_action_trace_profile_keeps_actions_without_outputs_or_secrets(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "codex.txt"
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "message",
                            "type": "agent_message",
                            "text": "raw model prose must not enter the profile",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "relative-solution-command",
                            "type": "command_execution",
                            "command": "cat solution.txt",
                            "status": "completed",
                            "exit_code": 0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "command",
                            "type": "command_execution",
                            "command": (
                                "/bin/bash -lc 'trivy fs --skip-db-update "
                                "--cache-dir /root/.cache/trivy "
                                "/root/package-lock.json'"
                            ),
                            "aggregated_output": "hidden command output",
                            "status": "failed",
                            "exit_code": 1,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "secret-command",
                            "type": "command_execution",
                            "command": (
                                "trivy fs --api-key placeholder_sensitive_value "
                                "--header 'Authorization: Bearer "
                                "placeholder_bearer_value' "
                                "https://example.invalid/scan?token=hidden"
                            ),
                            "status": "failed",
                            "exit_code": 1,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "oracle-command",
                            "type": "command_execution",
                            "command": "cat /root/tests/oracle_solution.json",
                            "status": "completed",
                            "exit_code": 0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "file",
                            "type": "file_change",
                            "changes": [
                                {"path": "/root/security_audit.csv", "kind": "add"},
                                {"path": "/root/tests/oracle.json", "kind": "add"},
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profile = _extract_train_action_trace_profile(trace)

    assert profile["commands_observed"] == 1
    assert profile["failed_commands_observed"] == 1
    assert profile["changed_task_paths"] == ["/root/security_audit.csv"]
    assert profile["commands_returned"] == 1
    signature = profile["command_signatures"][0]
    assert signature["executable_basename"] == "trivy"
    assert signature["safe_flags"] == ["--cache-dir", "--skip-db-update"]
    assert signature["task_local_paths"] == [
        "/root/.cache/trivy",
        "/root/package-lock.json",
    ]
    assert signature["original_command_hash"]
    assert "command" not in signature
    serialized = json.dumps(profile, sort_keys=True)
    assert "trivy" in serialized
    assert "--skip-db-update" in serialized
    assert "hidden command output" not in serialized
    assert "raw model prose" not in serialized
    assert "placeholder_sensitive_value" not in serialized
    assert "placeholder_bearer_value" not in serialized
    assert "example.invalid" not in serialized
    assert "Authorization" not in serialized
    assert "oracle_solution" not in serialized
    assert "/root/tests" not in serialized


def test_train_action_trace_profile_normalizes_status_and_rejects_sensitive_paths(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "codex.txt"
    key_like_path = "/root/sk-" + ("placeholder" * 4)
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "jq -r . /root/input.json",
                            "status": "SENSITIVE_STATUS_VALUE_123",
                            "exit_code": 987654321,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "cat /root/api_key/SENSITIVE_PATH_VALUE_456",
                            "status": "completed",
                            "exit_code": 0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "file_change",
                            "changes": [
                                {
                                    "path": "/root/access_token/SENSITIVE_CHANGE_789",
                                    "kind": "add",
                                },
                                {"path": key_like_path, "kind": "add"},
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profile = _extract_train_action_trace_profile(trace)

    assert profile["commands_observed"] == 1
    assert profile["command_signatures"][0]["status"] == "unknown"
    assert profile["command_signatures"][0]["exit_code"] == 1
    assert profile["changed_task_paths"] == []
    serialized = json.dumps(profile, sort_keys=True)
    assert "SENSITIVE_STATUS_VALUE_123" not in serialized
    assert "SENSITIVE_PATH_VALUE_456" not in serialized
    assert "SENSITIVE_CHANGE_789" not in serialized
    assert key_like_path not in serialized


def test_train_action_trace_profile_rejects_trace_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "trivy fs /root/package-lock.json",
                    "status": "completed",
                    "exit_code": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trace = tmp_path / "codex.txt"
    trace.symlink_to(outside)

    assert _extract_train_action_trace_profile(trace) == {}


def test_unlocalized_online_verifier_family_is_blocked_before_model_start() -> None:
    sink = MemoryEventSink()
    backend = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        model="gpt-5.4-mini",
        provider_mode="openai_compatible",
        event_sink=sink,
    )
    request = SkillLearnTrialRequest(
        item_id="nlp-paper-reproduction-2",
        family="nlp-paper-reproduction",
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="epoch-offline",
        pair_id="unlocalized-family",
        repeat=0,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash="manifest-offline",
    )

    observation = backend.run(request, skill_source_dir=None, trace_id="offline-block")

    assert observation.valid is False
    assert observation.error_type == "offline_verifier_profile_missing"
    assert any(
        row["event"] == "skilllearn_trial_blocked_missing_offline_verifier_profile"
        and row["payload"]["model_container_started"] is False
        and row["payload"]["runtime_network_attempted"] is False
        for row in sink.events
    )
    assert not any(row["event"] == "skilllearn_trial_started" for row in sink.events)


def test_inactive_druid_verifier_is_blocked_before_model_start() -> None:
    sink = MemoryEventSink()
    backend = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        model="gpt-5.4-mini",
        provider_mode="openai_compatible",
        event_sink=sink,
    )
    request = SkillLearnTrialRequest(
        item_id="fix-security-bug-1",
        family="fix-security-bug",
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="epoch-offline",
        pair_id="inactive-druid-family",
        repeat=0,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash="manifest-offline",
    )

    observation = backend.run(request, skill_source_dir=None, trace_id="druid-block")

    assert observation.valid is False
    assert observation.error_type == "offline_verifier_profile_inactive"
    blocked = next(
        row
        for row in sink.events
        if row["event"]
        == "skilllearn_trial_blocked_inactive_offline_verifier_profile"
    )
    assert blocked["payload"]["activation_blocker"] == (
        "druid_maven_cache_incomplete"
    )
    assert blocked["payload"]["catalog_profile_id"] == (
        "druid-security-py312-v1"
    )
    assert blocked["payload"]["model_container_started"] is False
    assert blocked["payload"]["runtime_network_attempted"] is False
    assert not any(row["event"] == "skilllearn_trial_started" for row in sink.events)


def test_trial_prewarm_binds_declared_offline_verifier_runtime() -> None:
    image = SkillLearnPrebuiltImage(
        tag="assumption-v2-item:poster",
        cache_key="image-key",
        environment_hash="environment-hash",
        image_id="sha256:" + "a" * 64,
        agent_runtime_key="agent-runtime-key",
        agent_runtime_volume="agent-runtime-volume",
        agent_runtime_version="codex-cli 0.144.1",
        reused=True,
    )

    class ImageCache:
        def ensure(self, **kwargs):
            return image

    class VerifierCache:
        def __init__(self) -> None:
            self.calls = []

        def ensure(self, **kwargs):
            self.calls.append(kwargs)
            return OfflineVerifierRuntime(
                profile=kwargs["profile"],
                runtime_key="verifier-runtime-key",
                volume_name="verifier-runtime-volume",
                base_image_id=kwargs["base_image_id"],
                reused=True,
            )

    verifier_cache = VerifierCache()
    backend = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        record_upstream=False,
        prebuilt_cache=ImageCache(),
        offline_verifier_cache=verifier_cache,
    )
    runner = ModuleType("prewarm_runner")
    runner.subprocess = RecordingSubprocess()
    backend._runner_module = runner

    warmed_image, runtime = backend.prewarm_trial_environment(
        family="anthropic-poster-design",
        item_id="anthropic-poster-design-1",
        trace_id="prewarm-poster",
    )

    assert warmed_image is image
    assert runtime is not None
    assert runtime.profile is POSTER_VERIFIER_PROFILE
    assert verifier_cache.calls[0]["base_image_tag"] == image.tag
    assert verifier_cache.calls[0]["base_image_id"] == image.image_id


def test_trial_prewarm_uses_native_verifier_only_for_network_free_script() -> None:
    image = SkillLearnPrebuiltImage(
        tag="assumption-v2-item:dbscan",
        cache_key="image-key",
        environment_hash="environment-hash",
        image_id="sha256:" + "b" * 64,
        agent_runtime_key="agent-runtime-key",
        agent_runtime_volume="agent-runtime-volume",
        agent_runtime_version="codex-cli 0.144.1",
        reused=True,
    )

    class ImageCache:
        def ensure(self, **kwargs):
            return image

    backend = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        record_upstream=False,
        prebuilt_cache=ImageCache(),
    )
    runner = ModuleType("native_prewarm_runner")
    runner.subprocess = RecordingSubprocess()
    backend._runner_module = runner

    warmed_image, runtime = backend.prewarm_trial_environment(
        family="dbscan-parameter-tuning",
        item_id="dbscan-parameter-tuning-1",
        trace_id="prewarm-dbscan",
    )

    assert warmed_image is image
    assert runtime is None


def test_prebuilt_cache_fails_closed_before_any_online_install(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    environment = benchmark / "tasks" / "family" / "item-1" / "environment"
    (benchmark / "core").mkdir(parents=True)
    (benchmark / "core" / "eval_runner.py").write_text("# frozen runner\n")
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM scratch\n")
    docker = FakeDockerSubprocess()
    runner = ModuleType("cache_only_runner")
    runner.subprocess = docker
    runner.get_agent = lambda agent_id: {
        "runtime_deps": "RUN-DEPS",
        "install": "npm install -g @openai/codex",
    }

    with pytest.raises(RuntimeError, match="shared_agent_runtime_cache_missing_offline"):
        SkillLearnPrebuiltImageCache(benchmark).ensure(
            family="family",
            item_id="item-1",
            agent_id="codex",
            runner=runner,
            trace_id="cache-only",
        )

    assert not any(command[:2] == ["docker", "build"] for command in docker.commands)
    assert not any(command[:3] == ["docker", "volume", "create"] for command in docker.commands)
    assert not any(command[:2] == ["docker", "run"] for command in docker.commands)


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
    assert result.evolution.promotion_decision.summary.baseline_preserved_count == 0
    assert (
        result.evolution.promotion_decision.promotion_contract[
            "baseline_safety_policy"
        ]
        == PROSPECTIVE_ABSTENTION_PAIRED_GUARD
    )
    assert result.to_dict()["promotion_decision"] == (
        result.evolution.promotion_decision.to_dict()
    )
    assert "selected_candidate_hypothesis_ids" not in result.to_dict()
    assert len(result.residuals) == 4
    assert all(row.context["task_instruction"] for row in result.residuals)
    assert all(row.split.value == "train" for row in result.residuals)
    assert guard.test_accessed is False
    assert len(backend.calls) == 8
    assert archive.incumbent_id == result.evolution.archive_node.id
    promoted = archive.hypotheses[result.evolution.accepted_hypothesis_id]
    assert promoted.status is HypothesisStatus.PROMOTED
    assert result.to_dict()["evaluated_candidate_treatment_hash"] == (
        skilllearn_program_treatment_hash(promoted)
    )
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
    assert all(pair.candidate.baseline_preserved is False for pair in pairs)


def test_skilllearn_nonactivation_aliases_observed_baseline(tmp_path: Path) -> None:
    harness, backend, _, _, _, _ = _harness(tmp_path)
    payload = _program_dict()
    payload["trigger"] = {
        "all_of": [
            {"key": "family", "op": "eq", "value": "never-matched-family"}
        ],
        "any_of": [],
        "none_of": [],
    }
    program = HypothesisProgram.from_dict(payload)

    pairs = harness.counterfactual_runner.run(
        harness.tasks(("organize-messy-files-5",)),
        program=program,
        split=SplitName.VALIDATION,
        trace_id="nonactivation-baseline-alias",
    )

    assert len(backend.calls) == 1
    assert pairs[0].candidate.action_activated is False
    assert pairs[0].candidate.baseline_preserved is True
    assert pairs[0].candidate.selected_result.lane == "skilllearn_incumbent"


def test_skilllearn_bundle_routes_complementary_members_in_one_pair_per_item(
    tmp_path: Path,
) -> None:
    harness, backend, _, _, _, sink = _harness(tmp_path)
    organize = _family_trigger_program(
        "bundle-organize",
        ("organize-messy-files",),
    )
    offer = _family_trigger_program(
        "bundle-offer",
        ("offer-letter-generator",),
    )
    tasks = harness.tasks(
        ("organize-messy-files-5", "offer-letter-generator-5")
    )

    pairs = harness.counterfactual_runner.run_bundle(
        tasks,
        programs=(offer, organize),
        split=SplitName.VALIDATION,
        trace_id="complementary-bundle",
    )

    assert len(pairs) == 2
    assert pairs[0].candidate.activated_hypothesis_ids == ("bundle-organize",)
    assert pairs[1].candidate.activated_hypothesis_ids == ("bundle-offer",)
    assert all(pair.candidate.action_activated for pair in pairs)
    assert len(backend.calls) == 4
    assert sum(row[1] == "policy_on" for row in backend.calls) == 2
    expected_delta_hash = skilllearn_program_set_treatment_hash((organize, offer))
    events = [
        row
        for row in sink.events
        if row["event"] == "skilllearn_counterfactual_pair_completed"
    ]
    assert [
        row["payload"]["matched_candidate_hypothesis_ids"]
        for row in events
    ] == [["bundle-organize"], ["bundle-offer"]]
    assert all(
        row["payload"]["selected_candidate_hypothesis_ids"]
        == ["bundle-offer", "bundle-organize"]
        for row in events
    )
    assert all(
        row["payload"]["candidate_delta_program_set_hash"]
        == expected_delta_hash
        for row in events
    )
    on_requests = [
        request
        for request in backend.requests
        if request.variant is TrialVariant.POLICY_ON
    ]
    assert [request.matched_candidate_hypothesis_ids for request in on_requests] == [
        ("bundle-organize",),
        ("bundle-offer",),
    ]
    assert all(
        request.candidate_delta_program_set_hash == expected_delta_hash
        for request in on_requests
    )
    assert all(
        request.candidate_full_program_set_hash == expected_delta_hash
        for request in backend.requests
    )
    assert all(
        request.matched_candidate_program_set_hash
        == skilllearn_program_set_treatment_hash(
            (organize,) if request.item_id == "organize-messy-files-5" else (offer,)
        )
        for request in backend.requests
    )


def test_skilllearn_bundle_delta_miss_strictly_aliases_active_baseline(
    tmp_path: Path,
) -> None:
    harness, backend, _, _, _, sink = _harness(tmp_path)
    baseline = replace(
        _family_trigger_program(
            "incumbent-organize",
            ("organize-messy-files",),
        ),
        status=HypothesisStatus.PROMOTED,
    )
    delta = _family_trigger_program(
        "delta-offer",
        ("offer-letter-generator",),
    )

    pairs = harness.counterfactual_runner.run_bundle(
        harness.tasks(("organize-messy-files-5",)),
        programs=(delta,),
        baseline_programs=(baseline,),
        split=SplitName.VALIDATION,
        trace_id="bundle-delta-miss",
    )

    assert len(backend.calls) == 1
    assert backend.requests[0].candidate_full_program_set_hash == (
        skilllearn_program_set_treatment_hash((baseline, delta))
    )
    assert backend.requests[0].matched_candidate_hypothesis_ids == ()
    pair = pairs[0]
    assert pair.candidate.action_activated is False
    assert pair.candidate.baseline_preserved is True
    assert pair.candidate.selected_result.lane == "skilllearn_incumbent"
    assert pair.candidate.activated_hypothesis_ids == ("incumbent-organize",)
    assert pair.candidate_outcome == pair.baseline_outcome
    event = next(
        row
        for row in sink.events
        if row["event"] == "skilllearn_counterfactual_pair_completed"
    )
    assert event["payload"]["selected_candidate_hypothesis_ids"] == [
        "delta-offer"
    ]
    assert event["payload"]["matched_candidate_hypothesis_ids"] == []
    assert event["payload"]["candidate_trial_executed"] is False


def test_skilllearn_bundle_identity_is_set_sensitive_and_order_invariant(
    tmp_path: Path,
) -> None:
    harness, backend, _, _, _, sink = _harness(tmp_path)
    organize = _family_trigger_program(
        "bundle-organize",
        ("organize-messy-files",),
    )
    offer = _family_trigger_program(
        "bundle-offer",
        ("offer-letter-generator",),
    )
    tasks = harness.tasks(("organize-messy-files-5",))

    for programs in ((organize,), (organize, offer), (offer, organize)):
        harness.counterfactual_runner.run_bundle(
            tasks,
            programs=programs,
            split=SplitName.VALIDATION,
            trace_id="bundle-identity",
        )

    events = [
        row["payload"]
        for row in sink.events
        if row["event"] == "skilllearn_counterfactual_pair_completed"
    ]
    assert len(events) == 3
    assert events[0]["pair_id"] != events[1]["pair_id"]
    assert events[1]["pair_id"] == events[2]["pair_id"]
    assert (
        events[0]["candidate_delta_program_set_hash"]
        != events[1]["candidate_delta_program_set_hash"]
    )
    assert (
        events[1]["candidate_delta_program_set_hash"]
        == events[2]["candidate_delta_program_set_hash"]
    )
    assert harness.counterfactual_runner.behavior_set_hash((organize, offer)) == (
        harness.counterfactual_runner.behavior_set_hash((offer, organize))
    )
    on_requests = [
        request
        for request in backend.requests
        if request.variant is TrialVariant.POLICY_ON
    ]
    assert len(on_requests) == 3
    assert on_requests[0].request_hash != on_requests[1].request_hash
    assert on_requests[1].request_hash == on_requests[2].request_hash
    assert (
        on_requests[1].selected_candidate_hypothesis_ids
        == on_requests[2].selected_candidate_hypothesis_ids
        == ("bundle-offer", "bundle-organize")
    )


def test_v314_bundle_selector_executes_selected_union_once_per_validation_item(
    tmp_path: Path,
) -> None:
    organize = _family_trigger_program(
        "bundle-organize",
        ("organize-messy-files",),
    )
    offer = _family_trigger_program(
        "bundle-offer",
        ("offer-letter-generator",),
    )
    rejected_payload = _program_dict()
    rejected_payload["id"] = "bundle-no-support"
    rejected_payload["trigger"] = {
        "all_of": [
            {"key": "family", "op": "eq", "value": "never-matched-family"}
        ],
        "any_of": [],
        "none_of": [],
    }
    rejected_payload["verifier"]["repair_on_failure"] = False
    harness, backend, model, archive, _, sink = _harness(
        tmp_path,
        proposal_rows=[
            organize.to_dict(),
            offer.to_dict(),
            rejected_payload,
        ],
        candidate_selection_policy=(
            COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION
        ),
        candidate_bundle_policy=CANDIDATE_BUNDLE_POLICY_VERSION,
        contrastive_training_evidence_policy=(
            ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
        ),
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
        trace_id="bundle-selector-single-validation",
    )

    assert result.evolution is not None
    assert result.evolution.selected_candidate_hypothesis_ids == (
        "bundle-offer",
        "bundle-organize",
    )
    assert result.to_dict()["selected_candidate_hypothesis_ids"] == [
        "bundle-offer",
        "bundle-organize",
    ]
    selected_programs = tuple(
        archive.hypotheses[hypothesis_id]
        for hypothesis_id in result.evolution.selected_candidate_hypothesis_ids
    )
    assert result.to_dict()["evaluated_candidate_treatment_hash"] == (
        skilllearn_program_set_treatment_hash(selected_programs)
    )
    assert result.evolution.promoted is True
    assert len(backend.calls) == 8
    assert sum(row[1] == "policy_on" for row in backend.calls) == 2
    selection_events = [
        row
        for row in sink.events
        if row["event"] == "hypothesis_training_candidate_selection_completed"
    ]
    assert len(selection_events) == 1
    pair_events = [
        row
        for row in sink.events
        if row["event"] == "skilllearn_counterfactual_pair_completed"
    ]
    assert len(pair_events) == 2
    coverage = model.requests[0]["capabilities"]["train_coverage_objective"]
    assert coverage["family_target_deficit_capped_at_target"] is True
    assert coverage["post_target_actual_family_count_tiebreak"] is True
    assert coverage["actual_family_count_precedes_failure_support"] is True
    assert coverage["failure_support_precedes_bundle_size"] is True
    assert "coverage_reward_capped_at_target" not in coverage


def test_lowered_equivalent_program_replays_without_resampling(
    tmp_path: Path,
) -> None:
    harness, backend, _, _, _, sink = _harness(tmp_path)
    first = _program_dict()
    second = _program_dict()
    second["action_graph"][0]["id"] = "renamed-execute-action"
    second["action_graph"][1]["depends_on"] = ["renamed-execute-action"]
    second["verifier"]["checks"] = ["different_external_metadata"]
    second["expected_effect"]["maximum_cost_ratio"] = 1.05
    first_program = HypothesisProgram.from_dict(first)
    second_program = HypothesisProgram.from_dict(second)
    cache = CounterfactualEvidenceReplayCache(event_sink=sink)
    tasks = harness.tasks(("organize-messy-files-5",))

    source = cache.run_or_replay(
        runner=harness.counterfactual_runner,
        tasks=tasks,
        program=first_program,
        baseline_programs=(),
        split=SplitName.VALIDATION,
        trace_id="lowered-replay-source",
    )
    replayed = cache.run_or_replay(
        runner=harness.counterfactual_runner,
        tasks=tasks,
        program=second_program,
        baseline_programs=(),
        split=SplitName.VALIDATION,
        trace_id="lowered-replay-target",
    )

    assert replayed == source
    assert first_program.payload_hash != second_program.payload_hash
    assert len(backend.calls) == 2
    replay_event = next(
        row
        for row in sink.events
        if row["event"] == "counterfactual_evidence_replayed"
        and row["payload"]["target_trace_id"] == "lowered-replay-target"
    )
    assert replay_event["payload"]["new_counterfactual_executions"] == 0

    first_descriptor = harness._training_replay_descriptor(
        train_ids=("organize-messy-files-1",),
        incumbent_programs=(replace(first_program, status=HypothesisStatus.PROMOTED),),
    )
    second_descriptor = harness._training_replay_descriptor(
        train_ids=("organize-messy-files-1",),
        incumbent_programs=(replace(second_program, status=HypothesisStatus.PROMOTED),),
    )
    assert (
        first_descriptor["incumbent_behavior_set_hash"]
        == second_descriptor["incumbent_behavior_set_hash"]
    )


def test_missing_compiled_candidate_is_not_recorded_as_applied(
    tmp_path: Path,
) -> None:
    harness, backend, _, _, _, sink = _harness(tmp_path)
    delegate = harness.counterfactual_runner.compiler

    class RemovingCompiler:
        def compile(self, **kwargs):
            result = delegate.compile(**kwargs)
            if "challenger" in str(kwargs.get("method_name")):
                shutil.rmtree(result.output_root / "items")
            return result

    harness.counterfactual_runner.compiler = RemovingCompiler()
    pairs = harness.counterfactual_runner.run(
        harness.tasks(("organize-messy-files-5",)),
        program=HypothesisProgram.from_dict(_program_dict()),
        split=SplitName.VALIDATION,
        trace_id="missing-compiled-treatment",
    )

    assert len(backend.calls) == 1
    assert pairs[0].candidate.action_activated is False
    assert pairs[0].candidate.activated_hypothesis_ids == ()
    assert pairs[0].candidate_outcome.metrics["evaluation_valid"] == 0.0
    event = next(
        row
        for row in sink.events
        if row["event"] == "skilllearn_counterfactual_pair_completed"
    )
    assert event["payload"]["trigger_matched"] is True
    assert event["payload"]["treatment_applied"] is False
    assert event["payload"]["candidate_trial_executed"] is False


def test_missing_installed_treatment_receipt_is_not_recorded_as_applied(
    tmp_path: Path,
) -> None:
    harness, backend, _, _, _, sink = _harness(tmp_path)
    backend.requires_installed_skill_receipt = True

    pairs = harness.counterfactual_runner.run(
        harness.tasks(("organize-messy-files-5",)),
        program=HypothesisProgram.from_dict(_program_dict()),
        split=SplitName.VALIDATION,
        trace_id="missing-installed-treatment-receipt",
    )

    assert len(backend.calls) == 2
    assert pairs[0].candidate.action_activated is False
    event = next(
        row
        for row in sink.events
        if row["event"] == "skilllearn_counterfactual_pair_completed"
    )
    assert event["payload"]["treatment_applied"] is False
    assert event["payload"]["candidate_trial_executed"] is False


def test_external_active_ids_are_filtered_by_item_trigger(tmp_path: Path) -> None:
    harness, _, _, _, _, _ = _harness(tmp_path)
    matched_payload = _program_dict(status="promoted")
    matched_payload["id"] = "matched-incumbent"
    missed_payload = _program_dict(status="promoted")
    missed_payload["id"] = "missed-incumbent"
    missed_payload["trigger"] = {
        "all_of": [{"key": "family", "op": "eq", "value": "never-match"}],
        "any_of": [],
        "none_of": [],
    }
    candidate_payload = _program_dict()
    candidate_payload["id"] = "matched-candidate"

    pairs = harness.counterfactual_runner.run(
        harness.tasks(("organize-messy-files-5",)),
        program=HypothesisProgram.from_dict(candidate_payload),
        baseline_programs=(
            HypothesisProgram.from_dict(matched_payload),
            HypothesisProgram.from_dict(missed_payload),
        ),
        split=SplitName.VALIDATION,
        trace_id="per-item-active-ids",
    )

    assert pairs[0].baseline.activated_hypothesis_ids == ("matched-incumbent",)
    assert pairs[0].candidate.activated_hypothesis_ids == (
        "matched-incumbent",
        "matched-candidate",
    )


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


def test_malformed_root_response_preserves_typed_terminal_generation(
    tmp_path: Path,
) -> None:
    harness, backend, model, _, guard, sink = _harness(tmp_path)
    model.responses[:] = [{}]

    result = harness.run_generation(
        train_item_ids=(
            "organize-messy-files-1",
            "offer-letter-generator-1",
        ),
        validation_item_ids=(
            "organize-messy-files-5",
            "offer-letter-generator-5",
        ),
        trace_id="malformed-root-terminal",
    )

    assert result.evolution is None
    assert result.reason == "proposal_model_failed"
    assert result.to_dict()["proposal_model_failure_count"] == 1
    assert len(model.requests) == 1
    assert len(backend.calls) == 2
    assert guard.test_accessed is False
    rejected = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_proposal_response_rejected"
    )
    terminal = next(
        row
        for row in sink.events
        if row["event"] == "skilllearn_generation_stopped_after_proposal_model_failure"
    )
    assert rejected["payload"]["candidate_local_failure"] is False
    assert terminal["payload"]["failure_phase"] == "response_envelope"
    assert terminal["payload"]["response_hash"] == rejected["payload"]["response_hash"]
    assert terminal["payload"]["performance_claim_eligible"] is False


def test_repair_model_failure_blocks_generation_promotion(
    tmp_path: Path,
) -> None:
    bad = _program_dict()
    bad["id"] = "hyp-needs-repair"
    bad["action_graph"][0]["operation"] = "enable_lane"
    bad["action_graph"][0]["target"] = "missing_lane"
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


def test_malformed_repair_is_local_but_blocks_multi_root_generation(
    tmp_path: Path,
) -> None:
    bad = _program_dict()
    bad["id"] = "hyp-needs-malformed-repair"
    bad["action_graph"][0]["operation"] = "enable_lane"
    bad["action_graph"][0]["target"] = "missing_lane"
    good = _program_dict()
    good["id"] = "hyp-valid-sibling"
    harness, backend, model, archive, guard, sink = _harness(
        tmp_path,
        proposal_rows=[bad, good],
    )
    model.responses.append({"hypotheses": [good]})

    result = harness.run_generation(
        train_item_ids=(
            "organize-messy-files-1",
            "offer-letter-generator-1",
        ),
        validation_item_ids=(
            "organize-messy-files-5",
            "offer-letter-generator-5",
        ),
        trace_id="malformed-repair-multi-root",
    )

    assert result.evolution is not None
    assert result.reason == "proposal_model_failed"
    assert result.evolution.repair_model_failure_count == 1
    assert result.evolution.static_accepted_candidate_count == 1
    assert result.evolution.static_validation_node_count == 2
    assert result.evolution.promotion_decision is None
    assert len(model.requests) == 2
    assert len(backend.calls) == 2
    assert archive.hypotheses["hyp-needs-malformed-repair"].status is HypothesisStatus.REJECTED
    assert archive.hypotheses["hyp-valid-sibling"].status is HypothesisStatus.SHADOW
    assert archive.incumbent_id is None
    assert guard.test_accessed is False
    assert not any(
        row["event"] == "counterfactual_pair_completed" for row in sink.events
    )
    rejected = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_proposal_response_rejected"
    )
    abandoned = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_repair_abandoned_after_model_failure"
    )
    assert rejected["payload"]["candidate_local_failure"] is True
    assert abandoned["payload"]["failure_phase"] == "response_envelope"
    assert abandoned["payload"]["response_hash"] == rejected["payload"]["response_hash"]
    assert any(
        row["event"] == "evolution_generation_blocked_by_repair_model_failure"
        and row["payload"]["counterfactual_validation_executed"] is False
        and row["payload"]["archive_promotion_allowed"] is False
        for row in sink.events
    )


def test_invalid_external_pair_blocks_promotion(tmp_path: Path) -> None:
    harness, _, _, archive, _, _ = _harness(
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
    score = next(iter(archive.score_records.values()))
    assert score.valid is False
    assert score.invalidation_reason == "invalid_counterfactual_evidence"
    assert _advance_arm(
        result,
        consecutive_non_promotions=1,
        maximum=2,
        counterfactual_invalid_evidence_policy=(
            COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION
        ),
    ) == (False, 1, "invalid_counterfactual_evidence")


@pytest.mark.parametrize(
    "mismatch_field",
    ("provider_mismatch_count", "budget_mismatch_count"),
)
def test_counterfactual_mismatch_stops_arm_without_increment(
    mismatch_field: str,
) -> None:
    summary = {
        "invalid_pair_count": 0,
        "provider_mismatch_count": 0,
        "budget_mismatch_count": 0,
    }
    summary[mismatch_field] = 1

    class Generation:
        reason = "promotion_gate_rejected"
        evolution = None

        def to_dict(self) -> dict[str, object]:
            return {
                "proposal_model_failure_count": 0,
                "promotion_summary": summary,
                "promotion_decision": {"summary": summary},
            }

    assert _advance_arm(
        Generation(),
        consecutive_non_promotions=1,
        maximum=2,
        counterfactual_invalid_evidence_policy=(
            COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION
        ),
    ) == (False, 1, "invalid_counterfactual_evidence")

    assert _advance_arm(
        Generation(),
        consecutive_non_promotions=0,
        maximum=2,
    ) == (True, 1, "max_generations_reached")


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
    assert len(backend.calls) == 3
    assert sum(call[1] == "policy_off" for call in backend.calls) == 1
    assert not any(
        row["event"] == "counterfactual_evidence_replayed"
        for row in sink.events
    )
    assert sum(
        row["event"] == "counterfactual_evidence_not_recorded_invalid"
        for row in sink.events
    ) == 2


@pytest.mark.parametrize(
    "fingerprint_field",
    ("provider_fingerprint", "fairness_fingerprint"),
)
def test_mismatched_counterfactual_bundle_does_not_enter_replay_cache(
    tmp_path: Path,
    fingerprint_field: str,
) -> None:
    target = "offer-letter-generator-5"
    harness, backend, _, _, _, sink = _harness(tmp_path)
    delegate = harness.counterfactual_runner

    class MismatchRunner:
        runtime = delegate.runtime
        evaluator = delegate.evaluator

        def run(self, *args, **kwargs):
            pairs = delegate.run(*args, **kwargs)
            mismatched = []
            for pair in pairs:
                metadata = dict(pair.candidate.selected_result.metadata)
                metadata[fingerprint_field] = "mismatch"
                candidate = replace(
                    pair.candidate,
                    selected_result=replace(
                        pair.candidate.selected_result,
                        metadata=metadata,
                    ),
                )
                mismatched.append(replace(pair, candidate=candidate))
            return tuple(mismatched)

    cache = CounterfactualEvidenceReplayCache(event_sink=sink)
    program = HypothesisProgram.from_dict(_program_dict())
    tasks = harness.tasks((target,))
    runner = MismatchRunner()

    for trace_id in ("mismatch-cache-source", "mismatch-cache-target"):
        cache.run_or_replay(
            runner=runner,
            tasks=tasks,
            program=program,
            baseline_programs=(),
            split=SplitName.VALIDATION,
            trace_id=trace_id,
        )

    assert len(backend.calls) == 3
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


def test_training_replay_identity_binds_v315_action_profile_capture(
    tmp_path: Path,
) -> None:
    legacy, _, _, _, _, _ = _harness(tmp_path / "legacy")
    action, _, _, _, _, _ = _harness(
        tmp_path / "action",
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
    )
    train_ids = ("organize-messy-files-1",)

    legacy_descriptor = legacy._training_replay_descriptor(
        train_ids=train_ids,
        incumbent_programs=(),
    )
    action_descriptor = action._training_replay_descriptor(
        train_ids=train_ids,
        incumbent_programs=(),
    )

    assert "train_action_design_policy" not in legacy_descriptor
    assert "train_action_trace_profile_version" not in legacy_descriptor
    assert action_descriptor["train_action_design_policy"] == (
        TRAIN_ACTION_DESIGN_POLICY_VERSION
    )
    assert action_descriptor["train_action_environment_profile_version"]
    assert action_descriptor["train_action_trace_profile_version"]
    assert stable_hash(legacy_descriptor) != stable_hash(action_descriptor)


def test_shared_baseline_arm_cache_reuses_one_immutable_cohort_across_runners(
    tmp_path: Path,
) -> None:
    shared_cache = BaselineArmEvidenceReplayCache(
        policy=SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
    )
    recursive, recursive_backend, _, _, _, recursive_sink = _harness(
        tmp_path / "recursive",
        baseline_arm_replay_cache=shared_cache,
        baseline_arm_evidence_replay_policy=(
            SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    no_recursive, no_recursive_backend, _, _, _, no_recursive_sink = _harness(
        tmp_path / "no-recursive",
        baseline_arm_replay_cache=shared_cache,
        baseline_arm_evidence_replay_policy=(
            SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    programs = []
    for index in range(3):
        payload = _program_dict()
        payload["id"] = f"shared-baseline-generation-{index + 1}"
        payload["action_graph"][1]["value"] += f" Generation {index + 1}."
        programs.append(HypothesisProgram.from_dict(payload))
    item_id = "organize-messy-files-5"

    first = recursive.counterfactual_runner.run(
        recursive.tasks((item_id,)),
        program=programs[0],
        split=SplitName.VALIDATION,
        trace_id="shared-baseline-recursive-g1",
    )[0]
    second = no_recursive.counterfactual_runner.run(
        no_recursive.tasks((item_id,)),
        program=programs[1],
        split=SplitName.VALIDATION,
        trace_id="shared-baseline-no-recursive-g2",
    )[0]
    third = recursive.counterfactual_runner.run(
        recursive.tasks((item_id,)),
        program=programs[2],
        split=SplitName.VALIDATION,
        trace_id="shared-baseline-recursive-g3",
    )[0]

    validation_off_requests = [
        request
        for request in (*recursive_backend.requests, *no_recursive_backend.requests)
        if request.split is SplitName.VALIDATION
        and request.variant is TrialVariant.POLICY_OFF
    ]
    assert len(validation_off_requests) == 1
    assert len(shared_cache) == 1
    assert first.baseline_outcome == second.baseline_outcome == third.baseline_outcome
    assert (
        first.baseline.selected_result.cost
        == second.baseline.selected_result.cost
        == third.baseline.selected_result.cost
    )
    pair_events = [
        row
        for row in (*recursive_sink.events, *no_recursive_sink.events)
        if row["event"] == "skilllearn_counterfactual_pair_completed"
    ]
    assert len({row["payload"]["baseline_evidence_hash"] for row in pair_events}) == 1
    assert [row["payload"]["baseline_replayed"] for row in pair_events] == [
        False,
        True,
        True,
    ]
    assert [
        row["payload"]["baseline_trial_executed"] for row in pair_events
    ] == [True, False, False]
    assert all(
        row["payload"]["run_order"] == "on_only_shared_baseline_replay"
        for row in pair_events[1:]
    )
    replay_events = [
        row
        for row in (*recursive_sink.events, *no_recursive_sink.events)
        if row["event"] == "skilllearn_baseline_arm_evidence_replayed"
    ]
    assert len(replay_events) == 2
    assert all(
        row["payload"]["new_baseline_executions"] == 0
        for row in replay_events
    )


def test_terminal_invalid_baseline_memo_prevents_cross_consumer_resampling(
    tmp_path: Path,
) -> None:
    shared_cache = BaselineArmEvidenceReplayCache(
        policy=TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
    )
    source_backend = ValidationBaselineSequenceBackend(
        ("trial_network_byte_limit_exceeded",)
    )
    target_backend = ValidationBaselineSequenceBackend((None,))
    recursive, _, _, _, _, recursive_sink = _harness(
        tmp_path / "recursive-terminal-invalid",
        backend_override=source_backend,
        invalid_trial_max_attempts=3,
        baseline_arm_replay_cache=shared_cache,
        baseline_arm_evidence_replay_policy=(
            TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    no_recursive, _, _, _, _, no_recursive_sink = _harness(
        tmp_path / "no-recursive-terminal-invalid",
        backend_override=target_backend,
        invalid_trial_max_attempts=3,
        baseline_arm_replay_cache=shared_cache,
        baseline_arm_evidence_replay_policy=(
            TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    program = HypothesisProgram.from_dict(_program_dict())
    item_id = "organize-messy-files-5"

    first = recursive.counterfactual_runner.run(
        recursive.tasks((item_id,)),
        program=program,
        split=SplitName.VALIDATION,
        trace_id="terminal-invalid-source",
    )[0]
    second = no_recursive.counterfactual_runner.run(
        no_recursive.tasks((item_id,)),
        program=program,
        split=SplitName.VALIDATION,
        trace_id="terminal-invalid-target",
    )[0]

    assert source_backend.validation_baseline_call_count == 1
    assert target_backend.validation_baseline_call_count == 0
    assert len(shared_cache) == 1
    assert first.baseline_outcome.metrics["evaluation_valid"] == 0.0
    assert second.baseline_outcome.metrics["evaluation_valid"] == 0.0
    memoized = next(
        row
        for row in recursive_sink.events
        if row["event"] == "skilllearn_baseline_arm_terminal_invalid_memoized"
    )
    replayed = next(
        row
        for row in no_recursive_sink.events
        if row["event"] == "skilllearn_baseline_arm_terminal_invalid_replayed"
    )
    assert memoized["payload"]["promotion_evidence"] is False
    assert memoized["payload"]["terminal_for_replay_key"] is True
    assert memoized["payload"]["new_baseline_executions"] == 1
    assert replayed["payload"]["promotion_evidence"] is False
    assert replayed["payload"]["terminal_for_replay_key"] is True
    assert replayed["payload"]["new_baseline_executions"] == 0
    assert (
        replayed["payload"]["source_terminal_outcome_hash"]
        == memoized["payload"]["source_terminal_outcome_hash"]
    )
    pair_events = [
        next(
            row
            for row in sink.events
            if row["event"] == "skilllearn_counterfactual_pair_completed"
        )
        for sink in (recursive_sink, no_recursive_sink)
    ]
    assert [
        row["payload"]["baseline_terminal_invalid_memoized"]
        for row in pair_events
    ] == [True, False]
    assert [
        row["payload"]["baseline_terminal_invalid_replayed"]
        for row in pair_events
    ] == [False, True]
    assert all(
        row["payload"]["baseline_promotion_evidence_eligible"] is False
        for row in pair_events
    )
    assert len(
        {
            row["payload"]["baseline_evidence_hash"]
            for row in pair_events
        }
    ) == 1


def test_v2_invalid_baseline_remains_unmemoized_for_historical_replay_semantics(
    tmp_path: Path,
) -> None:
    shared_cache = BaselineArmEvidenceReplayCache(
        policy=SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
    )
    source_backend = ValidationBaselineSequenceBackend(
        ("trial_network_byte_limit_exceeded",)
    )
    target_backend = ValidationBaselineSequenceBackend((None,))
    first_harness, _, _, _, _, first_sink = _harness(
        tmp_path / "v2-invalid-source",
        backend_override=source_backend,
        invalid_trial_max_attempts=3,
        baseline_arm_replay_cache=shared_cache,
        baseline_arm_evidence_replay_policy=(
            SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    second_harness, _, _, _, _, second_sink = _harness(
        tmp_path / "v2-valid-target",
        backend_override=target_backend,
        invalid_trial_max_attempts=3,
        baseline_arm_replay_cache=shared_cache,
        baseline_arm_evidence_replay_policy=(
            SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    program = HypothesisProgram.from_dict(_program_dict())
    item_id = "organize-messy-files-5"

    first = first_harness.counterfactual_runner.run(
        first_harness.tasks((item_id,)),
        program=program,
        split=SplitName.VALIDATION,
        trace_id="v2-invalid-source",
    )[0]
    second = second_harness.counterfactual_runner.run(
        second_harness.tasks((item_id,)),
        program=program,
        split=SplitName.VALIDATION,
        trace_id="v2-valid-target",
    )[0]

    assert source_backend.validation_baseline_call_count == 1
    assert target_backend.validation_baseline_call_count == 1
    assert first.baseline_outcome.metrics["evaluation_valid"] == 0.0
    assert second.baseline_outcome.metrics["evaluation_valid"] == 1.0
    assert len(shared_cache) == 1
    assert any(
        row["event"] == "skilllearn_baseline_arm_evidence_not_recorded_invalid"
        for row in first_sink.events
    )
    assert any(
        row["event"] == "skilllearn_baseline_arm_evidence_recorded"
        for row in second_sink.events
    )
    assert not any(
        "terminal_invalid" in row["event"]
        for row in (*first_sink.events, *second_sink.events)
    )


def test_terminal_memo_policy_allows_declared_same_request_clean_replacement(
    tmp_path: Path,
) -> None:
    shared_cache = BaselineArmEvidenceReplayCache(
        policy=TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
    )
    source_backend = ValidationBaselineSequenceBackend(
        ("endpoint_error", None)
    )
    target_backend = ValidationBaselineSequenceBackend((None,))
    source, _, _, _, _, source_sink = _harness(
        tmp_path / "same-request-replacement-source",
        backend_override=source_backend,
        invalid_trial_max_attempts=2,
        baseline_arm_replay_cache=shared_cache,
        baseline_arm_evidence_replay_policy=(
            TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    target, _, _, _, _, target_sink = _harness(
        tmp_path / "same-request-replacement-target",
        backend_override=target_backend,
        invalid_trial_max_attempts=2,
        baseline_arm_replay_cache=shared_cache,
        baseline_arm_evidence_replay_policy=(
            TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    program = HypothesisProgram.from_dict(_program_dict())
    item_id = "organize-messy-files-5"

    first = source.counterfactual_runner.run(
        source.tasks((item_id,)),
        program=program,
        split=SplitName.VALIDATION,
        trace_id="same-request-replacement-source",
    )[0]
    second = target.counterfactual_runner.run(
        target.tasks((item_id,)),
        program=program,
        split=SplitName.VALIDATION,
        trace_id="same-request-replacement-target",
    )[0]

    assert source_backend.validation_baseline_call_count == 2
    assert target_backend.validation_baseline_call_count == 0
    assert first.baseline_outcome.metrics["evaluation_valid"] == 1.0
    assert second.baseline_outcome.metrics["evaluation_valid"] == 1.0
    assert any(
        row["event"] == "skilllearn_invalid_trial_clean_replacement"
        for row in source_sink.events
    )
    assert any(
        row["event"] == "skilllearn_baseline_arm_evidence_replayed"
        for row in target_sink.events
    )
    assert not any(
        "terminal_invalid" in row["event"]
        for row in (*source_sink.events, *target_sink.events)
    )


def test_baseline_arm_cache_rejects_invalid_and_conflicting_records() -> None:
    cache = BaselineArmEvidenceReplayCache(
        policy=SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
    )
    request = SkillLearnTrialRequest(
        item_id="validation-item",
        family="validation-family",
        split=SplitName.VALIDATION,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="eval-epoch",
        pair_id="source-pair",
        repeat=1,
        agent_id="codex",
        model="model",
        max_steps=20,
        manifest_hash="manifest",
        program_set_hash="no-skill-program-set",
        treatment_hash="no-skill-treatment",
    )
    invalid = SkillLearnTrialObservation(
        request=request,
        success=False,
        score=0.0,
        metrics={"evaluation_valid": 0.0, "task_success": 0.0},
        total_tokens=0,
        steps=0,
        duration_seconds=0.0,
        provider_fingerprint="provider",
        fairness_fingerprint="fairness",
        error_type="endpoint_error",
    )
    assert cache.record(
        "replay-key",
        observation=invalid,
        source_trace_id="invalid",
    ) is None
    assert len(cache) == 0

    source_metrics = {"evaluation_valid": 1.0, "task_success": 0.0}
    valid = replace(
        invalid,
        metrics=source_metrics,
        error_type=None,
        total_tokens=100,
        steps=10,
    )
    recorded = cache.record(
        "replay-key",
        observation=valid,
        source_trace_id="valid",
    )
    assert recorded is not None
    source_metrics["task_success"] = 1.0
    assert recorded.observation.metrics["task_success"] == 0.0

    conflict = replace(
        valid,
        success=True,
        score=1.0,
        metrics={"evaluation_valid": 1.0, "task_success": 1.0},
    )
    assert cache.record(
        "replay-key",
        observation=conflict,
        source_trace_id="conflict",
    ) is None
    assert len(cache) == 1
    assert cache.get("replay-key") is recorded

    terminal_cache = BaselineArmEvidenceReplayCache(
        policy=TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
    )
    memo = terminal_cache.memoize_terminal_invalid(
        "terminal-replay-key",
        observation=invalid,
        source_trace_id="terminal-invalid",
    )
    assert memo is not None
    assert memo.error_type == "endpoint_error"
    assert terminal_cache.record(
        "terminal-replay-key",
        observation=valid,
        source_trace_id="late-valid",
    ) is None
    assert terminal_cache.get("terminal-replay-key") is memo

    valid_first = terminal_cache.record(
        "valid-first-key",
        observation=valid,
        source_trace_id="valid-first",
    )
    assert valid_first is not None
    assert terminal_cache.memoize_terminal_invalid(
        "valid-first-key",
        observation=invalid,
        source_trace_id="late-invalid",
    ) is None
    assert terminal_cache.get("valid-first-key") is valid_first
    assert cache.memoize_terminal_invalid(
        "v2-does-not-memoize-invalid",
        observation=invalid,
        source_trace_id="v2-invalid",
    ) is None


def test_shared_baseline_replay_key_separates_split_and_baseline_treatment(
    tmp_path: Path,
) -> None:
    harness, _, _, _, _, _ = _harness(
        tmp_path,
        baseline_arm_evidence_replay_policy=(
            SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    runner = harness.counterfactual_runner
    task = harness.tasks(("organize-messy-files-5",))[0]
    promoted = replace(
        HypothesisProgram.from_dict(_program_dict()),
        status=HypothesisStatus.PROMOTED,
    )

    no_skill = runner._baseline_arm_replay_key(
        task,
        split=SplitName.VALIDATION,
        baseline_programs=(),
        baseline_treatment_hash=NO_SKILL_TREATMENT_HASH,
    )
    promoted_a = runner._baseline_arm_replay_key(
        task,
        split=SplitName.VALIDATION,
        baseline_programs=(promoted,),
        baseline_treatment_hash="baseline-treatment-a",
    )
    promoted_b = runner._baseline_arm_replay_key(
        task,
        split=SplitName.VALIDATION,
        baseline_programs=(promoted,),
        baseline_treatment_hash="baseline-treatment-b",
    )
    test_split = runner._baseline_arm_replay_key(
        task,
        split=SplitName.TEST,
        baseline_programs=(promoted,),
        baseline_treatment_hash="baseline-treatment-a",
    )

    assert len({no_skill, promoted_a, promoted_b, test_split}) == 4


def test_terminal_baseline_replay_key_binds_invalid_retry_identity(
    tmp_path: Path,
) -> None:
    retry_configurations = (
        {
            "invalid_trial_max_attempts": 2,
            "invalid_trial_retry_backoff_seconds": 0.125,
            "invalid_trial_retry_workers": 2,
        },
        {
            "invalid_trial_max_attempts": 3,
            "invalid_trial_retry_backoff_seconds": 0.125,
            "invalid_trial_retry_workers": 2,
        },
        {
            "invalid_trial_max_attempts": 2,
            "invalid_trial_retry_backoff_seconds": 0.25,
            "invalid_trial_retry_workers": 2,
        },
        {
            "invalid_trial_max_attempts": 2,
            "invalid_trial_retry_backoff_seconds": 0.125,
            "invalid_trial_retry_workers": 3,
        },
    )
    terminal_harnesses = []
    for index, retry_configuration in enumerate(retry_configurations):
        harness, _, _, _, _, _ = _harness(
            tmp_path / f"terminal-{index}",
            baseline_arm_evidence_replay_policy=(
                TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
            ),
            **retry_configuration,
        )
        terminal_harnesses.append(harness)

    terminal_runners = [
        harness.counterfactual_runner for harness in terminal_harnesses
    ]
    task = terminal_harnesses[0].tasks(("organize-messy-files-5",))[0]
    terminal_keys = {
        runner._baseline_arm_replay_key(
            task,
            split=SplitName.VALIDATION,
            baseline_programs=(),
            baseline_treatment_hash=NO_SKILL_TREATMENT_HASH,
        )
        for runner in terminal_runners
    }

    assert terminal_runners[0].invalid_trial_retry_descriptor() == {
        "policy": INVALID_TRIAL_RETRY_POLICY_VERSION,
        **retry_configurations[0],
    }
    assert len(terminal_keys) == len(retry_configurations)

    for policy_index, policy in enumerate(
        (
            LEGACY_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
            SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
        )
    ):
        historical_keys = set()
        for config_index, retry_configuration in enumerate(
            (retry_configurations[0], retry_configurations[-1])
        ):
            harness, _, _, _, _, _ = _harness(
                tmp_path / f"historical-{policy_index}-{config_index}",
                baseline_arm_evidence_replay_policy=policy,
                **retry_configuration,
            )
            runner = harness.counterfactual_runner
            historical_keys.add(
                runner._baseline_arm_replay_key(
                    harness.tasks(("organize-messy-files-5",))[0],
                    split=SplitName.VALIDATION,
                    baseline_programs=(),
                    baseline_treatment_hash=NO_SKILL_TREATMENT_HASH,
                )
            )
        assert len(historical_keys) == 1


def test_legacy_baseline_arm_caches_remain_runner_local(tmp_path: Path) -> None:
    first, first_backend, _, _, _, _ = _harness(tmp_path / "first")
    second, second_backend, _, _, _, _ = _harness(tmp_path / "second")
    assert (
        first.counterfactual_runner.baseline_arm_replay_cache
        is not second.counterfactual_runner.baseline_arm_replay_cache
    )
    program = HypothesisProgram.from_dict(_program_dict())
    item_id = "organize-messy-files-5"

    first.counterfactual_runner.run(
        first.tasks((item_id,)),
        program=program,
        split=SplitName.VALIDATION,
        trace_id="legacy-first",
    )
    second.counterfactual_runner.run(
        second.tasks((item_id,)),
        program=program,
        split=SplitName.VALIDATION,
        trace_id="legacy-second",
    )

    assert sum(
        request.variant is TrialVariant.POLICY_OFF
        for request in first_backend.requests
    ) == 1
    assert sum(
        request.variant is TrialVariant.POLICY_OFF
        for request in second_backend.requests
    ) == 1


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


def test_paired_ablation_v2_binds_one_shared_baseline_cache(
    tmp_path: Path,
) -> None:
    recursive, _, _, _, _, _ = _harness(
        tmp_path / "recursive",
        baseline_arm_evidence_replay_policy=(
            SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    no_recursive, _, _, _, _, _ = _harness(
        tmp_path / "no-recursive",
        baseline_arm_evidence_replay_policy=(
            SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    no_recursive.validator.proposer = None
    assert (
        recursive.counterfactual_runner.baseline_arm_replay_cache
        is not no_recursive.counterfactual_runner.baseline_arm_replay_cache
    )

    _run_paired_arms(
        recursive_harness=recursive,
        no_recursive_harness=no_recursive,
        train_ids=(
            "organize-messy-files-1",
            "organize-messy-files-2",
            "offer-letter-generator-1",
            "offer-letter-generator-2",
        ),
        validation_ids=(
            "organize-messy-files-5",
            "offer-letter-generator-5",
        ),
        manifest_hash=recursive.manifest.manifest_hash,
        max_generations=1,
        max_consecutive_non_promotions=1,
    )

    assert (
        recursive.counterfactual_runner.baseline_arm_replay_cache
        is no_recursive.counterfactual_runner.baseline_arm_replay_cache
    )
    assert len(recursive.counterfactual_runner.baseline_arm_replay_cache) == 2


@pytest.mark.parametrize(
    "mismatched_retry_configuration",
    (
        {"invalid_trial_max_attempts": 2},
        {"invalid_trial_retry_backoff_seconds": 0.125},
        {"invalid_trial_retry_workers": 2},
    ),
)
def test_paired_terminal_replay_rejects_mismatched_retry_configuration(
    tmp_path: Path,
    mismatched_retry_configuration: dict[str, Any],
) -> None:
    recursive, recursive_backend, _, _, _, _ = _harness(
        tmp_path / "recursive",
        baseline_arm_evidence_replay_policy=(
            TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
    )
    no_recursive, no_recursive_backend, _, _, _, _ = _harness(
        tmp_path / "no-recursive",
        baseline_arm_evidence_replay_policy=(
            TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
        **mismatched_retry_configuration,
    )

    with pytest.raises(
        ValueError,
        match="terminal-invalid replay arms must share the invalid trial retry",
    ):
        _run_paired_arms(
            recursive_harness=recursive,
            no_recursive_harness=no_recursive,
            train_ids=(
                "organize-messy-files-1",
                "organize-messy-files-2",
            ),
            validation_ids=("organize-messy-files-5",),
            manifest_hash=recursive.manifest.manifest_hash,
            max_generations=1,
            max_consecutive_non_promotions=1,
        )

    assert recursive_backend.calls == []
    assert no_recursive_backend.calls == []
    assert (
        recursive.counterfactual_runner.baseline_arm_replay_cache
        is not no_recursive.counterfactual_runner.baseline_arm_replay_cache
    )


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
    assert len(recursive_backend.calls) == 10
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


def test_contrastive_miner_labels_valid_failures_and_success_controls(
    tmp_path: Path,
) -> None:
    successful = {
        "anthropic-poster-design-2",
        "anthropic-poster-design-5",
    }
    train_ids = (
        "organize-messy-files-1",
        "organize-messy-files-2",
        *sorted(successful),
    )
    contrastive, _, _, _, _, sink = _harness(
        tmp_path / "contrastive",
        backend_override=SelectiveTrainingSuccessBackend(successful),
        candidate_selection_policy=(
            CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION
        ),
        contrastive_training_evidence_policy=(
            CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
        ),
    )

    observations = contrastive.collect_training_observations(
        train_item_ids=train_ids,
        trace_id="contrastive-mixed-observations",
    )
    examples = contrastive.residual_miner.mine(
        observations,
        trace_id="contrastive-mixed-residuals",
    )

    failures = [row for row in examples if not row.baseline_success]
    controls = [row for row in examples if row.baseline_success]
    assert len(failures) == 2
    assert len(controls) == 2
    assert all(row.context for row in failures)
    assert all(
        row.failure_type == "baseline_success_control"
        and row.evaluator_feedback == ()
        and row.context == {}
        for row in controls
    )
    mined = next(
        row
        for row in sink.events
        if row["event"] == "skilllearn_training_residuals_mined"
    )
    assert mined["payload"]["residual_count"] == 2
    assert mined["payload"]["success_control_count"] == 2
    assert mined["payload"]["example_count"] == 4

    legacy, _, _, _, _, _ = _harness(
        tmp_path / "legacy",
        backend_override=SelectiveTrainingSuccessBackend(successful),
    )
    legacy_observations = legacy.collect_training_observations(
        train_item_ids=train_ids,
        trace_id="legacy-mixed-observations",
    )
    legacy_residuals = legacy.residual_miner.mine(legacy_observations)
    assert len(legacy_residuals) == 2
    assert all(not row.baseline_success for row in legacy_residuals)
    assert legacy.candidate_selection_policy == (
        TRAIN_ONLY_CANDIDATE_SELECTION_VERSION
    )
    assert legacy.contrastive_training_evidence_policy is None


def test_contrastive_all_success_stops_both_arms_before_proposal(
    tmp_path: Path,
) -> None:
    train_ids = (
        "organize-messy-files-1",
        "organize-messy-files-2",
        "offer-letter-generator-1",
        "offer-letter-generator-2",
    )
    policy_args = {
        "candidate_selection_policy": (
            CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION
        ),
        "contrastive_training_evidence_policy": (
            CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
        ),
    }
    recursive, _, recursive_model, _, _, sink = _harness(
        tmp_path / "recursive",
        backend_override=SelectiveTrainingSuccessBackend(set(train_ids)),
        **policy_args,
    )
    no_recursive, _, no_recursive_model, _, _, _ = _harness(
        tmp_path / "no-recursive",
        **policy_args,
    )
    no_recursive.validator.proposer = None

    paired = _run_paired_arms(
        recursive_harness=recursive,
        no_recursive_harness=no_recursive,
        train_ids=train_ids,
        validation_ids=(
            "organize-messy-files-5",
            "offer-letter-generator-5",
        ),
        manifest_hash=recursive.manifest.manifest_hash,
        max_generations=1,
        max_consecutive_non_promotions=1,
    )

    for key in ("recursive_generations", "no_recursive_generations"):
        generation = paired[key][0]
        assert generation.reason == "no_valid_failed_training_rows"
        assert generation.evolution is None
        assert generation.to_dict()["training_residual_count"] == 0
        assert generation.to_dict()["success_control_count"] == 4
        assert generation.to_dict()["example_count"] == 4
    assert recursive_model.requests == []
    assert no_recursive_model.requests == []
    checkpoint = next(
        row
        for row in sink.events
        if row["event"] == "skilllearn_paired_ablation_checkpoint_frozen"
    )
    assert checkpoint["payload"]["labeled_transition_ids"] == sorted(
        row.transition_id for row in paired["recursive_generations"][0].residuals
    )
    assert checkpoint["payload"]["residual_count"] == 0
    assert checkpoint["payload"]["success_control_count"] == 4


def test_contrastive_selection_prefers_zero_false_positives_over_more_support(
    tmp_path: Path,
) -> None:
    success_ids = {
        "anthropic-poster-design-2",
        "anthropic-poster-design-5",
    }
    broad = _program_dict()
    broad["id"] = "broad-four-failures-two-false-positives"
    precise = _program_dict()
    precise["id"] = "precise-two-failures-zero-false-positives"
    precise["trigger"] = {
        "all_of": [
            {
                "key": "family",
                "op": "eq",
                "value": "organize-messy-files",
            }
        ],
        "any_of": [],
        "none_of": [],
    }
    harness, _, _, _, _, sink = _harness(
        tmp_path,
        backend_override=SelectiveTrainingSuccessBackend(success_ids),
        proposal_rows=[broad, precise],
        candidate_selection_policy=(
            CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION
        ),
        contrastive_training_evidence_policy=(
            CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
        ),
    )
    harness.validator.proposer = None

    result = harness.run_generation(
        train_item_ids=(
            "organize-messy-files-1",
            "organize-messy-files-2",
            "offer-letter-generator-1",
            "offer-letter-generator-2",
            *sorted(success_ids),
        ),
        validation_item_ids=(
            "organize-messy-files-5",
            "offer-letter-generator-5",
        ),
        trace_id="contrastive-precision-selection",
    )

    assert result.evolution is not None
    assert result.evolution.root_hypothesis_id == precise["id"]
    selection = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_training_candidate_selection_completed"
    )["payload"]
    by_root = {row["root_id"]: row for row in selection["candidates"]}
    precise_metrics = by_root[precise["id"]]["contrastive_training_metrics"]
    broad_metrics = by_root[broad["id"]]["contrastive_training_metrics"]
    assert (precise_metrics["activation_precision_numerator"], precise_metrics["activation_precision_denominator"]) == (2, 2)
    assert (broad_metrics["activation_precision_numerator"], broad_metrics["activation_precision_denominator"]) == (4, 6)
    assert precise_metrics["success_false_positive_activation_count"] == 0
    assert broad_metrics["success_false_positive_activation_count"] == 2
    assert selection["selection_policy"] == (
        CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION
    )
    assert selection["selection_uses_validation"] is False
    assert selection["selection_uses_validation_outcomes"] is False


def test_family_coverage_selection_prefers_two_families_over_exact_two_of_two() -> None:
    residuals = (
        _family_residual("a-failure-1", family="family-a"),
        _family_residual("a-failure-2", family="family-a"),
        _family_residual("b-failure", family="family-b"),
        _family_residual(
            "b-success",
            family="family-b",
            baseline_success=True,
        ),
    )
    single_family = _family_trigger_program(
        "single-family-exact-two-of-two",
        ("family-a",),
    )
    two_families = _family_trigger_program(
        "two-family-lower-precision",
        ("family-a", "family-b"),
    )
    target = _training_family_coverage_target(
        residuals,
        minimum_activation_rate=1.0,
    )
    single_metrics = _training_candidate_metrics(
        single_family,
        residuals,
    ).to_dict(family_coverage_target=target)
    two_family_metrics = _training_candidate_metrics(
        two_families,
        residuals,
    ).to_dict(family_coverage_target=target)

    single_score = _training_candidate_score(
        single_family,
        residuals,
        selection_policy=(
            PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
        ),
        family_coverage_target=target,
    )
    two_family_score = _training_candidate_score(
        two_families,
        residuals,
        selection_policy=(
            PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
        ),
        family_coverage_target=target,
    )

    assert target == 2
    assert (
        single_metrics["activation_precision_numerator"],
        single_metrics["activation_precision_denominator"],
    ) == (2, 2)
    assert (
        two_family_metrics["activation_precision_numerator"],
        two_family_metrics["activation_precision_denominator"],
    ) == (3, 4)
    assert single_metrics["failure_activation_family_deficit"] == 1
    assert two_family_metrics["failure_activation_family_deficit"] == 0
    assert two_family_score < single_score


def test_family_coverage_target_caps_breadth_before_exact_precision() -> None:
    residuals = (
        _family_residual("a-failure", family="family-a"),
        _family_residual("b-failure", family="family-b"),
        _family_residual("c-failure", family="family-c"),
        _family_residual(
            "c-success",
            family="family-c",
            baseline_success=True,
        ),
        _family_residual("d-failure", family="family-d"),
        replace(
            _family_residual("heldout-validation", family="heldout-family"),
            split=SplitName.VALIDATION,
            features={
                "benchmark": "skilllearnbench",
                "family": "heldout-family",
                "validation_only_signal": True,
            },
        ),
    )
    target_met_precise = _family_trigger_program(
        "target-met-precise",
        ("family-a", "family-b"),
    )
    extra_breadth_false_positive = _family_trigger_program(
        "extra-breadth-false-positive",
        ("family-a", "family-b", "family-c"),
    )
    target = _training_family_coverage_target(
        residuals,
        minimum_activation_rate=0.5,
    )
    precise_score = _training_candidate_score(
        target_met_precise,
        residuals,
        selection_policy=(
            PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
        ),
        family_coverage_target=target,
    )
    broad_score = _training_candidate_score(
        extra_breadth_false_positive,
        residuals,
        selection_policy=(
            PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
        ),
        family_coverage_target=target,
    )
    precise_metrics = _training_candidate_metrics(
        target_met_precise,
        residuals,
    ).to_dict(family_coverage_target=target)
    broad_metrics = _training_candidate_metrics(
        extra_breadth_false_positive,
        residuals,
    ).to_dict(family_coverage_target=target)

    assert target == 2
    assert precise_score < broad_score
    assert precise_metrics["failure_activation_family_count"] == 2
    assert broad_metrics["failure_activation_family_count"] == 3
    assert broad_metrics["train_family_count"] == 4
    assert broad_metrics["failure_activation_family_target"] == 2
    assert broad_metrics["failure_activation_family_deficit"] == 0
    assert broad_metrics["failure_activation_family_target_met"] is True


def test_family_coverage_proposal_request_contains_diverse_batch_contract(
    tmp_path: Path,
) -> None:
    proposal_rows = [
        _family_trigger_program("family-a-specialist", ("family-a",)).to_dict(),
        _family_trigger_program("family-b-specialist", ("family-b",)).to_dict(),
        _family_trigger_program(
            "cross-family-coverage",
            ("family-a", "family-b"),
        ).to_dict(),
    ]
    harness, _, model, archive, guard, _ = _harness(
        tmp_path,
        proposal_rows=proposal_rows,
        candidate_selection_policy=(
            CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION
        ),
        contrastive_training_evidence_policy=(
            CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
        ),
    )
    kernel = EvolutionKernel(
        proposer=harness.proposer,
        validator=harness.validator,
        counterfactual_runner=harness.counterfactual_runner,
        promotion_gate=harness.promotion_gate,
        archive=archive,
        split_guard=guard,
        proposal_candidates_per_generation=3,
        candidate_selection_policy=(
            PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
        ),
        contrastive_training_evidence_policy=(
            CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
        ),
    )
    residuals = (
        _family_residual("a-failure", family="family-a"),
        _family_residual("b-failure", family="family-b"),
    )
    context = ValidationContext(
        evaluator_epoch="skilllearn-eval-epoch-0",
        residuals=residuals,
        available_lanes=frozenset({"baseline", "candidate"}),
        baseline_lane="baseline",
        action_semantics=SKILL_ACTION_LOWERING_VERSION,
        contrastive_training_evidence_policy=(
            CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
        ),
    )

    proposals = kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="family-coverage-proposal-contract",
    )

    assert len(proposals) == 3
    assert len(model.requests) == 1
    assert "hypotheses" in model.requests[0]["output_schema"]["properties"]
    assert "family_slot_response_contract" not in model.requests[0]
    request_contract = model.requests[0]["proposal_batch_contract"]
    capability_contract = model.requests[0]["capabilities"][
        "proposal_batch_contract"
    ]
    assert request_contract["policy"] == PROPOSAL_DIVERSITY_POLICY_VERSION
    assert request_contract["required_count"] == 3
    assert request_contract["diversity_unit"] == (
        "train_failure_activation_or_action_treatment"
    )
    assert request_contract["activation_signature_distinctness"] == (
        "search_preference_audit_only"
    )
    assert request_contract["action_treatment_diversity"] == (
        "allowed_when_activation_signatures_coincide"
    )
    assert request_contract["max_action_nodes_per_hypothesis"] == 4
    assert request_contract["compact_output"] is True
    response_schema = model.requests[0]["output_schema"]["properties"][
        "hypotheses"
    ]
    assert response_schema["minItems"] == response_schema["maxItems"] == 3
    action_schema = response_schema["items"]["action_graph"][0]
    assert action_schema["target"] == (
        "task-local subject of the imperative directive"
    )
    assert action_schema["value"] == (
        "complete imperative task-local sentence grounded in TRAIN residual "
        "context.task_instruction; never an enum-only value, mapping/mode/check "
        "label, or preserve_baseline claim"
    )
    assert response_schema["items"]["fallback"] == "preserve_baseline"
    assert capability_contract["required_count"] == 3
    assert capability_contract["profile_roles"] == [
        "train_only_precision_anchor",
        "train_only_cross_family_coverage",
        "train_only_action_treatment_diversity",
    ]
    coverage = model.requests[0]["capabilities"]["train_coverage_objective"]
    assert coverage == {
        "policy": PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
        "evidence_scope": "train_only",
        "coverage_unit": "distinct_failure_family",
        "minimum_activation_rate": 1.0,
        "train_family_count": 2,
        "failure_activation_family_target": 2,
        "coverage_reward_capped_at_target": True,
        "validation_features_used": False,
        "validation_outcomes_used": False,
    }
    constraints = model.requests[0]["constraints"]
    assert constraints["candidate_search_uses_train_only"] is True
    assert constraints["candidate_search_family_target"] == 2
    assert constraints["candidate_search_validation_outcomes_forbidden"] is True
    assert constraints[
        "proposal_activation_signature_distinctness_is_search_preference"
    ] is True
    assert constraints[
        "proposal_activation_signatures_are_audited_not_rejected"
    ] is True
    assert constraints[
        "proposal_action_or_backend_treatment_diversity_allowed"
    ] is True
    assert constraints[
        "proposal_same_activation_signature_allowed_when_treatment_differs"
    ] is True
    assert (
        "proposal_train_failure_activation_sets_must_be_pairwise_distinct"
        not in constraints
    )
    assert constraints[
        "prompt_directive_action_values_must_be_complete_imperative_task_local_sentences"
    ] is True
    assert constraints["prompt_directive_action_value_grounding_source"] == (
        "TRAIN residual context.task_instruction"
    )
    assert constraints[
        "prompt_directive_enum_only_action_values_forbidden"
    ] is True
    assert constraints[
        "prompt_directive_mapping_mode_check_labels_forbidden"
    ] is True
    assert constraints[
        "prompt_directive_activated_action_preserve_baseline_claim_forbidden"
    ] is True
    assert constraints[
        "prompt_directive_top_level_fallback_remains_preserve_baseline"
    ] is True


def test_profile_grounded_family_slots_scope_three_singular_train_requests(
    tmp_path: Path,
) -> None:
    families = ("family-a", "family-b", "family-c")
    profiles = {
        f"profile-{family}": _family_slot_profile(family)
        for family in families
    }
    residuals = tuple(
        _family_slot_residual(
            f"{family}-failure-{index}",
            family=family,
            profile_hash=f"profile-{family}",
        )
        for family in families
        for index in range(2)
    ) + (
        _family_residual(
            "success-control-1",
            family="success-family-1",
            baseline_success=True,
        ),
        _family_residual(
            "success-control-2",
            family="success-family-2",
            baseline_success=True,
        ),
    )
    proposal_rows = [
        _family_slot_program(family).to_dict() for family in families
    ]
    harness, _, model, _, _, sink = _harness(
        tmp_path,
        proposal_rows=proposal_rows,
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        proposal_formation_policy=PROPOSAL_FORMATION_POLICY_VERSION,
    )
    context = ValidationContext(
        evaluator_epoch="skilllearn-eval-epoch-0",
        residuals=residuals,
        available_lanes=frozenset({"baseline", "candidate"}),
        baseline_lane="baseline",
        action_semantics=SKILL_ACTION_LOWERING_VERSION,
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        action_design_profiles=profiles,
    )

    proposals = harness.kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="profile-grounded-family-slots",
    )

    assert len(proposals) == len(model.requests) == 3
    assert stable_hash(model.requests[0]) == (
        "aaa6f35db6662ede625cfd9cebb0d46b728755e862d9a8be0e8c2430fd157693"
    )
    success_ids = {"success-control-1", "success-control-2"}
    for request, expected_family in zip(model.requests, families):
        assert request["max_hypotheses"] == 1
        assert request["output_schema"]["required"] == ["hypothesis"]
        assert set(request["output_schema"]["properties"]) == {"hypothesis"}
        assert request["family_slot_response_contract"] == {
            "policy": PROPOSAL_FORMATION_POLICY_VERSION,
            "response_field": "hypothesis",
            "response_type": "object",
            "required_count": 1,
            "root_batch_contract_applies": False,
            "response_rejection_by_diversity_allowed": False,
            "proposal_retry_by_diversity_allowed": False,
            "compact_output": True,
        }
        assert "proposal_batch_contract" not in request
        capabilities = request["capabilities"]
        slot_contract = capabilities["family_slot_contract"]
        assert slot_contract["policy"] == PROPOSAL_FORMATION_POLICY_VERSION
        assert slot_contract["target_failure_family"] == expected_family
        assert slot_contract["target_failure_support_count"] == 2
        assert slot_contract["success_control_count"] == 2
        assert slot_contract["validation_outcomes_used"] is False
        assert slot_contract["verifier_content_used"] is False
        assert slot_contract["test_content_used"] is False
        assert capabilities["action_quality_contract"]["policy"] == (
            TRAIN_ACTION_DESIGN_POLICY_VERSION
        )
        assert set(capabilities["train_action_design_profiles"]) == {
            f"profile-{expected_family}"
        }
        assert capabilities["prior_hypotheses"] == []
        assert capabilities["prior_promotion_feedback"] == []
        portable = slot_contract["portable_recipe_policy"]
        assert portable["minimum_same_family_train_evidence_for_literal"] == 2
        assert portable["otherwise_extract_from"] == (
            "current_task_or_artifact"
        )
        assert portable["reusable_preferred_primitive_count"] > 0
        preferred = portable["preferred_allowlisted_profile_primitives"]
        failed = portable["failed_profile_primitives_to_avoid"]
        assert any(
            row["kind"] == "executable"
            and row["value"] == f"{expected_family}-tool"
            and row["train_failure_evidence_count"] == 2
            for row in preferred
        )
        assert any(
            row["kind"] == "executable"
            and row["value"] == f"{expected_family}-broken"
            and row["train_failure_evidence_count"] == 2
            for row in failed
        )
        request_failures = [
            row for row in request["residuals"] if not row["baseline_success"]
        ]
        request_successes = [
            row for row in request["residuals"] if row["baseline_success"]
        ]
        assert {row["family"] for row in request_failures} == {
            expected_family
        }
        assert len(request_failures) == 2
        assert {
            row["transition_id"] for row in request_successes
        } == success_ids
        constraints = request["constraints"]
        assert constraints["proposal_target_failure_family"] == expected_family
        assert constraints[
            "trigger_must_include_exact_target_family_predicate"
        ] == {"key": "family", "op": "eq", "value": expected_family}
        assert constraints[
            "reusable_preferred_primitive_requires_action_binding"
        ] is True
        assert constraints["exact_constant_alone_is_insufficient"] is True
        action_value_schema = request["output_schema"]["properties"][
            "hypothesis"
        ]["action_graph"][0]["value"]
        assert "must bind a canonical preferred profile primitive" in (
            action_value_schema
        )
        assert "exact constant alone is insufficient" in action_value_schema

    plan = next(
        row
        for row in sink.events
        if row["event"] == "proposal_family_slot_plan_created"
    )
    completed = [
        row
        for row in sink.events
        if row["event"] == "proposal_family_slot_completed"
    ]
    assert plan["payload"]["slot_count"] == 3
    assert plan["payload"]["distinct_target_family_count"] == 3
    assert plan["payload"]["raw_content_persisted"] is False
    assert len(completed) == 3
    assert all(
        row["payload"]["candidate_matched_target_support"] == 2
        and row["payload"]["matched_family_count"] == 1
        and row["payload"]["profile_binding_count"] >= 1
        and row["payload"]["preferred_primitive_count"] >= 1
        and row["payload"]["preferred_primitive_set_hash"]
        and row["payload"]["failed_primitive_set_hash"]
        and row["payload"]["portable_delta_kinds"]
        and row["payload"]["raw_content_persisted"] is False
        for row in completed
    )


def test_family_slot_v2_fixes_trigger_artifact_blueprint_and_hides_failed_values(
    tmp_path: Path,
) -> None:
    families = ("family-a", "family-b", "family-c")
    profiles = {
        f"profile-{family}": _family_slot_profile(family)
        for family in families
    }
    residuals = tuple(
        _family_slot_residual(
            f"{family}-failure-{index}",
            family=family,
            profile_hash=f"profile-{family}",
        )
        for family in families
        for index in range(2)
    )
    proposal_rows = [
        _family_slot_v2_program(
            family,
            include_failed_primitive=(index == 0),
        ).to_dict()
        for index, family in enumerate(families)
    ]
    harness, _, model, _, _, sink = _harness(
        tmp_path,
        proposal_rows=proposal_rows,
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        proposal_formation_policy=PROPOSAL_FORMATION_POLICY_V2,
    )
    context = ValidationContext(
        evaluator_epoch="skilllearn-eval-epoch-0",
        residuals=residuals,
        available_lanes=frozenset({"baseline", "candidate"}),
        baseline_lane="baseline",
        action_semantics=SKILL_ACTION_LOWERING_VERSION,
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        action_design_profiles=profiles,
    )

    proposals = harness.kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="profile-grounded-family-slots-v2",
    )

    assert len(proposals) == len(model.requests) == 3
    for request, expected_family in zip(model.requests, families):
        serialized_request = json.dumps(request, sort_keys=True)
        assert f"{expected_family}-broken" not in serialized_request
        capabilities = request["capabilities"]
        assert "train_action_design_profiles" not in capabilities
        profile_summary = capabilities[
            "train_action_design_profile_summary"
        ]
        assert profile_summary["failed_primitive_count"] == 1
        assert profile_summary["failed_primitive_set_hash"]
        assert profile_summary["failed_primitive_values_disclosed"] is False
        slot_contract = capabilities["family_slot_contract"]
        assert slot_contract["policy"] == PROPOSAL_FORMATION_POLICY_V2
        portable = slot_contract["portable_recipe_policy"]
        assert "failed_profile_primitives_to_avoid" not in portable
        assert portable["failed_primitive_count"] == 1
        assert portable["failed_primitive_set_hash"]
        assert portable["failed_primitive_values_disclosed"] is False
        recommended = portable["recommended_artifact"]
        assert recommended == {
            "kind": "artifact_command_path",
            "value": f"/root/{expected_family}.json",
            "train_failure_evidence_count": 2,
            "reusable_across_same_family_failures": True,
        }
        assert portable["recommended_artifact_selection_priority"] == [
            "artifact_command_path",
            "artifact_task_local_path",
            "artifact_copied_file",
            "artifact_environment_source_file",
        ]
        blueprint = portable["required_artifact_workflow_blueprint"]
        assert f"/root/{expected_family}.json" in blueprint
        assert all(
            stage in blueprint
            for stage in ("read", "parse", "update", "serialize", "write")
        )
        hypothesis_schema = request["output_schema"]["properties"][
            "hypothesis"
        ]
        exact_trigger = {
            "all_of": [
                {"key": "family", "op": "eq", "value": expected_family}
            ],
            "any_of": [],
            "none_of": [],
        }
        empty_anti_trigger = {
            "all_of": [],
            "any_of": [],
            "none_of": [],
        }
        assert hypothesis_schema["trigger"] == exact_trigger
        assert hypothesis_schema["anti_trigger"] == empty_anti_trigger
        constraints = request["constraints"]
        assert constraints[
            "trigger_schema_must_equal_exact_target_family_only"
        ] == exact_trigger
        assert constraints["anti_trigger_schema_must_equal_empty"] == (
            empty_anti_trigger
        )
        assert constraints[
            "recommended_artifact_value_must_be_mentioned_exactly"
        ] == f"/root/{expected_family}.json"
        action_schema = hypothesis_schema["action_graph"][0]["value"]
        assert f"/root/{expected_family}.json" in action_schema
        assert "read -> parse -> update -> serialize -> write-back" in (
            action_schema
        )

    assert proposals[0].anti_trigger.is_empty
    assert [
        (row.key, row.op, row.value)
        for row in proposals[0].trigger.all_of
    ] == [
        ("family", "eq", "family-a")
    ]
    assert proposals[0].trigger.any_of == ()
    assert proposals[0].trigger.none_of == ()
    completed = [
        row
        for row in sink.events
        if row["event"] == "proposal_family_slot_completed"
    ]
    assert completed[0]["payload"]["failed_profile_binding_count"] == 1
    assert completed[0]["payload"]["response_rejected_by_diversity"] is False
    assert completed[0]["payload"]["proposal_retry_by_diversity"] is False
    assert completed[0]["payload"]["policy"] == PROPOSAL_FORMATION_POLICY_V2


def test_family_slot_profile_primitives_are_failure_dominant_and_path_safe() -> None:
    profile = {
        "runtime_environment": {
            "declared_os_packages": ["completed-nonzero"],
            "declared_python_packages": [],
            "declared_task_local_paths": ["/root/available.json"],
            "copied_task_files": [],
            "environment_source_files": [],
        },
        "baseline_action_trace": {
            "command_signatures": [
                {
                    "executable_basename": "completed-nonzero",
                    "task_local_paths": ["/root/not-the-cause-a.json"],
                    "status": "completed",
                    "exit_code": 1,
                },
                {
                    "executable_basename": "failed-zero",
                    "task_local_paths": ["/root/not-the-cause-b.json"],
                    "status": "failed",
                    "exit_code": 0,
                },
                {
                    "executable_basename": "successful-tool",
                    "task_local_paths": ["/root/success.json"],
                    "status": "completed",
                    "exit_code": 0,
                },
            ]
        },
    }

    preferred, failed = _allowlisted_profile_primitives(profile)

    assert ("executable", "completed-nonzero") in failed
    assert ("environment_os_package", "completed-nonzero") not in preferred
    assert ("executable", "failed-zero") in failed
    assert ("executable", "successful-tool") in preferred
    assert ("artifact_command_path", "/root/success.json") in preferred
    assert (
        "artifact_command_path",
        "/root/not-the-cause-a.json",
    ) not in failed | preferred
    assert (
        "artifact_command_path",
        "/root/not-the-cause-b.json",
    ) not in failed | preferred
    assert ("artifact_task_local_path", "/root/available.json") in preferred
    assert preferred.isdisjoint(failed)


def test_family_slot_formation_requires_train_action_profile_policy(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="profile-grounded family-slot proposal formation requires TRAIN action design policy",
    ):
        _harness(
            tmp_path,
            proposal_formation_policy=PROPOSAL_FORMATION_POLICY_VERSION,
            train_action_design_policy=None,
        )


def test_family_slot_formation_rejects_non_train_before_model(
    tmp_path: Path,
) -> None:
    families = ("family-a", "family-b", "family-c")
    residuals = tuple(
        replace(
            _family_slot_residual(
                f"{family}-failure",
                family=family,
                profile_hash=f"profile-{family}",
            ),
            split=(
                SplitName.VALIDATION
                if family == "family-c"
                else SplitName.TRAIN
            ),
        )
        for family in families
    )
    profiles = {
        f"profile-{family}": _family_slot_profile(family)
        for family in families
    }
    harness, _, model, _, _, _ = _harness(
        tmp_path,
        proposal_rows=[
            _family_slot_program(family).to_dict() for family in families
        ],
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        proposal_formation_policy=PROPOSAL_FORMATION_POLICY_VERSION,
    )
    context = ValidationContext(
        evaluator_epoch="skilllearn-eval-epoch-0",
        residuals=residuals,
        available_lanes=frozenset({"baseline", "candidate"}),
        baseline_lane="baseline",
        action_semantics=SKILL_ACTION_LOWERING_VERSION,
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        action_design_profiles=profiles,
    )

    with pytest.raises(
        PermissionError,
        match="proposal_residual_not_training",
    ):
        harness.kernel.propose_candidates(
            residuals,
            validation_context=context,
            trace_id="family-slot-non-train",
        )

    assert model.requests == []


def test_family_slot_plan_is_deterministic_rotates_and_shared_g1_matches(
    tmp_path: Path,
) -> None:
    families = tuple(f"family-{suffix}" for suffix in "abcdef")
    profiles = {
        f"profile-{family}": _family_slot_profile(family)
        for family in families
    }
    residuals = tuple(
        _family_slot_residual(
            f"{family}-failure-{index}",
            family=family,
            profile_hash=f"profile-{family}",
        )
        for family in families
        for index in range(2)
    )
    context = ValidationContext(
        evaluator_epoch="skilllearn-eval-epoch-0",
        residuals=residuals,
        available_lanes=frozenset({"baseline", "candidate"}),
        baseline_lane="baseline",
        action_semantics=SKILL_ACTION_LOWERING_VERSION,
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        action_design_profiles=profiles,
    )
    all_rows = [_family_slot_program(family).to_dict() for family in families]
    harness_a, _, model_a, archive_a, _, sink_a = _harness(
        tmp_path / "a",
        proposal_rows=all_rows,
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        proposal_formation_policy=PROPOSAL_FORMATION_POLICY_VERSION,
    )

    shared_g1 = harness_a.kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="family-slot-a-g1",
    )
    harness_a.kernel.validator = RecursiveValidationEngine(
        (),
        proposer=harness_a.proposer,
        event_sink=sink_a,
    )
    family_use_after_proposal = dict(
        harness_a.kernel._proposal_family_use_counts
    )
    harness_a.kernel.evolve_once(
        residuals=residuals,
        validation_tasks=harness_a.tasks(
            harness_a.manifest.validation_ids[:2]
        ),
        validation_context=context,
        proposal_candidates=shared_g1,
        trace_id="family-slot-a-shared-own-g1",
    )
    assert harness_a.kernel._proposal_family_use_counts == (
        family_use_after_proposal
    )
    own_replay = next(
        row
        for row in sink_a.events
        if row["event"] == "proposal_family_slot_usage_replayed"
    )
    assert own_replay["payload"]["source"] == "shared_proposal_candidates"
    assert own_replay["payload"]["proposal_set_replayed"] is True
    assert own_replay["payload"]["family_use_updated"] is False
    assert own_replay["payload"]["new_family_use_count"] == 0
    assert own_replay["payload"]["raw_content_persisted"] is False
    assert harness_a.kernel._promotion_feedback
    assert archive_a.hypotheses
    harness_a.kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="family-slot-a-g2",
    )
    a_g1_targets = [
        row["capabilities"]["family_slot_contract"]["target_failure_family"]
        for row in model_a.requests[:3]
    ]
    a_g2_targets = [
        row["capabilities"]["family_slot_contract"]["target_failure_family"]
        for row in model_a.requests[3:]
    ]
    assert a_g1_targets == list(families[:3])
    assert a_g2_targets == list(families[3:])
    assert set(a_g1_targets).isdisjoint(a_g2_targets)
    assert all(
        request["capabilities"]["prior_hypotheses"] == []
        and request["capabilities"]["prior_promotion_feedback"] == []
        and request["capabilities"][
            "prior_history_excluded_from_family_slot_proposal"
        ]
        is True
        and request["capabilities"]["family_slot_contract"][
            "validation_outcomes_used"
        ]
        is False
        for request in model_a.requests[3:]
    )

    harness_fresh, _, model_fresh, _, _, _ = _harness(
        tmp_path / "fresh",
        proposal_rows=all_rows[:3],
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        proposal_formation_policy=PROPOSAL_FORMATION_POLICY_VERSION,
    )
    harness_fresh.kernel.propose_candidates(
        tuple(reversed(residuals)),
        validation_context=context,
        trace_id="family-slot-fresh-g1",
    )
    fresh_targets = [
        row["capabilities"]["family_slot_contract"]["target_failure_family"]
        for row in model_fresh.requests
    ]
    assert fresh_targets == a_g1_targets

    harness_b, _, model_b, _, _, sink_b = _harness(
        tmp_path / "b",
        proposal_rows=all_rows[3:],
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        proposal_formation_policy=PROPOSAL_FORMATION_POLICY_VERSION,
    )
    harness_b.kernel.validator = RecursiveValidationEngine(
        (),
        proposer=harness_b.proposer,
        event_sink=sink_b,
    )
    b_generation_proposer = harness_b.kernel.proposer
    harness_b.kernel.proposer = harness_a.proposer
    harness_b.kernel.evolve_once(
        residuals=residuals,
        validation_tasks=harness_b.tasks(
            harness_b.manifest.validation_ids[:2]
        ),
        validation_context=context,
        proposal_candidates=shared_g1,
        trace_id="family-slot-b-shared-g1",
    )
    harness_b.kernel.proposer = b_generation_proposer
    assert harness_b.kernel._proposal_family_use_counts == (
        family_use_after_proposal
    )
    harness_b.kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="family-slot-b-g2",
    )
    b_g2_targets = [
        row["capabilities"]["family_slot_contract"]["target_failure_family"]
        for row in model_b.requests
    ]
    assert b_g2_targets == a_g2_targets
    shared_usage = next(
        row
        for row in sink_b.events
        if row["event"] == "proposal_family_slot_usage_recorded"
        and row["payload"]["source"] == "shared_proposal_candidates"
    )
    assert shared_usage["payload"]["requested_target_count"] == 3
    assert shared_usage["payload"]["distinct_requested_target_count"] == 3
    assert shared_usage["payload"]["actual_matched_count"] == 3
    assert shared_usage["payload"][
        "distinct_actual_matched_family_count"
    ] == 3
    assert shared_usage["payload"]["raw_content_persisted"] is False


def test_family_slot_rotation_uses_requested_targets_and_identity_binds_plan(
    tmp_path: Path,
) -> None:
    families = tuple(f"family-{suffix}" for suffix in "abcdef")
    residuals = tuple(
        _family_slot_residual(
            f"{family}-failure-{index}",
            family=family,
            profile_hash=f"profile-{family}",
        )
        for family in families
        for index in range(2)
    )
    profiles = {
        f"profile-{family}": _family_slot_profile(family)
        for family in families
    }
    repeated_program_set = [
        _family_slot_program("family-f").to_dict(),
        _family_slot_program("family-b").to_dict(),
        _family_slot_program("family-c").to_dict(),
    ]
    harness, _, model, _, _, sink = _harness(
        tmp_path,
        proposal_rows=[*repeated_program_set, *repeated_program_set],
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        proposal_formation_policy=PROPOSAL_FORMATION_POLICY_VERSION,
    )
    context = ValidationContext(
        evaluator_epoch="skilllearn-eval-epoch-0",
        residuals=residuals,
        available_lanes=frozenset({"baseline", "candidate"}),
        baseline_lane="baseline",
        action_semantics=SKILL_ACTION_LOWERING_VERSION,
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
        action_design_profiles=profiles,
    )

    first = harness.kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="requested-target-g1",
    )
    second = harness.kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="requested-target-g2",
    )

    targets = [
        row["capabilities"]["family_slot_contract"]["target_failure_family"]
        for row in model.requests
    ]
    assert targets[:3] == list(families[:3])
    assert targets[3:] == list(families[3:])
    assert harness.kernel._proposal_family_use_counts == {
        family: 1 for family in families
    }
    assert harness.proposer.family_slot_targets_for(second) == families[3:]
    harness.kernel._record_matched_proposal_families(
        first,
        residuals=residuals,
        validation_context=context,
        trace_id="same-program-set-different-slots",
        source="test_same_program_set_different_slots",
        requested_targets=families[3:],
    )
    assert harness.kernel._proposal_family_use_counts == {
        **{family: 1 for family in families[:3]},
        **{family: 2 for family in families[3:]},
    }
    usage_events = [
        row
        for row in sink.events
        if row["event"] == "proposal_family_slot_usage_recorded"
    ]
    assert len(usage_events) == 3
    same_set_events = [
        row
        for row in usage_events
        if row["payload"]["proposal_set_hash"]
        == usage_events[0]["payload"]["proposal_set_hash"]
    ]
    assert len(same_set_events) == 2
    assert same_set_events[0]["payload"]["usage_identity_hash"] != (
        same_set_events[1]["payload"]["usage_identity_hash"]
    )
    assert all(
        row["payload"]["requested_target_count"] == 3
        and row["payload"]["family_use_updated"] is True
        and row["payload"]["proposal_set_replayed"] is False
        for row in usage_events
    )
    first_slot = next(
        row
        for row in sink.events
        if row["event"] == "proposal_family_slot_completed"
        and row["trace_id"] == "requested-target-g1:family-slot-1"
    )
    assert first_slot["payload"]["target_family"] == "family-a"
    assert first_slot["payload"]["candidate_matched_target"] is False
    assert first_slot["payload"]["candidate_matched_target_support"] == 0


def test_action_quality_profiles_shape_prompt_and_audit_without_gating() -> None:
    profile_hash = "profile-train-runtime-01"
    profile = {
        "policy": TRAIN_ACTION_DESIGN_POLICY_VERSION,
        "runtime_environment": {
            "declared_os_packages": ["trivy"],
            "declared_python_packages": ["pypdf==5.1.0"],
            "declared_task_local_paths": [
                "/root/.cache/trivy",
                "/root/package-lock.json",
                "/root/sc100-blank.pdf",
            ],
            "copied_task_files": [
                "/root/package-lock.json",
                "/root/sc100-blank.pdf",
            ],
            "environment_source_files": [],
        },
        "baseline_action_trace": {
            "command_signatures": [
                {
                    "executable_basename": "trivy",
                    "safe_flags": ["--format"],
                    "task_local_paths": ["/root/package-lock.json"],
                    "original_command_hash": "trace-command-hash",
                    "status": "failed",
                    "exit_code": 1,
                }
            ]
        },
    }
    failure = ResidualExample(
        transition_id="action-quality-failure",
        task_id="dependency-vulnerability-check-1",
        family="dependency-vulnerability-check",
        split=SplitName.TRAIN,
        features={
            "benchmark": "skilllearnbench",
            "family": "dependency-vulnerability-check",
            "has_container_environment": True,
        },
        failure_type="task_failed_with_actionable_training_feedback",
        evaluator_feedback=("Infer a concrete operator.",),
        baseline_success=False,
        context={
            "task_instruction": (
                "Audit the dependency file offline and write the results to CSV."
            ),
            "action_context_profile_hash": profile_hash,
            TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY: profile,
        },
    )
    success = ResidualExample(
        transition_id="action-quality-success",
        task_id="offer-letter-generator-1",
        family="offer-letter-generator",
        split=SplitName.TRAIN,
        features={
            "benchmark": "skilllearnbench",
            "family": "offer-letter-generator",
            "has_container_environment": True,
        },
        failure_type="baseline_success_control",
        evaluator_feedback=(),
        baseline_success=True,
        context={},
    )

    def proposal(program_id: str, value: str) -> dict[str, Any]:
        payload = _program_dict()
        payload["id"] = program_id
        payload["action_graph"] = [
            {
                "id": "material-delta",
                "operation": "execute_step",
                "target": "task_procedure",
                "value": value,
            }
        ]
        return payload

    rows = [
        proposal(
            "instruction-paraphrase",
            "Collect authoritative offline records and write the requested CSV.",
        ),
        proposal(
            "concrete-trivy-command",
            "Run `trivy fs --skip-db-update --cache-dir /root/.cache/trivy "
            "--format json /root/package-lock.json` before converting JSON to CSV.",
        ),
        proposal(
            "exact-brand-mapping",
            "Set the background to #141413 and the primary accent to #D97757.",
        ),
        proposal(
            "pdf-field-operation",
            "Call pypdf.PdfReader('/root/sc100-blank.pdf').get_fields() and "
            "update_page_form_field_values() before a reopen round-trip check.",
        ),
    ]
    sink = MemoryEventSink()
    model = QueueProposalModel([{"hypotheses": rows}])
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    contract = train_action_quality_contract(TRAIN_ACTION_DESIGN_POLICY_VERSION)
    assert contract is not None

    programs = proposer.propose(
        (failure, success),
        evaluator_epoch="skilllearn-eval-epoch-0",
        max_hypotheses=4,
        capabilities={
            "primary_metric": "task_success",
            "action_contract": {
                "allowed_action_operations": [
                    "execute_step",
                    "check_condition",
                    "produce_artifact",
                    "request_evidence",
                ],
                "semantics": SKILL_ACTION_LOWERING_VERSION,
                "external_evidence_is_hidden": True,
            },
            "action_quality_contract": contract,
            "train_action_design_profiles": {profile_hash: profile},
        },
        trace_id="action-quality-audit-only",
    )

    assert [program.id for program in programs] == [row["id"] for row in rows]
    request = model.requests[0]
    assert TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY not in request["residuals"][0][
        "context"
    ]
    assert request["residuals"][0]["context"][
        "action_context_profile_hash"
    ] == profile_hash
    assert request["residuals"][1]["context"] == {}
    assert request["constraints"][
        "task_instruction_is_baseline_requirement_not_treatment"
    ] is True
    assert request["constraints"]["action_quality_enforcement"] == (
        "prompt_and_audit_only"
    )
    assert "material delta absent" in request["output_schema"]["hypotheses"][0][
        "action_graph"
    ][0]["value"]

    audit = next(
        row for row in sink.events if row["event"] == "proposal_action_delta_audited"
    )["payload"]
    by_hash = {
        row["hypothesis_hash"]: row for row in audit["candidate_audits"]
    }
    program_audits = {
        program.id: by_hash[program.payload_hash] for program in programs
    }
    assert program_audits["instruction-paraphrase"]["restatement_risk"] is True
    assert program_audits["concrete-trivy-command"]["observed_delta_kinds"] == [
        "concrete_local_tool_command"
    ]
    assert program_audits["exact-brand-mapping"]["observed_delta_kinds"] == [
        "exact_constant_or_mapping"
    ]
    assert "artifact_internal_manipulation" in program_audits[
        "pdf-field-operation"
    ]["observed_delta_kinds"]
    assert audit["response_rejected"] is False
    assert audit["proposal_retry_requested"] is False
    assert audit["recursive_repair_requested_by_audit"] is False
    assert audit["candidate_selection_affected"] is False
    assert audit["promotion_gate_affected"] is False


def test_harness_builds_train_only_action_profiles_for_proposal(
    tmp_path: Path,
) -> None:
    harness, _, model, _, guard, _ = _harness(
        tmp_path,
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
    )
    train_ids = (
        "dependency-vulnerability-check-1",
        "dependency-vulnerability-check-4",
    )

    observations = harness.collect_training_observations(
        train_item_ids=train_ids,
        trace_id="action-profile-training",
    )
    residuals = harness.residual_miner.mine(
        observations,
        trace_id="action-profile-residuals",
    )
    harness.propose_candidates(
        residuals,
        trace_id="action-profile-proposal",
    )

    assert len(residuals) == 2
    assert all(
        row.context.get("action_context_profile_hash") for row in residuals
    )
    internal_profiles = [
        row.context[TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY]
        for row in residuals
    ]
    assert all(
        "trivy" in profile["runtime_environment"]["declared_os_packages"]
        for profile in internal_profiles
    )
    request = model.requests[0]
    assert request["capabilities"]["action_quality_contract"]["policy"] == (
        TRAIN_ACTION_DESIGN_POLICY_VERSION
    )
    assert len(request["capabilities"]["train_action_design_profiles"]) == 1
    assert len(
        {row.context["action_context_profile_hash"] for row in residuals}
    ) == 1
    assert all(
        TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY not in row["context"]
        for row in request["residuals"]
    )
    context = harness.validation_context(residuals)
    assert context.train_action_design_policy == TRAIN_ACTION_DESIGN_POLICY_VERSION
    assert len(context.action_design_profiles) == 1
    assert guard.test_accessed is False


def test_action_quality_audit_failure_cannot_change_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from assumption_agent import proposer as proposer_module

    def fail_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise TypeError("synthetic diagnostic failure")

    monkeypatch.setattr(proposer_module, "_action_delta_audit_row", fail_audit)
    sink = MemoryEventSink()
    model = QueueProposalModel(
        [{"hypotheses": [_program_dict()]}]
    )
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    residual = _family_residual("audit-no-throw", family="family-a")

    programs = proposer.propose(
        (residual,),
        evaluator_epoch="skilllearn-eval-epoch-0",
        max_hypotheses=1,
        capabilities={
            "action_quality_contract": train_action_quality_contract(
                TRAIN_ACTION_DESIGN_POLICY_VERSION
            ),
            "train_action_design_profiles": {},
        },
        trace_id="audit-no-throw",
    )

    assert len(programs) == 1
    failure = next(
        row
        for row in sink.events
        if row["event"] == "proposal_action_delta_audit_failed"
    )["payload"]
    assert failure["response_rejected"] is False
    assert failure["proposal_retry_requested"] is False
    assert failure["candidate_selection_affected"] is False
    assert failure["promotion_gate_affected"] is False
    assert failure["raw_error_persisted"] is False


def test_harness_binds_v312_repair_request_scope_into_validation(
    tmp_path: Path,
) -> None:
    harness, _, _, _, _, _ = _harness(
        tmp_path,
        repair_request_scope_policy=REPAIR_REQUEST_SCOPE_POLICY_VERSION,
    )
    residuals = (_family_residual("a-failure", family="family-a"),)

    context = harness.validation_context(residuals)

    assert context.repair_request_scope_policy == (
        REPAIR_REQUEST_SCOPE_POLICY_VERSION
    )
    assert harness.kernel.repair_request_scope_policy == (
        REPAIR_REQUEST_SCOPE_POLICY_VERSION
    )


def test_training_support_reports_success_anti_trigger_protection_without_new_gate() -> None:
    residuals = (
        ResidualExample(
            transition_id="failure-1",
            task_id="organize-messy-files-1",
            family="organize-messy-files",
            split=SplitName.TRAIN,
            features={"benchmark": "skilllearnbench", "family": "organize-messy-files"},
            failure_type="trajectory_keypoints_missing",
            evaluator_feedback=("missing",),
            baseline_success=False,
        ),
        ResidualExample(
            transition_id="failure-2",
            task_id="organize-messy-files-2",
            family="organize-messy-files",
            split=SplitName.TRAIN,
            features={"benchmark": "skilllearnbench", "family": "organize-messy-files"},
            failure_type="trajectory_keypoints_missing",
            evaluator_feedback=("missing",),
            baseline_success=False,
        ),
        ResidualExample(
            transition_id="success-1",
            task_id="offer-letter-generator-1",
            family="offer-letter-generator",
            split=SplitName.TRAIN,
            features={"benchmark": "skilllearnbench", "family": "offer-letter-generator"},
            failure_type="baseline_success_control",
            evaluator_feedback=(),
            baseline_success=True,
            context={},
        ),
    )
    program_payload = _program_dict()
    program_payload["anti_trigger"] = {
        "all_of": [
            {"key": "family", "op": "eq", "value": "offer-letter-generator"}
        ],
        "any_of": [],
        "none_of": [],
    }
    program = HypothesisProgram.from_dict(program_payload)
    context = ValidationContext(
        evaluator_epoch=program.evaluator_epoch,
        residuals=residuals,
        available_lanes=frozenset({"baseline", "candidate"}),
        baseline_lane="baseline",
        contrastive_training_evidence_policy=(
            CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
        ),
    )

    result = TrainingSupportCheck(min_support=2).evaluate(program, context)

    assert result.passed is True
    assert result.evidence["failure_activation_count"] == 2
    assert result.evidence["success_false_positive_activation_count"] == 0
    assert result.evidence["success_anti_trigger_protection_count"] == 1
    assert result.evidence["failure_anti_trigger_block_count"] == 0


def test_root_repair_and_replay_bind_labeled_success_controls() -> None:
    residuals = (
        ResidualExample(
            transition_id="failure-transition",
            task_id="organize-messy-files-1",
            family="organize-messy-files",
            split=SplitName.TRAIN,
            features={"benchmark": "skilllearnbench", "family": "organize-messy-files"},
            failure_type="trajectory_keypoints_missing",
            evaluator_feedback=("missing",),
            baseline_success=False,
            context={"task_instruction": "organize files"},
        ),
        ResidualExample(
            transition_id="success-transition",
            task_id="offer-letter-generator-1",
            family="offer-letter-generator",
            split=SplitName.TRAIN,
            features={"benchmark": "skilllearnbench", "family": "offer-letter-generator"},
            failure_type="baseline_success_control",
            evaluator_feedback=(),
            baseline_success=True,
            context={},
        ),
    )
    contract = {
        "policy": CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
        "label_field": "baseline_success",
        "failure_label": False,
        "success_control_label": True,
        "success_control_role": "anti_trigger_negative_control",
        "context_may_be_used_for_trigger": False,
        "context_may_shape_actions": True,
    }
    capabilities = {
        "primary_metric": "task_success",
        "training_evidence_contract": contract,
    }
    sink = MemoryEventSink()
    model = QueueProposalModel(
        [
            {"hypotheses": [_program_dict()]},
            {"hypotheses": [_program_dict()]},
        ]
    )
    proposer = StructuredHypothesisProposer(model, event_sink=sink)

    first = proposer.propose(
        residuals,
        evaluator_epoch="skilllearn-eval-epoch-0",
        capabilities=capabilities,
        trace_id="contrastive-root-source",
    )
    replay = proposer.propose(
        residuals,
        evaluator_epoch="skilllearn-eval-epoch-0",
        capabilities=capabilities,
        trace_id="contrastive-root-replay",
    )
    changed_controls = (
        residuals[0],
        replace(
            residuals[1],
            features={**residuals[1].features, "difficulty": "easy"},
        ),
    )
    proposer.propose(
        changed_controls,
        evaluator_epoch="skilllearn-eval-epoch-0",
        capabilities=capabilities,
        trace_id="contrastive-root-changed-control",
    )

    assert replay == first
    assert len(model.requests) == 2
    root_request = model.requests[0]
    assert [row["evidence_label"] for row in root_request["residuals"]] == [
        "failure",
        "success_control",
    ]
    assert root_request["residuals"][1]["context"] == {}
    assert root_request["constraints"][
        "success_rows_are_anti_trigger_negative_controls"
    ] is True
    assert root_request["constraints"][
        "residual_context_may_shape_actions_but_must_not_be_used_in_trigger_or_anti_trigger"
    ] is True
    assert any(
        row["event"] == "root_proposal_evidence_replayed"
        and row["payload"]["new_proposal_model_executions"] == 0
        for row in sink.events
    )

    repair_model = QueueProposalModel([{"hypothesis": _program_dict()}])
    repair_proposer = StructuredHypothesisProposer(repair_model)
    repair_proposer.revise(
        first[0],
        failed_checks=({"check": "training_support", "passed": False},),
        residuals=residuals,
        depth=1,
        capabilities=capabilities,
        trace_id="contrastive-repair",
    )
    repair_request = repair_model.requests[0]
    assert [row["evidence_label"] for row in repair_request["residuals"]] == [
        "failure",
        "success_control",
    ]
    assert repair_request["capabilities"]["training_evidence_contract"] == contract
    assert repair_request["constraints"][
        "success_rows_must_not_increase_failure_trigger_support"
    ] is True


def test_contrastive_policy_versions_are_strictly_paired(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be paired"):
        _harness(
            tmp_path,
            candidate_selection_policy=(
                CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION
            ),
        )


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
    action_contract = model.requests[0]["capabilities"]["action_contract"]
    assert action_contract["allowed_action_operations"] == [
        "check_condition",
        "execute_step",
        "produce_artifact",
        "request_evidence",
    ]
    assert model.requests[0]["constraints"]["allowed_action_operations"] == (
        action_contract["allowed_action_operations"]
    )
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


def test_compiler_runtime_source_receipt_rejects_post_compile_mutation(
    tmp_path: Path,
) -> None:
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    items = adapter.discover()
    manifest = build_family_out_manifest(
        items,
        benchmark="skilllearnbench",
        seed="skilllearnbench-v2-family-out",
    )
    item_id = manifest.validation_ids[0]
    compiled = SkillLearnProgramCompiler().compile(
        programs=(
            HypothesisProgram.from_dict(_program_dict(status="promoted")),
        ),
        items=items,
        split_manifest=manifest,
        output_root=tmp_path,
        target_item_ids=(item_id,),
        target_split="validation",
    )

    receipt = compiled.source_receipt_for(item_id)
    assert receipt.compile_manifest_hash == compiled.manifest_hash
    assert receipt.treatment_hash == compiled.treatment_hash_for(item_id)
    assert receipt.source_file_hashes

    source = compiled.source_for(item_id)
    assert source is not None
    skill_path = next(source.rglob("SKILL.md"))
    skill_path.write_text(
        skill_path.read_text(encoding="utf-8") + "\nmutated\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="content mismatch"):
        compiled.source_receipt_for(item_id)


def test_trial_request_rejects_partial_compile_provenance() -> None:
    common = {
        "item_id": "item-1",
        "family": "family",
        "split": SplitName.TRAIN,
        "variant": TrialVariant.POLICY_ON,
        "evaluator_epoch": "epoch",
        "pair_id": "pair",
        "repeat": 1,
        "agent_id": "codex",
        "model": "gpt-5.4-mini",
        "max_steps": 100,
        "manifest_hash": "manifest",
    }
    with pytest.raises(ValueError, match="item source receipt"):
        SkillLearnTrialRequest(
            **common,
            compile_manifest_hash="compile",
        )
    with pytest.raises(ValueError, match="typed compile provenance"):
        SkillLearnTrialRequest(
            **common,
            compile_manifest_hash="compile",
            skill_source_receipt_hash="receipt",
            typed_binding_set_hash="binding-set",
        )
    with pytest.raises(ValueError, match="policy-on treatment identity"):
        SkillLearnTrialRequest(
            **common,
            program_id="external-control",
            program_set_hash="program-set",
            treatment_hash=NO_SKILL_TREATMENT_HASH,
            external_skill_source_receipt_hash="external-receipt",
        )
    with pytest.raises(ValueError, match="policy-on treatment identity"):
        SkillLearnTrialRequest(
            **{
                **common,
                "variant": TrialVariant.POLICY_OFF,
            },
            program_id="external-control",
            program_set_hash="program-set",
            treatment_hash="treatment",
            external_skill_source_receipt_hash="external-receipt",
        )


def test_runtime_treatment_adapter_fails_closed_on_partial_injection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    skill = source / "frozen-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: frozen-skill\n---\n# Frozen\n", encoding="utf-8")
    asset = source / "frozen-skill" / "assets" / "reference.bin"
    asset.parent.mkdir()
    asset.write_bytes(b"\x00\x01frozen-reference\xff")
    copies = [
        ("skills", "/root/.codex/skills"),
        ("skills", "/root/.agents/skills"),
    ]

    class ReadbackSubprocess:
        def __init__(self) -> None:
            self.destinations: dict[str, Path] = {}

        def run(self, args, **kwargs):
            command = list(args)
            assert command[:2] == ["docker", "cp"]
            container_path = command[2].split(":", 1)[1]
            destination = container_path.removesuffix("/.")
            installed_source = self.destinations.get(destination)
            if installed_source is None:
                return SimpleNamespace(returncode=1, stdout="", stderr="missing")
            readback = Path(command[3])
            for child in installed_source.iterdir():
                target = readback / child.name
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    def runner_with_injected_count(count: int) -> tuple[ModuleType, ReadbackSubprocess]:
        runner = ModuleType(f"receipt_runner_{count}")
        delegate = ReadbackSubprocess()

        def copy_skills_to_dest(src: Path, destination: Path) -> bool:
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)
            directory_skills = [
                path
                for path in src.iterdir()
                if path.is_dir() and (path / "SKILL.md").is_file()
            ]
            if directory_skills:
                for path in directory_skills:
                    shutil.copytree(path, destination / path.name)
                return True
            if (src / "SKILL.md").is_file():
                target = destination / src.name
                target.mkdir()
                shutil.copy2(src / "SKILL.md", target / "SKILL.md")
                return True
            markdown_files = sorted(src.glob("*.md"))
            if markdown_files:
                for path in markdown_files:
                    target = destination / path.stem.replace("_", "-")
                    target.mkdir()
                    shutil.copy2(path, target / "SKILL.md")
                return True
            return False

        def original_inject(container_name, skill_source_dir, destinations):
            for _, destination in destinations[:count]:
                delegate.destinations[destination] = Path(skill_source_dir)

        runner.subprocess = delegate
        runner._copy_skills_to_dest = copy_skills_to_dest
        runner._inject_skills_runtime = original_inject
        backend = SkillLearnSubprocessBackend(
            BENCH_ROOT,
            model="gpt-5.4-mini",
            provider_mode="openai_compatible",
        )
        backend._install_treatment_receipt_adapter(runner)
        return runner, delegate

    partial, _ = runner_with_injected_count(1)
    with pytest.raises(RuntimeError, match="installed_treatment_receipt_invalid"):
        partial._inject_skills_runtime("trial", source, copies)
    assert partial._assumption_v2_installed_skill_receipt is None

    complete, _ = runner_with_injected_count(2)
    complete._inject_skills_runtime("trial", source, copies)
    receipt = complete._assumption_v2_installed_skill_receipt
    assert receipt["destination_count"] == 2
    assert receipt["source_file_hashes"] == (
        (
            "frozen-skill/SKILL.md",
            hashlib.sha256(skill.read_bytes()).hexdigest(),
        ),
        (
            "frozen-skill/assets/reference.bin",
            hashlib.sha256(asset.read_bytes()).hexdigest(),
        ),
    )
    (source / "frozen-skill" / "package.json").write_text(
        '{"name":"unreceipted-runtime-install"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError,
        match="installed_treatment_receipt_invalid",
    ):
        complete._inject_skills_runtime("trial", source, copies)
    assert complete._assumption_v2_installed_skill_receipt is None


@pytest.mark.parametrize("layout", ("root_skill", "flat_markdown"))
def test_runtime_treatment_adapter_matches_upstream_normalization(
    tmp_path: Path,
    layout: str,
) -> None:
    source = tmp_path / "normalized-family"
    source.mkdir()
    if layout == "root_skill":
        (source / "SKILL.md").write_text(
            "---\nname: normalized-family\n---\n# Root skill\n",
            encoding="utf-8",
        )
        expected_paths = ("normalized-family/SKILL.md",)
    else:
        (source / "first_skill.md").write_text(
            "# First\n",
            encoding="utf-8",
        )
        (source / "second.md").write_text(
            "# Second\n",
            encoding="utf-8",
        )
        expected_paths = (
            "first-skill/SKILL.md",
            "second/SKILL.md",
        )

    def copy_skills_to_dest(src: Path, destination: Path) -> bool:
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        directory_skills = [
            path
            for path in src.iterdir()
            if path.is_dir() and (path / "SKILL.md").is_file()
        ]
        if directory_skills:
            for path in directory_skills:
                shutil.copytree(path, destination / path.name)
            return True
        if (src / "SKILL.md").is_file():
            target = destination / src.name
            target.mkdir()
            shutil.copy2(src / "SKILL.md", target / "SKILL.md")
            return True
        markdown_files = sorted(src.glob("*.md"))
        if not markdown_files:
            return False
        for path in markdown_files:
            target = destination / path.stem.replace("_", "-")
            target.mkdir()
            shutil.copy2(path, target / "SKILL.md")
        return True

    class NormalizedReadback:
        def __init__(self) -> None:
            self.destinations: dict[str, Path] = {}

        def run(self, args, **kwargs):
            command = list(args)
            container_path = command[2].split(":", 1)[1]
            destination = container_path.removesuffix("/.")
            installed_source = self.destinations[destination]
            readback = Path(command[3])
            for child in installed_source.iterdir():
                target = readback / child.name
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    runner = ModuleType(f"normalized_receipt_runner_{layout}")
    delegate = NormalizedReadback()
    runner.subprocess = delegate
    runner._copy_skills_to_dest = copy_skills_to_dest

    def original_inject(container_name, skill_source_dir, destinations):
        for index, (_, destination) in enumerate(destinations):
            installed = tmp_path / f"installed-{layout}-{index}"
            assert copy_skills_to_dest(Path(skill_source_dir), installed)
            delegate.destinations[destination] = installed

    runner._inject_skills_runtime = original_inject
    backend = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        model="gpt-5.4-mini",
        provider_mode="openai_compatible",
    )
    backend._install_treatment_receipt_adapter(runner)
    runner._inject_skills_runtime(
        "trial",
        source,
        [
            ("skills", "/root/.codex/skills"),
            ("skills", "/root/.agents/skills"),
        ],
    )

    receipt = runner._assumption_v2_installed_skill_receipt
    assert receipt["destination_count"] == 2
    for destination in delegate.destinations.values():
        assert tuple(
            sorted(
                path.relative_to(destination).as_posix()
                for path in destination.rglob("*")
                if path.is_file()
            )
        ) == expected_paths


def test_low_reasoning_local_compaction_policy_renders_exact_codex_cli_values() -> None:
    assert LOW_REASONING_LOCAL_COMPACTION_POLICY.codex_cli_values() == (
        "--config",
        'model_reasoning_effort="low"',
        "--config",
        'model_verbosity="low"',
        "--config",
        "model_auto_compact_token_limit=32768",
        "--config",
        'model_auto_compact_token_limit_scope="body_after_prefix"',
        "--config",
        "tool_output_token_limit=10000",
        "--enable",
        "enable_request_compression",
        "--disable",
        "remote_compaction_v2",
    )


def test_agent_execution_policy_changes_request_and_execution_fingerprints() -> None:
    request = SkillLearnTrialRequest(
        item_id="item",
        family="family",
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="epoch",
        pair_id="pair",
        repeat=1,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash="manifest",
    )
    compact_request = replace(
        request,
        codex_agent_execution_policy_hash=(
            LOW_REASONING_LOCAL_COMPACTION_POLICY.policy_hash
        ),
    )
    model_only_request = replace(
        request,
        codex_agent_execution_policy_hash=(
            MODEL_ONLY_ACTION_BUDGET_POLICY.policy_hash
        ),
    )
    assert len(
        {request.request_hash, compact_request.request_hash, model_only_request.request_hash}
    ) == 3

    legacy_provider = _provider_fingerprint(
        "codex",
        "gpt-5.4-mini",
        "openai_compatible",
        LEGACY_CODEX_AGENT_EXECUTION_POLICY,
    )
    compact_provider = _provider_fingerprint(
        "codex",
        "gpt-5.4-mini",
        "openai_compatible",
        LOW_REASONING_LOCAL_COMPACTION_POLICY,
    )
    model_only_provider = _provider_fingerprint(
        "codex",
        "gpt-5.4-mini",
        "openai_compatible",
        MODEL_ONLY_ACTION_BUDGET_POLICY,
    )
    assert len({legacy_provider, compact_provider, model_only_provider}) == 3
    common = {
        "agent_id": "codex",
        "model": "gpt-5.4-mini",
        "provider_mode": "openai_compatible",
        "max_steps": 100,
        "prebuilt_enabled": True,
        "agent_runtime_key": "runtime",
        "prebuilt_image_key": "image",
        "prebuilt_image_id": "sha256:image",
        "offline_verifier_runtime_key": "verifier",
    }
    assert _fairness_fingerprint(
        **common,
        provider_fingerprint=legacy_provider,
        codex_agent_execution_policy=LEGACY_CODEX_AGENT_EXECUTION_POLICY,
    ) != _fairness_fingerprint(
        **common,
        provider_fingerprint=model_only_provider,
        codex_agent_execution_policy=MODEL_ONLY_ACTION_BUDGET_POLICY,
    )


def test_action_budget_cost_accounting_never_mixes_tokens_and_steps() -> None:
    request = SkillLearnTrialRequest(
        item_id="item",
        family="family",
        split=SplitName.VALIDATION,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="epoch",
        pair_id="pair",
        repeat=1,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash="manifest",
        codex_agent_execution_policy_hash=(
            MODEL_ONLY_ACTION_BUDGET_POLICY.policy_hash
        ),
    )
    complete = SkillLearnTrialObservation(
        request=request,
        success=True,
        score=1.0,
        metrics={"evaluation_valid": 1.0},
        total_tokens=250_000,
        steps=20,
        duration_seconds=1.0,
        provider_fingerprint="provider",
        fairness_fingerprint="fairness",
        step_budget_policy=str(
            MODEL_ONLY_ACTION_BUDGET_POLICY.action_budget_policy
        ),
        step_budget_unit=str(MODEL_ONLY_ACTION_BUDGET_POLICY.action_budget_unit),
        step_budget_limit=100,
        step_budget_token_usage_complete=True,
        step_budget_receipt_hash="a" * 64,
    )
    truncated = replace(
        complete,
        total_tokens=0,
        steps=100,
        step_budget_truncated=True,
        step_budget_token_usage_complete=False,
    )

    assert complete.cost_units == 20.0
    assert truncated.cost_units == 100.0


def test_backend_pool_rejects_mixed_agent_execution_policies() -> None:
    legacy = SkillLearnSubprocessBackend(BENCH_ROOT)
    compact = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        codex_agent_execution_policy=LOW_REASONING_LOCAL_COMPACTION_POLICY,
    )

    with pytest.raises(ValueError, match="frozen configuration"):
        SkillLearnBackendPool((legacy, compact))


def test_openai_compatible_trial_compiles_sanitized_codex_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "provider-secret-must-not-enter-command-or-events"
    monkeypatch.setenv("ASSUMPTION_V2_API_KEY", secret)
    monkeypatch.setenv("ASSUMPTION_V2_API_BASE", "https://ruoli.dev")
    monkeypatch.setenv("ASSUMPTION_V2_API_ALLOWED_IPV4S", "45.78.76.197")
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
        codex_agent_execution_policy=MODEL_ONLY_ACTION_BUDGET_POLICY,
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
        assert "--strict-config" in run_template
        assert "tools.web_search=false" not in run_template
        assert 'web_search="disabled"' in run_template
        assert "codex-action-supervisor.mjs" in run_template
        assert "/opt/assumption-v2-agent/bin/node" in run_template
        assert "/opt/assumption-v2-agent/bin/codex exec" in run_template
        assert (
            "env PATH=/opt/assumption-v2-agent/bin:/usr/local/bin:/usr/bin:/bin"
            in run_template
        )
        assert "--limit 100" in run_template
        assert "--trace /logs/agent/codex.txt" in run_template
        assert "--process-scope dedicated_container" in run_template
        assert "rm -f /logs/agent/codex_action_budget_receipt.json" in run_template
        assert agent["trajectory_tee"] is None
        assert 'model_reasoning_effort="low"' in run_template
        assert 'model_verbosity="low"' in run_template
        assert "model_auto_compact_token_limit=32768" in run_template
        assert 'model_auto_compact_token_limit_scope="body_after_prefix"' in run_template
        assert "tool_output_token_limit=10000" in run_template
        assert "enable_request_compression" in run_template
        assert "remote_compaction_v2" in run_template
        assert "image_generation" in run_template
        assert "standalone_web_search" in run_template
        assert "package installation" in run_template
        assert 'model_provider="assumption_v2_openai_compatible"' in run_template
        assert "https://ruoli.dev/v1" in run_template
        assert "wire_api" in run_template and "responses" in run_template
        assert "supports_websockets=false" in run_template
        assert "requires_openai_auth=false" in run_template
        assert "api.openai.com" not in run_template
        assert secret not in run_template
        runner.subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "trial",
                "image",
                "sleep",
                "3600",
            ]
        )
        command = delegate.commands[-1]
        assert command[0:2] == ["docker", "run"]
        assert command[command.index("--network") + 1] == "assumption-v2-restricted"
        assert command[command.index("--dns") + 1] == "127.0.0.1"
        assert "ruoli.dev:45.78.76.197" in command
        assert "PIP_NO_INDEX=1" in command
        assert command[command.index("--pull") + 1] == "never"
        assert secret not in repr(command)
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
        assert prepared["payload"]["web_search_mode"] == "disabled"
        assert prepared["payload"]["web_search_enabled"] is False
        assert prepared["payload"]["action_budget_limit"] == 100
        assert prepared["payload"]["action_budget_cost_accounting_policy"] == (
            "uniform_codex_action_start_cost_v1"
        )
        assert prepared["payload"]["action_budget_process_scope"] == (
            "dedicated_container"
        )
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


def test_poster_verifier_uses_readonly_local_runtime_and_skips_online_script(
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("curl https://astral.sh | sh\n", encoding="utf-8")
    (tests_dir / "test_outputs.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    runtime = OfflineVerifierRuntime(
        profile=POSTER_VERIFIER_PROFILE,
        runtime_key="runtime-key",
        volume_name="offline-verifier-volume",
        base_image_id="sha256:base",
        reused=True,
    )
    delegate = RecordingSubprocess()
    sink = MemoryEventSink()
    proxy = _DockerVerifierIsolationSubprocessProxy(
        delegate,
        offline_verifier_runtime=runtime,
        egress_policy=DockerEgressPolicy.from_values(
            base_url="https://ruoli.dev",
            allowed_ipv4s=("45.78.76.197",),
        ),
        event_sink=sink,
        trace_id="offline-poster",
    )

    proxy.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            "poster-trial",
            "-v",
            f"{tests_dir}:/tests:ro",
            "image",
            "sleep",
            "3600",
        ]
    )
    run_command = delegate.commands[-1]
    assert f"offline-verifier-volume:{OFFLINE_VERIFIER_MOUNT}:ro" in run_command
    assert f"{tests_dir}:/tests:ro" not in run_command

    proxy.run(
        ["docker", "exec", "poster-trial", "bash", "/tests/test.sh"],
        timeout=1800,
    )
    verifier_command = delegate.commands[-1]
    assert verifier_command[:5] == [
        "docker",
        "exec",
        "poster-trial",
        "sh",
        "-lc",
    ]
    assert "python3 -m pytest" in verifier_command[-1]
    assert "PIP_NO_INDEX=1" in verifier_command[-1]
    assert "RESULTS_PATH=/root/results.json" in verifier_command[-1]
    assert "security_audit.csv" in verifier_command[-1]
    assert "itinerary.json" in verifier_command[-1]
    assert "/tests/test.sh" not in verifier_command[-1]
    assert any(
        row["event"] == "skilllearn_offline_verifier_command_selected"
        and row["payload"]["original_online_test_script_executed"] is False
        for row in sink.events
    )


def test_shared_model_slot_serializes_agents_but_not_verifier() -> None:
    delegate = BlockingAgentSubprocess()
    limiter = SkillLearnModelInferenceLimiter(1)
    sink = MemoryEventSink()
    egress = DockerEgressPolicy.from_values(
        base_url="https://ruoli.dev",
        allowed_ipv4s=("45.78.76.197",),
    )
    first = _DockerVerifierIsolationSubprocessProxy(
        delegate,
        egress_policy=egress,
        model_inference_limiter=limiter,
        event_sink=sink,
        trace_id="slot-first",
    )
    second = _DockerVerifierIsolationSubprocessProxy(
        delegate,
        egress_policy=egress,
        model_inference_limiter=limiter,
        event_sink=sink,
        trace_id="slot-second",
    )
    command = ["docker", "exec", "trial", "sh", "-c", "codex exec --help"]
    first_thread = threading.Thread(target=lambda: first.run(command))
    second_thread = threading.Thread(target=lambda: second.run(command))

    first_thread.start()
    assert delegate.agent_entered.wait(timeout=1.0)
    second_thread.start()
    second.run(["docker", "exec", "verifier", "bash", "/tests/test.sh"])

    assert delegate.verifier_entered.is_set()
    delegate.release_agents.set()
    first_thread.join(timeout=2.0)
    second_thread.join(timeout=2.0)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert delegate.maximum_active_agents == 1
    assert limiter.maximum_active == 1
    assert sum(
        row["event"] == "skilllearn_agent_slot_acquired" for row in sink.events
    ) == 2
    assert sum(
        row["event"] == "skilllearn_agent_slot_released" for row in sink.events
    ) == 2


def test_shared_model_slot_releases_after_agent_exception() -> None:
    delegate = BlockingAgentSubprocess(fail_first=True)
    limiter = SkillLearnModelInferenceLimiter(1)
    sink = MemoryEventSink()
    proxy = _DockerVerifierIsolationSubprocessProxy(
        delegate,
        egress_policy=DockerEgressPolicy.from_values(
            base_url="https://ruoli.dev",
            allowed_ipv4s=("45.78.76.197",),
        ),
        model_inference_limiter=limiter,
        event_sink=sink,
        trace_id="slot-exception",
    )
    command = ["docker", "exec", "trial", "sh", "-c", "codex exec --help"]

    with pytest.raises(RuntimeError, match="synthetic agent failure"):
        proxy.run(command)
    delegate.release_agents.set()
    proxy.run(command)

    assert delegate.agent_calls == 2
    assert limiter.maximum_active == 1
    assert MODEL_INFERENCE_CONCURRENCY_POLICY_VERSION == limiter.policy
    assert sum(
        row["event"] == "skilllearn_agent_slot_released" for row in sink.events
    ) == 2


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


def test_complete_codex_trace_provider_failure_precedes_invalid_action_receipt(
    tmp_path: Path,
) -> None:
    family = "temperature-simulation"
    item_id = "temperature-simulation-3"
    benchmark_root = tmp_path / "benchmark"
    trials_dir = tmp_path / "trials"
    test_script = (
        benchmark_root / "tasks" / family / item_id / "tests" / "test.sh"
    )
    test_script.parent.mkdir(parents=True)
    test_script.write_text("python3 /tests/test_outputs.py\n", encoding="utf-8")
    request = SkillLearnTrialRequest(
        item_id=item_id,
        family=family,
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="epoch-offline",
        pair_id="full-trace-provider-error",
        repeat=0,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash="manifest-offline",
    )
    trial_dir = trials_dir / "no_skill" / family / item_id / request.trial_id
    agent_dir = trial_dir / "agent"
    verifier_dir = trial_dir / "verifier"
    agent_dir.mkdir(parents=True)
    verifier_dir.mkdir()
    terminal_row = json.dumps(
        {
            "type": "turn.failed",
            "error": {
                "message": "exceeded retry limit, last status: 429 Too Many Requests"
            },
        }
    )
    generic_error_row = json.dumps(
        {"type": "error", "message": "stream disconnected"}
    )
    (agent_dir / "codex.txt").write_text(
        ("non-json-prefix" * 256)
        + "\n"
        + generic_error_row
        + "\n"
        + terminal_row
        + "\n",
        encoding="utf-8",
    )
    (verifier_dir / "reward.txt").write_text("0\n", encoding="utf-8")
    (verifier_dir / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "summary": {
                        "tests": 1,
                        "passed": 0,
                        "failed": 1,
                        "skipped": 0,
                        "pending": 0,
                        "other": 0,
                    },
                    "tests": [{"name": "test_failure", "status": "failed"}],
                }
            }
        ),
        encoding="utf-8",
    )
    sink = MemoryEventSink()
    backend = SkillLearnSubprocessBackend(
        benchmark_root,
        trials_dir=trials_dir,
        provider_mode="openai_compatible",
        codex_agent_execution_policy=MODEL_ONLY_ACTION_BUDGET_POLICY,
        event_sink=sink,
    )
    upstream_result = {
        "passed": False,
        "reward": 0,
        "verifier_exit": 0,
        "agent_stdout": "non-json-prefix" * 128,
        "agent_stderr": "",
    }

    audited = backend._audit_trial_artifacts(
        runner=SimpleNamespace(TRIALS_DIR=str(trials_dir)),
        request=request,
        skill_config="no_skill",
        result=upstream_result,
        offline_verifier_profile=COMMON_PY38_VERIFIER_PROFILE,
        trace_id="full-trace-provider-error",
    )
    observation = backend._sanitize_result(
        request,
        result=audited,
        return_code=1,
        duration_seconds=1.0,
    )

    assert audited["step_budget_receipt_valid"] is False
    assert audited["error"] == "provider_rate_limit"
    assert audited["model_terminal_error"] == "provider_rate_limit"
    assert observation.error_type == "provider_rate_limit"
    assert backend.provider_circuit.open(observation.error_type) is True
    assert any(
        row["event"] == "skilllearn_agent_terminal_error_detected"
        and row["payload"]["source"] == "complete_codex_trace"
        and row["payload"]["error_type"] == "provider_rate_limit"
        for row in sink.events
    )


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
            "provider_usage_limit",
        ),
        (
            {
                "agent_timed_out": False,
                "verifier_exit": 0,
                "agent_stdout": json.dumps(
                    {
                        "type": "turn.failed",
                        "error": {
                            "message": (
                                "503 Service Unavailable: model has no available "
                                "distributor channel"
                            )
                        },
                    }
                ),
            },
            "provider_model_unavailable",
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
        model="gpt-5.4-mini",
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
    assert circuit.open("provider_usage_limit") is True
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
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash="manifest",
    )

    observation = backend.run(request, skill_source_dir=None, trace_id="circuit-open")

    assert observation.valid is False
    assert observation.error_type == "provider_circuit_open_provider_usage_limit"


def test_loaded_runners_isolate_agent_registry_across_parallel_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASSUMPTION_V2_API_KEY", "fake-secret")
    monkeypatch.setenv("ASSUMPTION_V2_API_BASE", "https://ruoli.dev")
    monkeypatch.setenv("ASSUMPTION_V2_API_ALLOWED_IPV4S", "45.78.76.197")
    first = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        provider_mode="openai_compatible",
        record_upstream=False,
    )
    second = SkillLearnSubprocessBackend(
        BENCH_ROOT,
        provider_mode="openai_compatible",
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
                assert agent["env"] == ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
                assert agent["setup"] is None
                assert "codex exec" in agent["run"]
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
    cache = SkillLearnPrebuiltImageCache(
        benchmark,
        cache_only=False,
        event_sink=sink,
    )

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
    oracle_changed = SkillLearnPrebuiltImageCache(
        benchmark,
        cache_only=False,
    ).ensure(
        family="family",
        item_id="item-1",
        agent_id="codex",
        runner=runner,
        trace_id="prebuild-oracle-change",
    )
    assert oracle_changed.cache_key == first.cache_key
    assert oracle_changed.reused is True

    (environment / "payload.txt").write_text("version-two\n", encoding="utf-8")
    payload_changed = SkillLearnPrebuiltImageCache(
        benchmark,
        cache_only=False,
    ).ensure(
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
        parallel_workers=6,
    )
    train_ids = tuple(harness.manifest.train_ids[:6])

    observations = harness.collect_training_observations(
        train_item_ids=train_ids,
        trace_id="parallel-training",
    )

    assert tuple(row.request.item_id for row in observations) == train_ids
    assert backend.maximum_active == 6


def test_counterfactual_parallelism_is_cross_item_and_pair_sequential(
    tmp_path: Path,
) -> None:
    backend = ConcurrentFakeBackend()
    harness, _, _, _, _, _ = _harness(
        tmp_path,
        backend_override=backend,
        parallel_workers=6,
    )
    validation_ids = tuple(harness.manifest.validation_ids[:6])
    program = HypothesisProgram.from_dict(_program_dict())

    pairs = harness.counterfactual_runner.run(
        harness.tasks(validation_ids),
        program=program,
        split=SplitName.VALIDATION,
        trace_id="parallel-counterfactual",
    )

    assert tuple(pair.task_id for pair in pairs) == validation_ids
    assert backend.maximum_active == 6
    assert all(
        backend.maximum_active_by_item[item_id] == 1
        for item_id in validation_ids
    )
    assert len(backend.calls) == 12


def _harness(
    tmp_path: Path,
    *,
    invalid_candidate_item: str | None = None,
    invalid_training_item: str | None = None,
    backend_override=None,
    parallel_workers: int = 1,
    invalid_trial_max_attempts: int = 1,
    invalid_trial_retry_backoff_seconds: float = 0.0,
    invalid_trial_retry_workers: int = 1,
    proposal_rows: list[dict[str, Any]] | None = None,
    candidate_selection_policy: str = TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
    candidate_bundle_policy: str | None = None,
    contrastive_training_evidence_policy: str | None = None,
    train_action_design_policy: str | None = None,
    proposal_formation_policy: str | None = None,
    repair_request_scope_policy: str | None = None,
    baseline_arm_replay_cache: BaselineArmEvidenceReplayCache | None = None,
    baseline_arm_evidence_replay_policy: str | None = None,
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
        (
            [
                {"hypothesis": row}
                for row in (proposal_rows or [_program_dict()] * 3)
            ]
            if proposal_formation_policy
            in {
                PROPOSAL_FORMATION_POLICY_VERSION,
                PROPOSAL_FORMATION_POLICY_V2,
            }
            else [{"hypotheses": proposal_rows or [_program_dict()]}]
        )
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
                metric="task_success",
                minimum_pairs=2,
                confidence=0.9,
                minimum_net_gain_count=1,
                minimum_activation_rate=1.0,
                minimum_effect_lower_bound=0.0,
                maximum_harm_rate=0.05,
                maximum_cost_ratio=1.5,
                baseline_safety_policy=PROSPECTIVE_ABSTENTION_PAIRED_GUARD,
                candidate_threshold_policy=CANDIDATE_MAY_ONLY_TIGHTEN,
            ),
            event_sink=sink,
        ),
        archive=archive,
        evaluator_epoch="skilllearn-eval-epoch-0",
        output_root=tmp_path / "compiled",
        candidate_selection_policy=candidate_selection_policy,
        proposal_formation_policy=proposal_formation_policy,
        candidate_bundle_policy=candidate_bundle_policy,
        contrastive_training_evidence_policy=(
            contrastive_training_evidence_policy
        ),
        train_action_design_policy=train_action_design_policy,
        repair_request_scope_policy=repair_request_scope_policy,
        parallel_workers=parallel_workers,
        invalid_trial_max_attempts=invalid_trial_max_attempts,
        invalid_trial_retry_backoff_seconds=(
            invalid_trial_retry_backoff_seconds
        ),
        invalid_trial_retry_workers=invalid_trial_retry_workers,
        baseline_arm_replay_cache=baseline_arm_replay_cache,
        **(
            {
                "baseline_arm_evidence_replay_policy": (
                    baseline_arm_evidence_replay_policy
                )
            }
            if baseline_arm_evidence_replay_policy is not None
            else {}
        ),
        event_sink=sink,
    )
    return harness, backend, model, archive, guard, sink


def _family_residual(
    transition_id: str,
    *,
    family: str,
    baseline_success: bool = False,
) -> ResidualExample:
    return ResidualExample(
        transition_id=transition_id,
        task_id=transition_id,
        family=family,
        split=SplitName.TRAIN,
        features={"benchmark": "skilllearnbench", "family": family},
        failure_type=(
            "baseline_success_control"
            if baseline_success
            else "trajectory_keypoints_missing"
        ),
        evaluator_feedback=() if baseline_success else ("missing",),
        baseline_success=baseline_success,
        context={},
    )


def _family_trigger_program(
    program_id: str,
    families: tuple[str, ...],
) -> HypothesisProgram:
    payload = _program_dict()
    payload["id"] = program_id
    payload["trigger"] = {
        "all_of": [
            {"key": "benchmark", "op": "eq", "value": "skilllearnbench"}
        ],
        "any_of": [
            {"key": "family", "op": "eq", "value": family}
            for family in families
        ],
        "none_of": [],
    }
    return HypothesisProgram.from_dict(payload)


def _family_slot_profile(family: str) -> dict[str, Any]:
    return {
        "policy": TRAIN_ACTION_DESIGN_POLICY_VERSION,
        "runtime_environment": {
            "declared_os_packages": [f"{family}-env"],
            "declared_python_packages": [],
            "declared_task_local_paths": [f"/root/{family}.json"],
            "copied_task_files": [f"/root/{family}.json"],
            "environment_source_files": [],
        },
        "baseline_action_trace": {
            "command_signatures": [
                {
                    "executable_basename": f"{family}-tool",
                    "safe_flags": ["--offline"],
                    "task_local_paths": [f"/root/{family}.json"],
                    "original_command_hash": f"{family}-success-hash",
                    "status": "succeeded",
                    "exit_code": 0,
                },
                {
                    "executable_basename": f"{family}-broken",
                    "safe_flags": [],
                    "task_local_paths": [],
                    "original_command_hash": f"{family}-failed-hash",
                    "status": "failed",
                    "exit_code": 1,
                },
            ]
        },
    }


def _family_slot_residual(
    transition_id: str,
    *,
    family: str,
    profile_hash: str,
) -> ResidualExample:
    return replace(
        _family_residual(transition_id, family=family),
        context={
            "task_instruction": (
                f"Process the current {family} task artifact offline."
            ),
            "action_context_profile_hash": profile_hash,
        },
    )


def _family_slot_program(family: str) -> HypothesisProgram:
    payload = _family_trigger_program(
        f"{family}-profile-grounded",
        (family,),
    ).to_dict()
    payload["action_graph"] = [
        {
            "id": "portable-profile-recipe",
            "operation": "execute_step",
            "target": "task_procedure",
            "value": (
                f"Run {family}-tool --offline on /root/{family}.json and "
                "parse the current task artifact file."
            ),
            "depends_on": [],
        }
    ]
    return HypothesisProgram.from_dict(payload)


def _family_slot_v2_program(
    family: str,
    *,
    include_failed_primitive: bool = False,
) -> HypothesisProgram:
    payload = _family_slot_program(family).to_dict()
    payload["trigger"] = {
        "all_of": [{"key": "family", "op": "eq", "value": family}],
        "any_of": [],
        "none_of": [],
    }
    payload["anti_trigger"] = {
        "all_of": [],
        "any_of": [],
        "none_of": [],
    }
    failed_suffix = (
        f" Never invoke {family}-broken."
        if include_failed_primitive
        else ""
    )
    payload["action_graph"][0]["value"] = (
        f"Read the current artifact at /root/{family}.json, parse it with a "
        "preinstalled local parser, update the task-required content in "
        "memory, serialize it in the original format, and write the result "
        f"back to /root/{family}.json.{failed_suffix}"
    )
    return HypothesisProgram.from_dict(payload)


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
                "id": "execute",
                "operation": "execute_step",
                "target": "task_procedure",
                "value": "Translate every explicit task requirement into an ordered execution checklist.",
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
