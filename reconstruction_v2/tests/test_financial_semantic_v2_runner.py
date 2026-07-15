from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assumption_agent.benchmarks.codex_action_budget import (
    CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnTrialObservation,
)
from assumption_agent.models import stable_hash
from replication_runtime.financial_semantic_v2.backends import (
    DurableFinancialSemanticSubprocessBackendV2,
    DurableRawSubprocessBackendV2,
)
from replication_runtime.financial_semantic_v2.plan import (
    FixedPeriodOutTreatmentV2,
    MeasurementTargetV2,
    build_measurement_plan_v2,
)
from replication_runtime.financial_semantic_v2.recovery import (
    RecoveryDecisionV2,
)
from replication_runtime.financial_semantic_v2 import runner


def _sha(label: str) -> str:
    return stable_hash({"label": label})


def _plan():
    recipe_id = _sha("recipe")
    treatment = FixedPeriodOutTreatmentV2(
        recipe_id=recipe_id,
        program_set_hash=stable_hash({"recipe_ids": [recipe_id]}),
        period_out_treatment_id=_sha("treatment"),
        external_skill_source_receipt_hash=_sha("source"),
        candidate_skill_source=Path("/frozen/candidate"),
    )
    return build_measurement_plan_v2(
        targets=tuple(
            MeasurementTargetV2(
                item_id=f"period-out-{index}",
                fold_id=f"fold-{index}",
            )
            for index in range(8)
        ),
        manifest_hash=_sha("manifest"),
        evaluator_epoch="period-out-test",
        treatment=treatment,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        codex_agent_execution_policy_hash=_sha("execution-policy"),
    )


def _decision(
    status: str,
    *,
    completed: bool = False,
    may_claim: bool = False,
    error_type: str | None = None,
) -> RecoveryDecisionV2:
    return RecoveryDecisionV2(
        status=status,
        error_type=error_type,
        recovery_action=None,
        current_stage=(
            "observation_finalized" if completed else "semantic_plan_ready"
        ),
        transitions_applied=(),
        model_calls_accounted=1 if completed or error_type else 0,
        model_replay_authorized=False,
        may_claim_clean_model_execution=may_claim,
        completed=completed,
    )


def _observation(work) -> SkillLearnTrialObservation:
    return SkillLearnTrialObservation(
        request=work.request,
        success=True,
        score=1.0,
        metrics={"evaluation_valid": 1.0},
        total_tokens=10,
        steps=1,
        duration_seconds=0.5,
        provider_fingerprint="provider",
        fairness_fingerprint="fairness",
        raw_trial_artifacts_persisted=True,
        prebuilt_image_key="image-key",
        prebuilt_image_id="image-id",
        prebuilt_cache_reused=True,
        agent_runtime_key="agent-key",
        agent_runtime_version="agent-v1",
        offline_verifier_profile_id="offline-profile",
        offline_verifier_runtime_key="offline-runtime",
        step_budget_policy="action-starts-v1",
        step_budget_unit="action_started",
        step_budget_limit=100,
        step_budget_token_usage_complete=True,
        step_budget_receipt_hash=_sha("budget-receipt"),
        installed_skill_source_receipt_hash="",
        runtime_profile_prompt_delivery_policy="none",
        runtime_profile_prompt_injection_receipt_hash="",
        runtime_profile_effective_prompt_sha256="",
    )


class _RawDelegate(DurableRawSubprocessBackendV2):
    def __init__(self, callback) -> None:
        self.callback = callback

    def run(self, request, *, skill_source_dir, trace_id):
        return self.callback(request, skill_source_dir, trace_id)


class _CandidateDelegate(DurableFinancialSemanticSubprocessBackendV2):
    def __init__(self, callback) -> None:
        self.callback = callback

    def run(self, request, *, skill_source_dir, trace_id):
        return self.callback(request, skill_source_dir, trace_id)


def _raw_wrapper(tmp_path: Path, delegate: _RawDelegate):
    work = next(work for work in _plan().work_units if work.arm == "raw")
    return work, runner.RecoveryBoundBackendV2(
        delegate=delegate,
        work=work,
        state_root=tmp_path / "state",
        trial_root=tmp_path / work.request.trial_id,
        expected_process_scope=(
            CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER
        ),
    )


def test_claim_is_immediately_before_the_only_backend_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    clean = _decision("clean_never_started", may_claim=True)
    completed = _decision("completed", completed=True)
    decisions = iter((clean, completed))
    work_holder: list[object] = []

    def execute(request, skill_source_dir, trace_id):
        events.append("backend")
        return _observation(work_holder[0])

    delegate = _RawDelegate(execute)
    work, wrapped = _raw_wrapper(tmp_path, delegate)
    work_holder.append(work)
    monkeypatch.setattr(
        runner,
        "recover_existing_artifacts_without_model_v2",
        lambda **_kwargs: events.append("recovery") or next(decisions),
    )
    monkeypatch.setattr(
        runner,
        "authorize_clean_model_execution_once_v2",
        lambda **_kwargs: events.append("claim"),
    )
    monkeypatch.setattr(
        runner,
        "load_completed_observation_without_model_v2",
        lambda **_kwargs: (
            events.append("load") or completed,
            _observation(work).to_dict(),
        ),
    )

    observed = wrapped.run(
        work.request,
        skill_source_dir=None,
        trace_id="financial-semantic-v2:" + work.work_unit_hash[:20],
    )

    assert observed.observation_hash == _observation(work).observation_hash
    assert events == ["recovery", "claim", "backend", "recovery", "load"]
    assert wrapped.backend_called


def test_completed_artifacts_return_without_claim_backend_or_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    completed = _decision("completed", completed=True)
    delegate_call = Mock(side_effect=AssertionError("backend was replayed"))
    delegate = _RawDelegate(delegate_call)
    work, wrapped = _raw_wrapper(tmp_path, delegate)
    claim = Mock(side_effect=AssertionError("new claim was consumed"))
    monkeypatch.setattr(
        runner,
        "recover_existing_artifacts_without_model_v2",
        lambda **_kwargs: completed,
    )
    monkeypatch.setattr(
        runner,
        "authorize_clean_model_execution_once_v2",
        claim,
    )
    monkeypatch.setattr(
        runner,
        "load_completed_observation_without_model_v2",
        lambda **_kwargs: (completed, _observation(work).to_dict()),
    )

    observed = wrapped.run(
        work.request,
        skill_source_dir=None,
        trace_id="financial-semantic-v2:" + work.work_unit_hash[:20],
    )

    assert observed.observation_hash == _observation(work).observation_hash
    claim.assert_not_called()
    delegate_call.assert_not_called()
    assert not wrapped.backend_called


def test_backend_exception_never_retries_or_reauthorizes_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decisions = iter(
        (
            _decision("clean_never_started", may_claim=True),
            _decision(
                "recovery_required",
                error_type="offline_verifier_not_completed",
            ),
        )
    )
    delegate_call = Mock(side_effect=RuntimeError("crash after model"))
    delegate = _RawDelegate(delegate_call)
    work, wrapped = _raw_wrapper(tmp_path, delegate)
    claim = Mock()
    monkeypatch.setattr(
        runner,
        "recover_existing_artifacts_without_model_v2",
        lambda **_kwargs: next(decisions),
    )
    monkeypatch.setattr(
        runner,
        "authorize_clean_model_execution_once_v2",
        claim,
    )

    with pytest.raises(runner.PeriodOutRunnerError, match="replay remains forbidden"):
        wrapped.run(
            work.request,
            skill_source_dir=None,
            trace_id="financial-semantic-v2:" + work.work_unit_hash[:20],
        )
    with pytest.raises(runner.PeriodOutRunnerError, match="cannot enter backend twice"):
        wrapped.run(
            work.request,
            skill_source_dir=None,
            trace_id="financial-semantic-v2:" + work.work_unit_hash[:20],
        )
    claim.assert_called_once()
    delegate_call.assert_called_once()


def test_raw_and_candidate_delegates_cannot_be_cross_bound(tmp_path: Path) -> None:
    plan = _plan()
    raw = next(work for work in plan.work_units if work.arm == "raw")
    candidate = next(
        work for work in plan.work_units if work.arm == "candidate"
    )
    raw_delegate = _RawDelegate(lambda *_args: None)
    candidate_delegate = _CandidateDelegate(lambda *_args: None)

    with pytest.raises(runner.PeriodOutRunnerError, match="cross-bound"):
        runner.RecoveryBoundBackendV2(
            delegate=candidate_delegate,
            work=raw,
            state_root=tmp_path / "raw",
            trial_root=tmp_path / raw.request.trial_id,
            expected_process_scope=(
                CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER
            ),
        )
    with pytest.raises(runner.PeriodOutRunnerError, match="cross-bound"):
        runner.RecoveryBoundBackendV2(
            delegate=raw_delegate,
            work=candidate,
            state_root=tmp_path / "candidate",
            trial_root=tmp_path / candidate.request.trial_id,
            expected_process_scope=(
                CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER
            ),
            expected_plan_hash=_sha("plan"),
            expected_program_id=plan.treatment.recipe_id,
            expected_treatment_hash=plan.treatment.period_out_treatment_id,
            expected_external_source_receipt_hash=(
                plan.treatment.external_skill_source_receipt_hash
            ),
        )


def test_explicit_recovery_only_path_scans_all_sixteen_without_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()
    calls: list[str] = []
    clean = _decision("clean_never_started", may_claim=True)
    monkeypatch.setattr(
        runner,
        "recover_existing_artifacts_without_model_v2",
        lambda **kwargs: calls.append(kwargs["work_unit_hash"]) or clean,
    )
    precomputed = {
        work.target.item_id: {"plan_hash": _sha(work.target.item_id)}
        for work in plan.work_units
        if work.arm == "candidate"
    }

    report = runner.recover_measurement_artifacts_without_model_v2(
        destination=tmp_path / "output",
        worker_root=tmp_path / "workers",
        plan=plan,
        precomputed_receipts=precomputed,
        candidate=SimpleNamespace(
            recipe_id=plan.treatment.recipe_id,
            external_skill_source_receipt_hash=(
                plan.treatment.external_skill_source_receipt_hash
            ),
        ),
        expected_process_scope=(
            CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER
        ),
        execution_freeze_hash=_sha("freeze"),
    )

    assert len(calls) == len(set(calls)) == 16
    assert report["completed_work_unit_count"] == 0
    assert report["unresolved_work_unit_count"] == 16
    assert report["backend_construction_count"] == 0
    assert report["backend_calls_this_invocation"] == 0
    assert report["model_calls_this_invocation"] == 0
    assert report["offline_verifier_calls_this_invocation"] == 0
    assert report["online_judge_calls"] == 0
    assert report["missing_artifact_policy"] == (
        "report_and_stop_without_replay"
    )
    attempt = (
        tmp_path / "output" / "recovery_attempts" / f"{report['report_hash']}.json"
    )
    assert attempt.is_file()
