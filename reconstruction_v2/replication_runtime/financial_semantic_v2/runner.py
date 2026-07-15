from __future__ import annotations

"""Run the frozen eight-pair SEC-13F period-out measurement once."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    V320_PROTOCOL_RELATIVE_PATH,
    _configure_environment,
)
from assumption_agent.benchmarks.financial_semantic_integration_v1 import (
    SharedFinancialSemanticPlannerV1,
)
from assumption_agent.benchmarks.offline_verifier import (
    SkillLearnOfflineVerifierRuntimeCache,
)
from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnModelInferenceLimiter,
    SkillLearnPrebuiltImageCache,
    SkillLearnProviderCircuit,
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.events import JsonlEventSink
from assumption_agent.models import stable_hash
from assumption_agent.secure_env import load_dotenv, map_legacy_model_env

from .backends import (
    DurableFinancialSemanticSubprocessBackendV2,
    DurableRawSubprocessBackendV2,
    WORK_STAGE_ORDER_V2,
    backend_runtime_identity_v2,
    future_terminal_semantics_v2,
    initialize_work_state_v2,
)
from .durable_state import (
    atomic_write_hashed_json_v2,
    load_durable_stage_chain_v2,
    read_hashed_json_v2,
    transition_durable_stage_v2,
)
from .materialize import FAMILY
from .pack import (
    payload_hash,
    read_json,
    sha256_file,
    verify_measurement_view,
)
from .plan import (
    FixedPeriodOutTreatmentV2,
    MeasurementPlanV2,
    MeasurementTargetV2,
    MeasurementWorkUnitV2,
    build_measurement_plan_v2,
    execute_measurement_plan_v2,
)
from .prewarm import PREWARM_VERSION
from .recovery import (
    MODEL_EXECUTION_CLAIM_FILENAME,
    OBSERVATION_FILENAME,
    SEMANTIC_EVIDENCE_FILENAME,
    RecoveryDecisionV2,
    RecoveryEvidenceError,
    authorize_clean_model_execution_once_v2,
    load_completed_observation_without_model_v2,
    recover_existing_artifacts_without_model_v2,
)
from .treatment import FixedFinancialCandidateIdentityV1


RUNNER_VERSION = "financial_semantic_sec13f_period_out_runner_v1"
REPORT_FILENAME = "measurement.report.json"
FAILURE_FILENAME = "measurement.failure.json"
EVENTS_FILENAME = "measurement.events.jsonl"


class PeriodOutRunnerError(RuntimeError):
    """The final period-out execution boundary failed closed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_hashed_payload(
    payload: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    body = dict(payload)
    declared = body.pop(field, None)
    if not isinstance(declared, str) or declared != payload_hash(body):
        raise PeriodOutRunnerError(f"{label} self hash mismatch")
    return declared


class BoundPrecomputedPlannerV2:
    """One immutable item-local view over a frozen precomputed plan."""

    def __init__(
        self,
        *,
        shared: SharedFinancialSemanticPlannerV1,
        instruction_sha256: str,
        plan: Mapping[str, Any],
        extraction_receipt: Mapping[str, Any],
    ) -> None:
        if not _is_sha256(instruction_sha256):
            raise PeriodOutRunnerError("precomputed instruction hash is invalid")
        self.asset = shared.asset
        self._instruction_sha256 = instruction_sha256
        self._plan = copy.deepcopy(dict(plan))
        self._receipt = copy.deepcopy(dict(extraction_receipt))
        if (
            self._plan.get("instruction_sha256") != instruction_sha256
            or self._receipt.get("plan_hash") != self._plan.get("plan_hash")
            or not _is_sha256(self._plan.get("plan_hash"))
            or not _is_sha256(self._receipt.get("receipt_hash"))
        ):
            raise PeriodOutRunnerError(
                "precomputed semantic plan receipt is inconsistent"
            )
        self._planner_hash = stable_hash(
            {
                "policy": "item_local_precomputed_financial_plan_v2",
                "shared_planner_hash": shared.planner_hash,
                "instruction_sha256": instruction_sha256,
                "plan_hash": self._plan["plan_hash"],
                "extraction_receipt_hash": self._receipt["receipt_hash"],
            }
        )

    @property
    def planner_hash(self) -> str:
        return self._planner_hash

    def build(self, instruction: str) -> tuple[dict[str, Any], dict[str, Any]]:
        observed = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        if observed != self._instruction_sha256:
            raise PeriodOutRunnerError(
                "runtime instruction differs from the precomputed plan"
            )
        return copy.deepcopy(self._plan), copy.deepcopy(self._receipt)


def _write_or_verify_hashed_json_v2(
    path: Path,
    body: Mapping[str, Any],
    *,
    hash_field: str,
) -> dict[str, Any]:
    """Create one immutable receipt or verify an identical prior write."""

    if path.exists() or path.is_symlink():
        value = read_hashed_json_v2(path, hash_field=hash_field)
        observed = dict(value)
        observed.pop(hash_field, None)
        if observed != dict(body):
            raise PeriodOutRunnerError(
                f"existing {path.name} identity drifted"
            )
        return value
    return atomic_write_hashed_json_v2(
        path,
        body,
        hash_field=hash_field,
    )


def _ensure_work_state_v2(
    *,
    state_root: Path,
    work: MeasurementWorkUnitV2,
    planned_payload: Mapping[str, Any],
    semantic_plan_payload: Mapping[str, Any],
) -> None:
    """Finish only the two deterministic pre-model stages on resume."""

    chain = load_durable_stage_chain_v2(
        state_root,
        stage_order=WORK_STAGE_ORDER_V2,
        work_unit_hash=work.work_unit_hash,
        request_hash=work.request.request_hash,
    )
    if not chain:
        initialize_work_state_v2(
            state_root=state_root,
            work_unit_hash=work.work_unit_hash,
            request_hash=work.request.request_hash,
            planned_payload=planned_payload,
            semantic_plan_payload=semantic_plan_payload,
        )
        return
    if dict(chain[0].payload) != dict(planned_payload):
        raise PeriodOutRunnerError("planned work identity drifted on resume")
    if len(chain) == 1:
        transition_durable_stage_v2(
            state_root,
            stage_order=WORK_STAGE_ORDER_V2,
            work_unit_hash=work.work_unit_hash,
            request_hash=work.request.request_hash,
            stage="semantic_plan_ready",
            predecessor_stage_hash=chain[0].stage_hash,
            payload=semantic_plan_payload,
        )
        return
    if dict(chain[1].payload) != dict(semantic_plan_payload):
        raise PeriodOutRunnerError(
            "semantic plan identity drifted on resume"
        )


def _trial_root_for_work_v2(
    worker_root: Path,
    work: MeasurementWorkUnitV2,
) -> Path:
    skill_config = (
        "no_skill"
        if work.arm == "raw"
        else "assumption-agent-v2-challenger"
    )
    return (
        worker_root
        / work.work_unit_hash
        / "trials"
        / skill_config
        / work.request.family
        / work.request.item_id
        / work.request.trial_id
    )


_RECOVERABLE_OBSERVATION_FIELDS = frozenset(
    {
        "request",
        "success",
        "score",
        "metrics",
        "total_tokens",
        "steps",
        "duration_seconds",
        "provider_fingerprint",
        "fairness_fingerprint",
        "error_type",
        "upstream_result_hash",
        "raw_trial_artifacts_persisted",
        "prebuilt_image_key",
        "prebuilt_image_id",
        "prebuilt_cache_reused",
        "agent_runtime_key",
        "agent_runtime_version",
        "offline_verifier_profile_id",
        "offline_verifier_runtime_key",
        "step_budget_policy",
        "step_budget_unit",
        "step_budget_limit",
        "step_budget_truncated",
        "step_budget_token_usage_complete",
        "step_budget_receipt_hash",
        "installed_skill_source_receipt_hash",
        "runtime_profile_prompt_delivery_policy",
        "runtime_profile_prompt_injection_receipt_hash",
        "runtime_profile_effective_prompt_sha256",
        "secret_value_persisted",
    }
)


def _hydrate_recovered_observation_v2(
    request: SkillLearnTrialRequest,
    payload: Mapping[str, Any],
) -> SkillLearnTrialObservation:
    """Rehydrate only the exact content-free observation schema."""

    body = dict(payload)
    if (
        set(body) != _RECOVERABLE_OBSERVATION_FIELDS
        or body.get("request") != request.to_dict()
        or body.get("secret_value_persisted") is not False
        or not isinstance(body.get("success"), bool)
        or isinstance(body.get("score"), bool)
        or not isinstance(body.get("score"), (int, float))
        or not math.isfinite(float(body["score"]))
        or not isinstance(body.get("metrics"), Mapping)
        or not all(
            isinstance(key, str)
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for key, value in body["metrics"].items()
        )
        or any(
            isinstance(body.get(field), bool)
            or not isinstance(body.get(field), int)
            or int(body[field]) < 0
            for field in ("total_tokens", "steps", "step_budget_limit")
        )
        or isinstance(body.get("duration_seconds"), bool)
        or not isinstance(body.get("duration_seconds"), (int, float))
        or not math.isfinite(float(body["duration_seconds"]))
        or float(body["duration_seconds"]) < 0.0
        or any(
            not isinstance(body.get(field), bool)
            for field in (
                "raw_trial_artifacts_persisted",
                "prebuilt_cache_reused",
                "step_budget_truncated",
                "step_budget_token_usage_complete",
            )
        )
        or (
            body.get("error_type") is not None
            and not isinstance(body.get("error_type"), str)
        )
        or any(
            not isinstance(body.get(field), str)
            for field in (
                "provider_fingerprint",
                "fairness_fingerprint",
                "upstream_result_hash",
                "prebuilt_image_key",
                "prebuilt_image_id",
                "agent_runtime_key",
                "agent_runtime_version",
                "offline_verifier_profile_id",
                "offline_verifier_runtime_key",
                "step_budget_policy",
                "step_budget_unit",
                "step_budget_receipt_hash",
                "installed_skill_source_receipt_hash",
                "runtime_profile_prompt_delivery_policy",
                "runtime_profile_prompt_injection_receipt_hash",
                "runtime_profile_effective_prompt_sha256",
            )
        )
    ):
        raise PeriodOutRunnerError(
            "recovered observation schema or identity drifted"
        )
    observation = SkillLearnTrialObservation(
        request=request,
        success=body["success"],
        score=body["score"],
        metrics=dict(body["metrics"]),
        total_tokens=body["total_tokens"],
        steps=body["steps"],
        duration_seconds=body["duration_seconds"],
        provider_fingerprint=body["provider_fingerprint"],
        fairness_fingerprint=body["fairness_fingerprint"],
        error_type=body["error_type"],
        upstream_result_hash=body["upstream_result_hash"],
        raw_trial_artifacts_persisted=(
            body["raw_trial_artifacts_persisted"]
        ),
        prebuilt_image_key=body["prebuilt_image_key"],
        prebuilt_image_id=body["prebuilt_image_id"],
        prebuilt_cache_reused=body["prebuilt_cache_reused"],
        agent_runtime_key=body["agent_runtime_key"],
        agent_runtime_version=body["agent_runtime_version"],
        offline_verifier_profile_id=body["offline_verifier_profile_id"],
        offline_verifier_runtime_key=body["offline_verifier_runtime_key"],
        step_budget_policy=body["step_budget_policy"],
        step_budget_unit=body["step_budget_unit"],
        step_budget_limit=body["step_budget_limit"],
        step_budget_truncated=body["step_budget_truncated"],
        step_budget_token_usage_complete=(
            body["step_budget_token_usage_complete"]
        ),
        step_budget_receipt_hash=body["step_budget_receipt_hash"],
        installed_skill_source_receipt_hash=(
            body["installed_skill_source_receipt_hash"]
        ),
        runtime_profile_prompt_delivery_policy=(
            body["runtime_profile_prompt_delivery_policy"]
        ),
        runtime_profile_prompt_injection_receipt_hash=(
            body["runtime_profile_prompt_injection_receipt_hash"]
        ),
        runtime_profile_effective_prompt_sha256=(
            body["runtime_profile_effective_prompt_sha256"]
        ),
    )
    if stable_hash(observation.to_dict()) != stable_hash(body):
        raise PeriodOutRunnerError(
            "recovered observation cannot be losslessly rehydrated"
        )
    return observation


class RecoveryBoundBackendV2:
    """Execute a physical backend at most once behind a durable claim."""

    def __init__(
        self,
        *,
        delegate: Any,
        work: MeasurementWorkUnitV2,
        state_root: Path,
        trial_root: Path,
        expected_process_scope: str,
        expected_plan_hash: str | None = None,
        expected_program_id: str | None = None,
        expected_treatment_hash: str | None = None,
        expected_external_source_receipt_hash: str | None = None,
    ) -> None:
        if work.arm == "raw":
            if not isinstance(delegate, DurableRawSubprocessBackendV2) or any(
                value is not None
                for value in (
                    expected_plan_hash,
                    expected_program_id,
                    expected_treatment_hash,
                    expected_external_source_receipt_hash,
                )
            ):
                raise PeriodOutRunnerError(
                    "RAW work was cross-bound to candidate runtime state"
                )
        elif work.arm == "candidate":
            if (
                not isinstance(
                    delegate,
                    DurableFinancialSemanticSubprocessBackendV2,
                )
                or not all(
                    _is_sha256(value)
                    for value in (
                        expected_plan_hash,
                        expected_program_id,
                        expected_treatment_hash,
                        expected_external_source_receipt_hash,
                    )
                )
            ):
                raise PeriodOutRunnerError(
                    "candidate work was cross-bound to RAW runtime state"
                )
        else:
            raise PeriodOutRunnerError("unknown physical work arm")
        self.delegate = delegate
        self.work = work
        self.state_root = state_root.resolve()
        self.trial_root = trial_root.resolve()
        self.expected_process_scope = str(expected_process_scope)
        self.expected_plan_hash = expected_plan_hash
        self.expected_program_id = expected_program_id
        self.expected_treatment_hash = expected_treatment_hash
        self.expected_external_source_receipt_hash = (
            expected_external_source_receipt_hash
        )
        self._run_lock = threading.Lock()
        self._entered = False
        self._backend_called = False
        self._last_decision: RecoveryDecisionV2 | None = None

    @property
    def backend_called(self) -> bool:
        return self._backend_called

    @property
    def last_decision(self) -> RecoveryDecisionV2 | None:
        return self._last_decision

    def _recovery_kwargs(self) -> dict[str, Any]:
        request = self.work.request
        return {
            "state_root": self.state_root,
            "trial_root": self.trial_root,
            "work_unit_hash": self.work.work_unit_hash,
            "request_hash": request.request_hash,
            "trial_id": request.trial_id,
            "arm": self.work.arm,
            "expected_action_limit": request.max_steps,
            "expected_process_scope": self.expected_process_scope,
            "expected_plan_hash": self.expected_plan_hash,
            "expected_program_id": self.expected_program_id,
            "expected_treatment_hash": self.expected_treatment_hash,
            "expected_external_source_receipt_hash": (
                self.expected_external_source_receipt_hash
            ),
        }

    def inspect_recovery(self) -> RecoveryDecisionV2:
        decision = recover_existing_artifacts_without_model_v2(
            **self._recovery_kwargs()
        )
        self._last_decision = decision
        if decision.status not in {
            "clean_never_started",
            "completed",
            "reconciled_completed",
        }:
            raise PeriodOutRunnerError(
                "work recovery blocked model execution: "
                f"{decision.status}:{decision.error_type}"
            )
        return decision

    def _load_completed(self) -> SkillLearnTrialObservation:
        try:
            decision, payload = (
                load_completed_observation_without_model_v2(
                    **self._recovery_kwargs()
                )
            )
        except RecoveryEvidenceError as exc:
            raise PeriodOutRunnerError(
                "completed observation recovery failed closed"
            ) from exc
        self._last_decision = decision
        return _hydrate_recovered_observation_v2(
            self.work.request,
            payload,
        )

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        with self._run_lock:
            if self._entered:
                raise PeriodOutRunnerError(
                    "physical work unit cannot enter backend twice"
                )
            self._entered = True
        expected_trace_id = (
            "financial-semantic-v2:" + self.work.work_unit_hash[:20]
        )
        if (
            request.request_hash != self.work.request.request_hash
            or request.trial_id != self.work.request.trial_id
            or trace_id != expected_trace_id
            or skill_source_dir != self.work.skill_source_dir
            or (
                self.work.arm == "raw"
                and (
                    request.variant is not TrialVariant.POLICY_OFF
                    or skill_source_dir is not None
                )
            )
            or (
                self.work.arm == "candidate"
                and (
                    request.variant is not TrialVariant.POLICY_ON
                    or skill_source_dir is None
                )
            )
        ):
            raise PeriodOutRunnerError(
                "physical work request crossed arm or trial identity"
            )

        before = self.inspect_recovery()
        if before.completed:
            return self._load_completed()
        if not before.may_claim_clean_model_execution:
            raise PeriodOutRunnerError(
                "non-clean work cannot claim model execution"
            )
        authorize_clean_model_execution_once_v2(
            state_root=self.state_root,
            trial_root=self.trial_root,
            work_unit_hash=self.work.work_unit_hash,
            request_hash=request.request_hash,
            trial_id=request.trial_id,
            arm=self.work.arm,
            expected_plan_hash=self.expected_plan_hash,
        )
        self._backend_called = True
        try:
            observation = self.delegate.run(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
        except Exception as exc:
            after_error = recover_existing_artifacts_without_model_v2(
                **self._recovery_kwargs()
            )
            self._last_decision = after_error
            if after_error.completed:
                return self._load_completed()
            raise PeriodOutRunnerError(
                "claimed backend failed; model replay remains forbidden: "
                f"{after_error.status}:{after_error.error_type}"
            ) from exc
        after = recover_existing_artifacts_without_model_v2(
            **self._recovery_kwargs()
        )
        self._last_decision = after
        if not after.completed:
            raise PeriodOutRunnerError(
                "backend returned without complete durable evidence: "
                f"{after.status}:{after.error_type}"
            )
        recovered = self._load_completed()
        if (
            not isinstance(observation, SkillLearnTrialObservation)
            or observation.observation_hash != recovered.observation_hash
        ):
            raise PeriodOutRunnerError(
                "backend observation differs from durable recovery receipt"
            )
        return observation


def build_plan_from_freeze_v1(
    *,
    measurement_view: Mapping[str, Any],
    execution_freeze: Mapping[str, Any],
    candidate: FixedFinancialCandidateIdentityV1,
    protocol: PaperProtocol,
) -> MeasurementPlanV2:
    view = verify_measurement_view(measurement_view)
    treatment_payload = execution_freeze.get("treatment")
    if not isinstance(treatment_payload, Mapping):
        raise PeriodOutRunnerError("execution freeze treatment is missing")
    if (
        treatment_payload.get("recipe_id") != candidate.recipe_id
        or treatment_payload.get("program_set_hash")
        != candidate.program_set_hash
        or treatment_payload.get("external_skill_source_receipt_hash")
        != candidate.external_skill_source_receipt_hash
        or not _is_sha256(
            treatment_payload.get("period_out_treatment_id")
        )
    ):
        raise PeriodOutRunnerError("execution freeze candidate identity drifted")
    targets = tuple(
        MeasurementTargetV2(
            item_id=str(item["item_id"]),
            fold_id=f"measurement-fold-{int(item['fold'])}",
        )
        for item in view["measurement_items"]
    )
    plan = build_measurement_plan_v2(
        targets=targets,
        manifest_hash=str(view["measurement_view_hash"]),
        evaluator_epoch=str(treatment_payload["evaluator_epoch"]),
        treatment=FixedPeriodOutTreatmentV2(
            recipe_id=candidate.recipe_id,
            program_set_hash=candidate.program_set_hash,
            period_out_treatment_id=str(
                treatment_payload["period_out_treatment_id"]
            ),
            external_skill_source_receipt_hash=(
                candidate.external_skill_source_receipt_hash
            ),
            candidate_skill_source=candidate.candidate_skill_source,
        ),
        agent_id=str(protocol.payload["agent_id"]),
        model=str(protocol.payload["model"]),
        max_steps=int(protocol.payload["max_steps"]),
        codex_agent_execution_policy_hash=(
            protocol.codex_agent_execution_policy.policy_hash
        ),
    )
    frozen_plan = execution_freeze.get("plan")
    if (
        not isinstance(frozen_plan, Mapping)
        or frozen_plan.get("plan_hash") != plan.plan_hash
        or frozen_plan.get("safe_payload") != plan.safe_payload()
    ):
        raise PeriodOutRunnerError("execution plan differs from its freeze")
    return plan


def _prewarm_by_item(
    *,
    prewarm: Mapping[str, Any],
    measurement_view_hash: str,
) -> dict[str, Mapping[str, Any]]:
    if (
        prewarm.get("prewarm_version") != PREWARM_VERSION
        or prewarm.get("measurement_view_hash") != measurement_view_hash
        or prewarm.get("formal_execution_cache_only") is not True
        or prewarm.get("formal_verifier_network") != "none"
        or prewarm.get("model_calls") != 0
        or prewarm.get("online_judge_calls") != 0
        or prewarm.get("sealed_content_accessed") is not False
    ):
        raise PeriodOutRunnerError("prewarm policy drifted")
    _verify_hashed_payload(prewarm, field="prewarm_hash", label="prewarm")
    rows = prewarm.get("formal_cache_rows")
    if not isinstance(rows, list) or len(rows) != 8:
        raise PeriodOutRunnerError("prewarm item rows drifted")
    result = {str(row.get("item_id")): row for row in rows if isinstance(row, Mapping)}
    if len(result) != 8:
        raise PeriodOutRunnerError("prewarm item identities are not unique")
    return result


def _verify_formal_local_cache(
    *,
    benchmark_root: Path,
    item_ids: Sequence[str],
    prewarm_rows: Mapping[str, Mapping[str, Any]],
    event_sink: JsonlEventSink,
) -> tuple[SkillLearnPrebuiltImageCache, SkillLearnOfflineVerifierRuntimeCache]:
    cache = SkillLearnPrebuiltImageCache(
        benchmark_root,
        cache_only=True,
        event_sink=event_sink,
    )
    offline = SkillLearnOfflineVerifierRuntimeCache(event_sink=event_sink)
    # Load the upstream runner without starting a model or container trial.
    from assumption_agent.benchmarks.skilllearn_lifecycle import (
        SkillLearnSubprocessBackend,
    )

    loader = SkillLearnSubprocessBackend(
        benchmark_root,
        agent_id="codex",
        provider_mode="openai_compatible",
        record_upstream=False,
        prebuilt_cache=cache,
        event_sink=event_sink,
    )
    runner = loader._load_runner()
    profile = None
    from assumption_agent.benchmarks.offline_verifier import (
        offline_verifier_profile_for_family,
    )

    profile = offline_verifier_profile_for_family(FAMILY)
    if profile is None:
        raise PeriodOutRunnerError("offline verifier profile is unavailable")
    for item_id in item_ids:
        expected = prewarm_rows.get(item_id)
        if not isinstance(expected, Mapping):
            raise PeriodOutRunnerError("prewarm row is missing")
        image = cache.ensure(
            family=FAMILY,
            item_id=item_id,
            agent_id="codex",
            runner=runner,
            trace_id=f"period-out-formal-preflight:{payload_hash({'item_id': item_id})[:20]}",
        )
        runtime = offline.ensure(
            profile=profile,
            base_image_tag=image.tag,
            base_image_id=image.image_id,
            delegate=runner.subprocess,
            trace_id=f"period-out-formal-verifier:{payload_hash({'item_id': item_id})[:20]}",
        )
        if (
            image.cache_key != expected.get("cache_key")
            or image.environment_hash != expected.get("environment_hash")
            or image.source_environment_hash
            != expected.get("source_environment_hash")
            or image.image_id != expected.get("image_id")
            or runtime.runtime_key
            != expected.get("offline_verifier_runtime_key")
            or runtime.profile.profile_hash
            != expected.get("offline_verifier_profile_hash")
            or not image.reused
        ):
            raise PeriodOutRunnerError("formal local cache differs from prewarm")
    return cache, offline


def _artifact_closure(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for current, directories, files in os.walk(
        resolved,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        kept: list[str] = []
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise PeriodOutRunnerError("artifact closure contains a symlink")
            relative = path.relative_to(resolved)
            if len(relative.parts) >= 2 and relative.parts[-2:] == (
                "agent",
                "tmp",
            ):
                continue
            kept.append(name)
        directories[:] = kept
        for name in sorted(files):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise PeriodOutRunnerError(
                    "artifact closure contains a special file"
                )
            relative = path.relative_to(resolved).as_posix()
            rows.append(
                {
                    "locator_hash": stable_hash(
                        {"relative_path": relative}
                    ),
                    "file_sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    rows.sort(key=lambda row: row["locator_hash"])
    return {
        "closure_policy": "content_hashes_without_raw_locators_v1",
        "file_count": len(rows),
        "files": rows,
        "file_set_hash": stable_hash(rows),
        "raw_locator_text_persisted": False,
    }


def _descriptive_results(execution: Any) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    fold_rows: dict[str, list[dict[str, Any]]] = {}
    for pair in execution.pair_results:
        raw = pair.raw_observation
        candidate = pair.candidate_observation
        if not isinstance(raw, SkillLearnTrialObservation) or not isinstance(
            candidate, SkillLearnTrialObservation
        ):
            raise PeriodOutRunnerError("backend returned an unknown observation")
        delta = int(candidate.success) - int(raw.success)
        relation = "gain" if delta > 0 else "harm" if delta < 0 else "tie"
        row = {
            "item_id_hash": stable_hash({"item_id": pair.target.item_id}),
            "fold_id": pair.target.fold_id,
            "pair_id": pair.pair_id,
            "raw_observation_hash": raw.observation_hash,
            "candidate_observation_hash": candidate.observation_hash,
            "raw_valid": raw.valid,
            "candidate_valid": candidate.valid,
            "raw_success": raw.success,
            "candidate_success": candidate.success,
            "delta": delta,
            "relation": relation,
        }
        pair_rows.append(row)
        fold_rows.setdefault(pair.target.fold_id, []).append(row)
    raw_successes = sum(row["raw_success"] for row in pair_rows)
    candidate_successes = sum(row["candidate_success"] for row in pair_rows)
    folds = []
    for fold_id, rows in sorted(fold_rows.items()):
        folds.append(
            {
                "fold_id": fold_id,
                "item_count": len(rows),
                "raw_successes": sum(row["raw_success"] for row in rows),
                "candidate_successes": sum(
                    row["candidate_success"] for row in rows
                ),
                "delta": sum(row["delta"] for row in rows),
                "gain_count": sum(row["relation"] == "gain" for row in rows),
                "harm_count": sum(row["relation"] == "harm" for row in rows),
                "tie_count": sum(row["relation"] == "tie" for row in rows),
            }
        )
    return {
        "pairs": pair_rows,
        "pair_set_hash": stable_hash(pair_rows),
        "folds": folds,
        "fold_set_hash": stable_hash(folds),
        "raw_successes": raw_successes,
        "candidate_successes": candidate_successes,
        "net_delta": candidate_successes - raw_successes,
        "gain_count": sum(row["relation"] == "gain" for row in pair_rows),
        "harm_count": sum(row["relation"] == "harm" for row in pair_rows),
        "tie_count": sum(row["relation"] == "tie" for row in pair_rows),
        "invalid_pair_count": sum(
            not row["raw_valid"] or not row["candidate_valid"]
            for row in pair_rows
        ),
    }


def recover_measurement_artifacts_without_model_v2(
    *,
    destination: Path,
    worker_root: Path,
    plan: MeasurementPlanV2,
    precomputed_receipts: Mapping[str, Mapping[str, Any]],
    candidate: FixedFinancialCandidateIdentityV1,
    expected_process_scope: str,
    execution_freeze_hash: str,
) -> dict[str, Any]:
    """Reconcile all sixteen units from existing artifacts and stop.

    This is deliberately separate from the measurement executor.  It neither
    constructs a benchmark backend nor reaches the claim/model boundary.  The
    only writes it can make are missing durable stage receipts whose contents
    are already proven by frozen host artifacts, plus one content-addressed
    recovery-attempt report.
    """

    plan.verify()
    rows: list[dict[str, Any]] = []
    for work in plan.work_units:
        expected_plan_hash: str | None = None
        expected_program_id: str | None = None
        expected_treatment_hash: str | None = None
        expected_source_hash: str | None = None
        if work.arm == "candidate":
            precomputed = precomputed_receipts.get(work.target.item_id)
            if not isinstance(precomputed, Mapping) or not _is_sha256(
                precomputed.get("plan_hash")
            ):
                raise PeriodOutRunnerError(
                    "candidate artifact recovery lacks its frozen plan"
                )
            expected_plan_hash = str(precomputed["plan_hash"])
            expected_program_id = candidate.recipe_id
            expected_treatment_hash = (
                plan.treatment.period_out_treatment_id
            )
            expected_source_hash = (
                candidate.external_skill_source_receipt_hash
            )
        recovery_kwargs = {
            "state_root": worker_root / work.work_unit_hash / "durable",
            "trial_root": _trial_root_for_work_v2(worker_root, work),
            "work_unit_hash": work.work_unit_hash,
            "request_hash": work.request.request_hash,
            "trial_id": work.request.trial_id,
            "arm": work.arm,
            "expected_action_limit": work.request.max_steps,
            "expected_process_scope": str(expected_process_scope),
            "expected_plan_hash": expected_plan_hash,
            "expected_program_id": expected_program_id,
            "expected_treatment_hash": expected_treatment_hash,
            "expected_external_source_receipt_hash": expected_source_hash,
        }
        decision = recover_existing_artifacts_without_model_v2(
            **recovery_kwargs
        )
        observation_hash: str | None = None
        if decision.completed:
            loaded, payload = load_completed_observation_without_model_v2(
                **recovery_kwargs
            )
            if not loaded.completed:
                raise PeriodOutRunnerError(
                    "artifact recovery lost a completed decision"
                )
            observation_hash = _hydrate_recovered_observation_v2(
                work.request,
                payload,
            ).observation_hash
        rows.append(
            {
                "work_unit_hash": work.work_unit_hash,
                "request_hash": work.request.request_hash,
                "arm": work.arm,
                "decision": decision.to_dict(),
                "observation_hash": observation_hash,
            }
        )

    completed_count = sum(
        bool(row["decision"]["completed"]) for row in rows
    )
    invalid_count = sum(
        row["decision"]["status"] == "invalid" for row in rows
    )
    transition_count = sum(
        len(row["decision"]["transitions_applied"]) for row in rows
    )
    body = {
        "runner_version": RUNNER_VERSION,
        "report_type": "artifact_recovery_only",
        "execution_freeze_hash": execution_freeze_hash,
        "plan_hash": plan.plan_hash,
        "physical_work_unit_count": len(plan.work_units),
        "completed_work_unit_count": completed_count,
        "unresolved_work_unit_count": len(rows) - completed_count,
        "invalid_work_unit_count": invalid_count,
        "recovery_completed": completed_count == len(rows),
        "evidence_valid": completed_count == len(rows) and invalid_count == 0,
        "decisions": rows,
        "decision_set_hash": stable_hash(rows),
        "stage_receipt_transitions_this_invocation": transition_count,
        "recovery_mode": "existing_frozen_artifacts_only",
        "upstream_safe_post_agent_resume_api_available": False,
        "missing_artifact_policy": "report_and_stop_without_replay",
        "backend_construction_count": 0,
        "backend_calls_this_invocation": 0,
        "model_calls_this_invocation": 0,
        "operator_execution_calls_this_invocation": 0,
        "offline_verifier_calls_this_invocation": 0,
        "observation_materialization_calls_this_invocation": 0,
        "model_replay_authorized": False,
        "model_replay_count": 0,
        "offline_artifacts_only": True,
        "network_calls": 0,
        "online_judge_calls": 0,
        "sealed_content_accessed": False,
        "secret_value_persisted": False,
    }
    attempt_hash = stable_hash(body)
    attempts = destination / "recovery_attempts"
    if attempts.is_symlink() or (
        attempts.exists() and not attempts.is_dir()
    ):
        raise PeriodOutRunnerError(
            "artifact-recovery attempt root is not a regular directory"
        )
    return _write_or_verify_hashed_json_v2(
        attempts / f"{attempt_hash}.json",
        body,
        hash_field="report_hash",
    )


def run_measurement_v1(
    *,
    project_root: str | Path,
    benchmark_root: str | Path,
    measurement_view_path: str | Path,
    prewarm_path: str | Path,
    execution_freeze: Mapping[str, Any],
    candidate: FixedFinancialCandidateIdentityV1,
    env_file: str | Path,
    minilm_snapshot_root: str | Path,
    qa_snapshot_root: str | Path,
    output_root: str | Path,
    recover_only: bool = False,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    benchmark = Path(benchmark_root).expanduser().resolve(strict=True)
    view_path = Path(measurement_view_path).expanduser().resolve(strict=True)
    prewarm_file = Path(prewarm_path).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve()
    if destination.is_symlink():
        raise FileExistsError(destination)
    destination_preexisting = destination.exists()
    if destination_preexisting:
        if not destination.is_dir():
            raise FileExistsError(destination)
        for name in (
            "execution.plan.json",
            "batch.started.json",
            REPORT_FILENAME,
            FAILURE_FILENAME,
        ):
            if (destination / name).is_symlink():
                raise PeriodOutRunnerError(
                    "output root contains a symlinked control receipt"
                )
        if (destination / REPORT_FILENAME).exists():
            raise FileExistsError(destination / REPORT_FILENAME)
        existing_names = {path.name for path in destination.iterdir()}
        if existing_names and "execution.plan.json" not in existing_names:
            raise PeriodOutRunnerError(
                "nonempty output root lacks an execution-plan resume marker"
            )
    else:
        destination.mkdir(parents=True)
    prior_failure_receipt_present = (
        destination / FAILURE_FILENAME
    ).is_file()
    execution_started = False
    try:
        load_dotenv(env_file)
        map_legacy_model_env()
        protocol = PaperProtocol.read(
            project / V320_PROTOCOL_RELATIVE_PATH
        )
        _configure_environment(protocol)
        view = verify_measurement_view(read_json(view_path))
        prewarm = read_json(prewarm_file)
        prewarm_rows = _prewarm_by_item(
            prewarm=prewarm,
            measurement_view_hash=str(view["measurement_view_hash"]),
        )
        if execution_freeze.get("manifest_version") != (
            "financial_semantic_sec13f_period_out_execution_freeze_v1"
        ):
            raise PeriodOutRunnerError("execution freeze version drifted")
        plan = build_plan_from_freeze_v1(
            measurement_view=view,
            execution_freeze=execution_freeze,
            candidate=candidate,
            protocol=protocol,
        )
        event_sink = JsonlEventSink(destination / EVENTS_FILENAME)
        planner = SharedFinancialSemanticPlannerV1(
            asset_path=candidate.operator_asset_path,
            minilm_runtime_asset_path=candidate.minilm_runtime_asset_path,
            minilm_snapshot_root=Path(minilm_snapshot_root).resolve(strict=True),
            qa_runtime_asset_path=candidate.qa_runtime_asset_path,
            qa_snapshot_root=Path(qa_snapshot_root).resolve(strict=True),
        )
        instruction_by_item = {
            str(item["item_id"]): str(item["instruction"])
            for item in view["measurement_items"]
        }
        instruction_hash_by_item = {
            str(item["item_id"]): str(item["instruction_sha256"])
            for item in view["measurement_items"]
        }
        precomputed: dict[str, BoundPrecomputedPlannerV2] = {}
        precomputed_receipts: dict[str, Mapping[str, Any]] = {}
        for item_id in sorted(instruction_by_item):
            semantic_plan, extraction = planner.build(
                instruction_by_item[item_id]
            )
            bound = BoundPrecomputedPlannerV2(
                shared=planner,
                instruction_sha256=instruction_hash_by_item[item_id],
                plan=semantic_plan,
                extraction_receipt=extraction,
            )
            precomputed[item_id] = bound
            precomputed_receipts[item_id] = {
                "applicable": True,
                "instruction_sha256": instruction_hash_by_item[item_id],
                "plan_hash": semantic_plan["plan_hash"],
                "extraction_receipt_hash": extraction["receipt_hash"],
                "planner_hash": bound.planner_hash,
                "model_calls": 0,
                "online_calls": 0,
            }

        execution_plan_body = {
            "runner_version": RUNNER_VERSION,
            "execution_freeze_hash": execution_freeze["manifest_hash"],
            "plan_hash": plan.plan_hash,
            "safe_payload": plan.safe_payload(),
            "precomputed_plan_set_hash": stable_hash(
                {
                    key: dict(value)
                    for key, value in sorted(precomputed_receipts.items())
                }
            ),
            "physical_work_unit_count": 16,
            "model_calls_before_batch": 0,
            "sealed_content_accessed": False,
        }
        _write_or_verify_hashed_json_v2(
            destination / "execution.plan.json",
            execution_plan_body,
            hash_field="plan_receipt_hash",
        )

        worker_root = destination / "worker_state"
        for work in plan.work_units:
            state = worker_root / work.work_unit_hash / "durable"
            semantic_payload = (
                precomputed_receipts[work.target.item_id]
                if work.arm == "candidate"
                else {
                    "applicable": False,
                    "arm": "raw",
                    "model_calls": 0,
                    "online_calls": 0,
                }
            )
            planned_payload = {
                **work.safe_payload(),
                "trial_id": work.request.trial_id,
                "execution_freeze_hash": execution_freeze["manifest_hash"],
                "model_calls": 0,
                "retry_count": 0,
            }
            _ensure_work_state_v2(
                state_root=state,
                work=work,
                planned_payload=planned_payload,
                semantic_plan_payload=semantic_payload,
            )

        if recover_only:
            return recover_measurement_artifacts_without_model_v2(
                destination=destination,
                worker_root=worker_root,
                plan=plan,
                precomputed_receipts=precomputed_receipts,
                candidate=candidate,
                expected_process_scope=str(
                    protocol.codex_agent_execution_policy
                    .action_budget_process_scope
                ),
                execution_freeze_hash=str(
                    execution_freeze["manifest_hash"]
                ),
            )

        cache, offline_cache = _verify_formal_local_cache(
            benchmark_root=benchmark,
            item_ids=[
                work.target.item_id
                for work in plan.work_units
                if work.arm == "raw"
            ],
            prewarm_rows=prewarm_rows,
            event_sink=event_sink,
        )
        if cache.cache_only is not True or not isinstance(
            offline_cache,
            SkillLearnOfflineVerifierRuntimeCache,
        ):
            raise PeriodOutRunnerError(
                "formal execution requires cache-only offline evaluation"
            )

        batch_start_body = {
            "runner_version": RUNNER_VERSION,
            "execution_freeze_hash": execution_freeze["manifest_hash"],
            "plan_hash": plan.plan_hash,
            "physical_work_unit_count": 16,
            "all_futures_required": True,
            "retry_authorized": False,
            "model_replay_authorized": False,
            "cache_only": True,
            "offline_judge_only": True,
        }
        _write_or_verify_hashed_json_v2(
            destination / "batch.started.json",
            batch_start_body,
            hash_field="batch_start_hash",
        )
        execution_started = True
        limiter = SkillLearnModelInferenceLimiter(16)
        backends: dict[str, Any] = {}

        def backend_factory(work: MeasurementWorkUnitV2) -> Any:
            state_root = worker_root / work.work_unit_hash / "durable"
            common = {
                "agent_id": work.request.agent_id,
                "model": work.request.model,
                "max_steps": work.request.max_steps,
                "provider_mode": "openai_compatible",
                "trials_dir": worker_root
                / work.work_unit_hash
                / "trials",
                "record_upstream": True,
                "prebuilt_cache": cache,
                "offline_verifier_cache": offline_cache,
                "provider_circuit": SkillLearnProviderCircuit(),
                "model_inference_limiter": limiter,
                "codex_agent_execution_policy": (
                    protocol.codex_agent_execution_policy
                ),
                "event_sink": event_sink,
                "durable_state_root": state_root,
                "durable_work_unit_hash": work.work_unit_hash,
                "durable_request_hash": work.request.request_hash,
            }
            if work.arm == "candidate":
                bound = precomputed[work.target.item_id]
                delegate = DurableFinancialSemanticSubprocessBackendV2(
                    benchmark,
                    planner=bound,
                    expected_program_id=candidate.recipe_id,
                    expected_program_set_hash=candidate.program_set_hash,
                    expected_treatment_hash=plan.treatment.period_out_treatment_id,
                    expected_external_skill_source_receipt_hash=(
                        candidate.external_skill_source_receipt_hash
                    ),
                    expected_precomputed_plan_hash=(
                        precomputed_receipts[work.target.item_id]["plan_hash"]
                    ),
                    **common,
                )
                wrapper = RecoveryBoundBackendV2(
                    delegate=delegate,
                    work=work,
                    state_root=state_root,
                    trial_root=_trial_root_for_work_v2(worker_root, work),
                    expected_process_scope=(
                        protocol.codex_agent_execution_policy
                        .action_budget_process_scope
                    ),
                    expected_plan_hash=str(
                        precomputed_receipts[work.target.item_id]["plan_hash"]
                    ),
                    expected_program_id=candidate.recipe_id,
                    expected_treatment_hash=(
                        plan.treatment.period_out_treatment_id
                    ),
                    expected_external_source_receipt_hash=(
                        candidate.external_skill_source_receipt_hash
                    ),
                )
            else:
                delegate = DurableRawSubprocessBackendV2(
                    benchmark,
                    **common,
                )
                wrapper = RecoveryBoundBackendV2(
                    delegate=delegate,
                    work=work,
                    state_root=state_root,
                    trial_root=_trial_root_for_work_v2(worker_root, work),
                    expected_process_scope=(
                        protocol.codex_agent_execution_policy
                        .action_budget_process_scope
                    ),
                )
            # Factory execution is sequential and happens for all sixteen
            # units before the executor submits its first backend call.  A
            # dirty non-recoverable unit therefore blocks the whole batch
            # before any clean unit can consume a model claim.
            wrapper.inspect_recovery()
            backends[work.work_unit_hash] = wrapper
            return wrapper

        with future_terminal_semantics_v2():
            execution = execute_measurement_plan_v2(
                plan=plan,
                backend_factory=backend_factory,
            )
        descriptive = _descriptive_results(execution)
        observations = [row.observation for row in execution.work_results]
        semantic_evidence: list[dict[str, Any]] = []
        final_decisions: list[RecoveryDecisionV2] = []
        work_by_hash = {
            work.work_unit_hash: work for work in plan.work_units
        }
        for work_hash, wrapper in sorted(backends.items()):
            work = work_by_hash[work_hash]
            decision = wrapper.inspect_recovery()
            if (
                not decision.completed
                or decision.model_calls_accounted != 1
                or decision.model_replay_authorized
            ):
                raise PeriodOutRunnerError(
                    "final durable work state did not prove one model call"
                )
            final_decisions.append(decision)
            state_root = worker_root / work_hash / "durable"
            claim_path = state_root / MODEL_EXECUTION_CLAIM_FILENAME
            if claim_path.is_symlink() or not claim_path.is_file():
                raise PeriodOutRunnerError(
                    "completed work lacks its pre-backend execution claim"
                )
            evidence_path = state_root / SEMANTIC_EVIDENCE_FILENAME
            evidence_rows = tuple(
                getattr(wrapper.delegate, "financial_runtime_evidence", ())
            )
            if work.arm == "candidate":
                receipt = read_hashed_json_v2(
                    evidence_path,
                    hash_field="receipt_hash",
                )
                evidence = receipt.get("evidence")
                if not isinstance(evidence, Mapping):
                    raise PeriodOutRunnerError(
                        "candidate runtime evidence is unavailable"
                    )
                evidence_body = dict(evidence)
                evidence_hash = evidence_body.pop("evidence_hash", None)
                if (
                    receipt.get("request_hash")
                    != work.request.request_hash
                    or receipt.get("evidence_hash") != evidence_hash
                    or stable_hash(evidence_body) != evidence_hash
                    or evidence.get("plan_hash")
                    != precomputed_receipts[work.target.item_id]["plan_hash"]
                    or evidence.get("program_id") != candidate.recipe_id
                    or evidence.get("treatment_hash")
                    != plan.treatment.period_out_treatment_id
                    or evidence.get("external_skill_source_receipt_hash")
                    != candidate.external_skill_source_receipt_hash
                    or evidence.get("online_calls") != 0
                ):
                    raise PeriodOutRunnerError(
                        "candidate runtime evidence identity drifted"
                    )
                if wrapper.backend_called and (
                    len(evidence_rows) != 1
                    or dict(evidence_rows[0]) != dict(evidence)
                ):
                    raise PeriodOutRunnerError(
                        "live candidate evidence differs from durable evidence"
                    )
                semantic_evidence.append(
                    {
                        "work_unit_hash": work_hash,
                        "evidence": dict(evidence),
                    }
                )
            elif (
                evidence_path.exists()
                or evidence_path.is_symlink()
                or evidence_rows
            ):
                raise PeriodOutRunnerError(
                    "RAW backend emitted candidate runtime evidence"
                )
        if len(final_decisions) != 16:
            raise PeriodOutRunnerError(
                "final recovery cardinality differs from frozen grid"
            )
        model_calls_this_invocation = sum(
            wrapper.backend_called for wrapper in backends.values()
        )
        recovered_work_count = 16 - model_calls_this_invocation
        closure = _artifact_closure(worker_root)
        body = {
            "runner_version": RUNNER_VERSION,
            "execution_completed": True,
            "evidence_valid": (
                descriptive["invalid_pair_count"] == 0
                and len(semantic_evidence) == 8
                and all(
                    isinstance(row, SkillLearnTrialObservation)
                    and row.raw_trial_artifacts_persisted
                    for row in observations
                )
            ),
            "execution_freeze_hash": execution_freeze["manifest_hash"],
            "measurement_view_hash": view["measurement_view_hash"],
            "prewarm_hash": prewarm["prewarm_hash"],
            "plan_hash": plan.plan_hash,
            "plan": plan.safe_payload(),
            "execution": execution.safe_payload(),
            "results": descriptive,
            "semantic_runtime_evidence": semantic_evidence,
            "semantic_runtime_evidence_set_hash": stable_hash(
                semantic_evidence
            ),
            "worker_artifact_closure": closure,
            "physical_model_call_count": 16,
            "raw_model_call_count": 8,
            "candidate_model_call_count": 8,
            "model_calls_this_invocation": model_calls_this_invocation,
            "recovered_work_count": recovered_work_count,
            "model_replay_count": 0,
            "recovery_only_invocation": False,
            "resume_invocation": destination_preexisting,
            "prior_failure_receipt_present": (
                prior_failure_receipt_present
            ),
            "model_inference_slot_limit": 16,
            "maximum_concurrent_model_calls": limiter.maximum_active,
            "all_futures_submitted_before_results_read": True,
            "independent_agent_trajectories": True,
            "independent_provider_circuit_count": 16,
            "retry_count": 0,
            "resampling_used": False,
            "mid_batch_provider_switch_used": False,
            "official_hipporag": False,
            "hipporag_status": "not_applicable_nonexecuted",
            "official_hipporag_execution_count": 0,
            "hipporag_proxy_substitution_used": False,
            "offline_evaluation_only": True,
            "prebuilt_cache_only": cache.cache_only,
            "offline_judge_only": True,
            "online_judge_calls": 0,
            "network_fallback_used": False,
            "performance_gate_applied": False,
            "performance_thresholds_bound": False,
            "promotion_authorized": False,
            "incumbent_update_authorized": False,
            "sealed_content_accessed": False,
            "project_authored_period_out_extension": True,
            "official_skilllearnbench_score": False,
            "backend_runtime_identity": backend_runtime_identity_v2(),
            "secret_value_persisted": False,
        }
        report = atomic_write_hashed_json_v2(
            destination / REPORT_FILENAME,
            body,
            hash_field="report_hash",
        )
        return report
    except Exception as exc:
        body = {
            "runner_version": RUNNER_VERSION,
            "execution_completed": False,
            "execution_started": execution_started,
            "model_replay_authorized": False,
            "model_replay_count": 0,
            "recovery_only_invocation": recover_only,
            "resume_invocation": destination_preexisting,
            "error_type": type(exc).__name__,
            "error_message_hash": stable_hash({"message": str(exc)}),
            "raw_error_persisted": False,
            "secret_value_persisted": False,
        }
        try:
            atomic_write_hashed_json_v2(
                destination / FAILURE_FILENAME,
                body,
                hash_field="report_hash",
            )
        except FileExistsError:
            # Preserve the first immutable failure receipt; a later recovery
            # invocation must not overwrite history to make the run appear
            # cleaner than it was.
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--measurement-view", type=Path, required=True)
    parser.add_argument("--prewarm", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--minilm-snapshot-root", type=Path, required=True)
    parser.add_argument("--qa-snapshot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--recover-only",
        action="store_true",
        help=(
            "reconcile existing frozen artifacts without constructing a "
            "backend or authorizing a model call"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .freeze import load_execution_freeze_v1

    args = _parser().parse_args(argv)
    freeze, candidate = load_execution_freeze_v1(
        args.execution_freeze,
        project_root=args.project_root,
    )
    report = run_measurement_v1(
        project_root=args.project_root,
        benchmark_root=args.benchmark_root,
        measurement_view_path=args.measurement_view,
        prewarm_path=args.prewarm,
        execution_freeze=freeze,
        candidate=candidate,
        env_file=args.env_file,
        minilm_snapshot_root=args.minilm_snapshot_root,
        qa_snapshot_root=args.qa_snapshot_root,
        output_root=args.output_root,
        recover_only=args.recover_only,
    )
    if report.get("report_type") == "artifact_recovery_only":
        print(
            json.dumps(
                {
                    "report_hash": report["report_hash"],
                    "recovery_completed": report["recovery_completed"],
                    "completed_work_unit_count": report[
                        "completed_work_unit_count"
                    ],
                    "unresolved_work_unit_count": report[
                        "unresolved_work_unit_count"
                    ],
                    "invalid_work_unit_count": report[
                        "invalid_work_unit_count"
                    ],
                    "model_calls_this_invocation": 0,
                    "backend_calls_this_invocation": 0,
                },
                sort_keys=True,
            )
        )
        return 0 if report["recovery_completed"] else 2
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "evidence_valid": report["evidence_valid"],
                "raw_successes": report["results"]["raw_successes"],
                "candidate_successes": report["results"][
                    "candidate_successes"
                ],
                "net_delta": report["results"]["net_delta"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
