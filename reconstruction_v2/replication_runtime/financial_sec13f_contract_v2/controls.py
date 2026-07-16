from __future__ import annotations

"""Crash-durable post-promotion controls for the SEC-13F candidate.

This module deliberately contains no promotion decision and no performance
gate.  It defines the fixed 8-item skill-only/operator-only grid, consumes one
no-clobber execution claim per incremental work unit, and exposes validators
for reusing the already completed Replication-C RAW/full observations without
executing them again.
"""

import concurrent.futures
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol, Sequence

from assumption_agent.benchmarks.financial_sec13f_contract_operator_v2 import (
    NUMERIC_ENGINE,
    OPERATOR_VERSION,
    QUERY_RECEIPT_VERSION,
    payload_hash,
)
from assumption_agent.benchmarks.offline_verifier import OfflineVerifierRuntime
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    DockerEgressPolicy,
    SkillLearnSubprocessBackend,
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.models import SplitName, stable_hash
from replication_runtime.financial_semantic_v2.backends import _DurableBackendMixinV2
from replication_runtime.financial_semantic_v2.durable_state import (
    atomic_write_hashed_json_v2,
    load_durable_stage_chain_v2,
    transition_durable_stage_v2,
)


CONTROLS_RUNTIME_VERSION = "financial_sec13f_contract_controls_runtime_v1"
CONTROL_PLAN_VERSION = "financial_sec13f_contract_controls_plan_v1"
CONTROL_EXECUTION_CLAIM_VERSION = (
    "financial_sec13f_contract_control_execution_claim_v1"
)
CONTROL_EXECUTION_CLAIM_FILENAME = "control_execution_claim.json"
SKILL_ONLY_VERIFIER_NETWORK_BEFORE_FILENAME = (
    "skill_only.verifier_network.before.json"
)
SKILL_ONLY_VERIFIER_NETWORK_AFTER_FILENAME = (
    "skill_only.verifier_network.after.json"
)
CONTROL_STAGE_ORDER_V1 = (
    "planned",
    "typed_plan_ready",
    "execution_claimed",
    "agent_completed",
    "operator_completed",
    "verifier_completed",
    "observation_finalized",
)
CONTROL_ITEM_COUNT = 8
CONTROL_WORK_UNIT_COUNT = 16
CONTROL_MAX_WORKERS = 16
CONTROL_MAX_MODEL_CALLS = 8

ControlArm = Literal["skill_only", "operator_only"]
_CONTROL_ARMS = frozenset({"skill_only", "operator_only"})
_SHA256_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_RAW_KEYS = frozenset(
    {
        "answer",
        "answers",
        "answers_payload",
        "entity",
        "expected_answer",
        "expected_output",
        "gold",
        "gold_payload",
        "instruction",
        "output_payload",
        "plan",
        "query",
        "question",
        "raw_answer",
        "raw_entity",
        "raw_instruction",
        "raw_plan",
    }
)


class ControlsRuntimeError(RuntimeError):
    """A frozen controls identity or durable boundary failed closed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ControlsRuntimeError(f"{label} is not a lowercase sha256")
    return str(value)


def _reject_raw_payload(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or key.casefold() in _FORBIDDEN_RAW_KEYS:
                raise ControlsRuntimeError(
                    "control evidence contains forbidden raw payload content"
                )
            _reject_raw_payload(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_raw_payload(nested)


@dataclass(frozen=True)
class ControlTargetBindingV1:
    item_id: str = field(repr=False)
    fold_id: str
    prior_pair_id: str
    prior_raw_observation_hash: str
    prior_candidate_observation_hash: str
    prior_raw_success: bool
    prior_candidate_success: bool
    candidate_output_sha256: str
    typed_plan_hash: str
    extraction_receipt_hash: str

    @property
    def item_id_hash(self) -> str:
        return stable_hash({"item_id": self.item_id})

    def verify(self) -> None:
        if (
            not self.item_id.strip()
            or not self.fold_id.strip()
            or not self.prior_pair_id.strip()
            or not isinstance(self.prior_raw_success, bool)
            or not isinstance(self.prior_candidate_success, bool)
            or not all(
                _is_sha256(value)
                for value in (
                    self.prior_raw_observation_hash,
                    self.prior_candidate_observation_hash,
                    self.candidate_output_sha256,
                    self.typed_plan_hash,
                    self.extraction_receipt_hash,
                )
            )
        ):
            raise ControlsRuntimeError("control target binding is malformed")

    def safe_payload(self) -> dict[str, Any]:
        return {
            "item_id_hash": self.item_id_hash,
            "fold_id": self.fold_id,
            "prior_pair_id": self.prior_pair_id,
            "prior_raw_observation_hash": self.prior_raw_observation_hash,
            "prior_candidate_observation_hash": (
                self.prior_candidate_observation_hash
            ),
            "prior_raw_success": self.prior_raw_success,
            "prior_candidate_success": self.prior_candidate_success,
            "candidate_output_sha256": self.candidate_output_sha256,
            "typed_plan_hash": self.typed_plan_hash,
            "extraction_receipt_hash": self.extraction_receipt_hash,
            "raw_content_persisted": False,
        }


@dataclass(frozen=True)
class ControlWorkUnitV1:
    arm: ControlArm
    target: ControlTargetBindingV1
    treatment_hash: str
    request: SkillLearnTrialRequest | None = field(
        default=None, compare=False, repr=False
    )
    skill_source_dir: Path | None = field(
        default=None, compare=False, repr=False
    )

    @property
    def request_hash(self) -> str:
        if self.request is not None:
            return self.request.request_hash
        return stable_hash(
            {
                "controls_runtime": CONTROLS_RUNTIME_VERSION,
                "arm": self.arm,
                "item_id_hash": self.target.item_id_hash,
                "fold_id": self.target.fold_id,
                "treatment_hash": self.treatment_hash,
                "typed_plan_hash": self.target.typed_plan_hash,
            }
        )

    @property
    def trial_id(self) -> str:
        if self.request is not None:
            return self.request.trial_id
        return f"controls_operator_only_{self.request_hash[:18]}"

    @property
    def work_unit_hash(self) -> str:
        return stable_hash(
            {
                "plan_version": CONTROL_PLAN_VERSION,
                "arm": self.arm,
                "item_id_hash": self.target.item_id_hash,
                "fold_id": self.target.fold_id,
                "request_hash": self.request_hash,
                "treatment_hash": self.treatment_hash,
            }
        )

    def verify(self) -> None:
        self.target.verify()
        if self.arm not in _CONTROL_ARMS or not _is_sha256(self.treatment_hash):
            raise ControlsRuntimeError("control work identity is malformed")
        if self.arm == "skill_only":
            if (
                self.request is None
                or self.request.variant is not TrialVariant.POLICY_ON
                or self.request.treatment_hash != self.treatment_hash
                or self.skill_source_dir is None
            ):
                raise ControlsRuntimeError("skill-only work identity drifted")
        elif self.request is not None or self.skill_source_dir is not None:
            raise ControlsRuntimeError("operator-only work crossed model state")

    def safe_payload(self) -> dict[str, Any]:
        return {
            "work_unit_hash": self.work_unit_hash,
            "arm": self.arm,
            "item_id_hash": self.target.item_id_hash,
            "fold_id": self.target.fold_id,
            "request_hash": self.request_hash,
            "treatment_hash": self.treatment_hash,
            "trial_id_hash": stable_hash({"trial_id": self.trial_id}),
            "model_call_authorization_count": (
                1 if self.arm == "skill_only" else 0
            ),
            "operator_call_authorization_count": (
                1 if self.arm == "operator_only" else 0
            ),
            "offline_verifier_call_authorization_count": 1,
            "retry_count": 0,
            "raw_content_persisted": False,
        }


@dataclass(frozen=True)
class ControlPlanV1:
    controls_preregistration_hash: str
    prior_measurement_report_hash: str
    prior_measurement_plan_hash: str
    evaluator_epoch: str
    candidate_recipe_id: str
    candidate_program_set_hash: str
    candidate_treatment_hash: str
    skill_only_treatment_hash: str
    operator_only_treatment_hash: str
    external_skill_source_receipt_hash: str
    work_units: tuple[ControlWorkUnitV1, ...]

    @property
    def plan_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        rows = [work.safe_payload() for work in self.work_units]
        return {
            "plan_version": CONTROL_PLAN_VERSION,
            "controls_preregistration_hash": self.controls_preregistration_hash,
            "prior_measurement_report_hash": self.prior_measurement_report_hash,
            "prior_measurement_plan_hash": self.prior_measurement_plan_hash,
            "evaluator_epoch": self.evaluator_epoch,
            "candidate_recipe_id": self.candidate_recipe_id,
            "candidate_program_set_hash": self.candidate_program_set_hash,
            "candidate_treatment_hash": self.candidate_treatment_hash,
            "skill_only_treatment_hash": self.skill_only_treatment_hash,
            "operator_only_treatment_hash": self.operator_only_treatment_hash,
            "external_skill_source_receipt_hash": (
                self.external_skill_source_receipt_hash
            ),
            "prior_observation_reuse_count": 16,
            "prior_observation_execution_count": 0,
            "physical_control_execution_count": 16,
            "skill_only_execution_count": 8,
            "operator_only_execution_count": 8,
            "skill_only_model_call_count": 8,
            "operator_only_model_call_count": 0,
            "maximum_workers": CONTROL_MAX_WORKERS,
            "maximum_concurrent_model_calls": CONTROL_MAX_MODEL_CALLS,
            "all_futures_submitted_before_results_read": True,
            "retry_policy": "none",
            "retry_count": 0,
            "resampling_authorized": False,
            "performance_gate_bound": False,
            "promotion_authorized": False,
            "work_units": rows,
            "work_unit_set_hash": stable_hash(rows),
            "raw_content_persisted": False,
        }

    def verify(self) -> None:
        if (
            not all(
                _is_sha256(value)
                for value in (
                    self.controls_preregistration_hash,
                    self.prior_measurement_report_hash,
                    self.prior_measurement_plan_hash,
                    self.candidate_recipe_id,
                    self.candidate_program_set_hash,
                    self.candidate_treatment_hash,
                    self.skill_only_treatment_hash,
                    self.operator_only_treatment_hash,
                    self.external_skill_source_receipt_hash,
                )
            )
            or not self.evaluator_epoch.strip()
            or len(self.work_units) != CONTROL_WORK_UNIT_COUNT
            or len({work.work_unit_hash for work in self.work_units})
            != CONTROL_WORK_UNIT_COUNT
        ):
            raise ControlsRuntimeError("control plan identity drifted")
        for work in self.work_units:
            work.verify()
        grouped: dict[str, set[str]] = {}
        for work in self.work_units:
            grouped.setdefault(work.target.item_id_hash, set()).add(work.arm)
        if (
            len(grouped) != CONTROL_ITEM_COUNT
            or any(arms != _CONTROL_ARMS for arms in grouped.values())
        ):
            raise ControlsRuntimeError("control plan does not form eight arm pairs")


def build_control_plan_v1(
    *,
    targets: Sequence[ControlTargetBindingV1],
    controls_preregistration_hash: str,
    prior_measurement_report_hash: str,
    prior_measurement_plan_hash: str,
    evaluator_epoch: str,
    candidate_recipe_id: str,
    candidate_program_set_hash: str,
    candidate_treatment_hash: str,
    skill_only_treatment_hash: str,
    operator_only_treatment_hash: str,
    external_skill_source_receipt_hash: str,
    candidate_skill_source: Path,
    agent_id: str,
    model: str,
    max_steps: int,
    codex_agent_execution_policy_hash: str,
) -> ControlPlanV1:
    normalized = tuple(targets)
    if len(normalized) != CONTROL_ITEM_COUNT:
        raise ControlsRuntimeError("controls require exactly eight targets")
    for target in normalized:
        target.verify()
    if len({target.item_id for target in normalized}) != CONTROL_ITEM_COUNT:
        raise ControlsRuntimeError("control target identities are not unique")
    for value, label in (
        (controls_preregistration_hash, "controls preregistration hash"),
        (prior_measurement_report_hash, "prior report hash"),
        (prior_measurement_plan_hash, "prior plan hash"),
        (candidate_recipe_id, "candidate recipe"),
        (candidate_program_set_hash, "program set"),
        (candidate_treatment_hash, "candidate treatment"),
        (skill_only_treatment_hash, "skill-only treatment"),
        (operator_only_treatment_hash, "operator-only treatment"),
        (external_skill_source_receipt_hash, "skill source receipt"),
        (codex_agent_execution_policy_hash, "execution policy"),
    ):
        _require_sha256(value, label)
    if candidate_program_set_hash != stable_hash(
        {"recipe_ids": [candidate_recipe_id]}
    ):
        raise ControlsRuntimeError("candidate program set drifted")
    if (
        not evaluator_epoch.strip()
        or not agent_id.strip()
        or not model.strip()
        or isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps <= 0
    ):
        raise ControlsRuntimeError("control agent identity is malformed")

    rows: list[ControlWorkUnitV1] = []
    for target in normalized:
        pair_id = "controls-pair-" + stable_hash(
            {
                "controls_preregistration_hash": controls_preregistration_hash,
                "item_id_hash": target.item_id_hash,
                "fold_id": target.fold_id,
            }
        )[:20]
        request = SkillLearnTrialRequest(
            item_id=target.item_id,
            family="financial-analysis",
            split=SplitName.VALIDATION,
            variant=TrialVariant.POLICY_ON,
            evaluator_epoch=evaluator_epoch,
            pair_id=pair_id,
            repeat=0,
            agent_id=agent_id,
            model=model,
            max_steps=max_steps,
            manifest_hash=controls_preregistration_hash,
            program_id=candidate_recipe_id,
            program_set_hash=candidate_program_set_hash,
            treatment_hash=skill_only_treatment_hash,
            external_skill_source_receipt_hash=(
                external_skill_source_receipt_hash
            ),
            codex_agent_execution_policy_hash=(
                codex_agent_execution_policy_hash
            ),
        )
        rows.extend(
            (
                ControlWorkUnitV1(
                    arm="skill_only",
                    target=target,
                    treatment_hash=skill_only_treatment_hash,
                    request=request,
                    skill_source_dir=candidate_skill_source,
                ),
                ControlWorkUnitV1(
                    arm="operator_only",
                    target=target,
                    treatment_hash=operator_only_treatment_hash,
                ),
            )
        )
    result = ControlPlanV1(
        controls_preregistration_hash=controls_preregistration_hash,
        prior_measurement_report_hash=prior_measurement_report_hash,
        prior_measurement_plan_hash=prior_measurement_plan_hash,
        evaluator_epoch=evaluator_epoch,
        candidate_recipe_id=candidate_recipe_id,
        candidate_program_set_hash=candidate_program_set_hash,
        candidate_treatment_hash=candidate_treatment_hash,
        skill_only_treatment_hash=skill_only_treatment_hash,
        operator_only_treatment_hash=operator_only_treatment_hash,
        external_skill_source_receipt_hash=(
            external_skill_source_receipt_hash
        ),
        work_units=tuple(
            sorted(rows, key=lambda row: (row.target.item_id_hash, row.arm))
        ),
    )
    result.verify()
    return result


def _transition_control_stage_v1(
    state_root: Path,
    *,
    work: ControlWorkUnitV1,
    stage: str,
    payload: Mapping[str, Any],
) -> Any:
    chain = load_durable_stage_chain_v2(
        state_root,
        stage_order=CONTROL_STAGE_ORDER_V1,
        work_unit_hash=work.work_unit_hash,
        request_hash=work.request_hash,
    )
    return transition_durable_stage_v2(
        state_root,
        stage_order=CONTROL_STAGE_ORDER_V1,
        work_unit_hash=work.work_unit_hash,
        request_hash=work.request_hash,
        stage=stage,
        predecessor_stage_hash=(chain[-1].stage_hash if chain else None),
        payload=payload,
    )


def initialize_control_state_v1(
    *, state_root: str | Path, work: ControlWorkUnitV1
) -> None:
    work.verify()
    root = Path(state_root)
    planned = transition_durable_stage_v2(
        root,
        stage_order=CONTROL_STAGE_ORDER_V1,
        work_unit_hash=work.work_unit_hash,
        request_hash=work.request_hash,
        stage="planned",
        predecessor_stage_hash=None,
        payload={
            **work.safe_payload(),
            "controls_runtime_version": CONTROLS_RUNTIME_VERSION,
            "model_calls": 0,
            "operator_calls": 0,
            "retry_count": 0,
        },
    )
    transition_durable_stage_v2(
        root,
        stage_order=CONTROL_STAGE_ORDER_V1,
        work_unit_hash=work.work_unit_hash,
        request_hash=work.request_hash,
        stage="typed_plan_ready",
        predecessor_stage_hash=planned.stage_hash,
        payload={
            "arm": work.arm,
            "typed_plan_hash": work.target.typed_plan_hash,
            "extraction_receipt_hash": work.target.extraction_receipt_hash,
            "raw_typed_plan_persisted": False,
            "model_calls": 0,
            "operator_calls": 0,
        },
    )


def authorize_control_execution_once_v1(
    *, state_root: str | Path, work: ControlWorkUnitV1
) -> dict[str, Any]:
    """Permanently consume one work unit's sole execution authorization."""

    work.verify()
    root = Path(state_root)
    chain = load_durable_stage_chain_v2(
        root,
        stage_order=CONTROL_STAGE_ORDER_V1,
        work_unit_hash=work.work_unit_hash,
        request_hash=work.request_hash,
    )
    if len(chain) != 2:
        raise ControlsRuntimeError("control execution state is not clean")
    claim_path = root / CONTROL_EXECUTION_CLAIM_FILENAME
    body = {
        "claim_version": CONTROL_EXECUTION_CLAIM_VERSION,
        "controls_runtime_version": CONTROLS_RUNTIME_VERSION,
        "work_unit_hash": work.work_unit_hash,
        "request_hash": work.request_hash,
        "arm": work.arm,
        "trial_id_hash": stable_hash({"trial_id": work.trial_id}),
        "preexecution_stage_head_hash": chain[-1].stage_hash,
        "model_call_authorization_count": (
            1 if work.arm == "skill_only" else 0
        ),
        "operator_call_authorization_count": (
            1 if work.arm == "operator_only" else 0
        ),
        "offline_verifier_call_authorization_count": 1,
        "retry_authorized": False,
        "replay_authorized": False,
        "resampling_authorized": False,
        "claim_consumed_by_first_execution_only": True,
    }
    claim = atomic_write_hashed_json_v2(
        claim_path,
        body,
        hash_field="receipt_hash",
        refuse_existing=True,
    )
    try:
        transition_durable_stage_v2(
            root,
            stage_order=CONTROL_STAGE_ORDER_V1,
            work_unit_hash=work.work_unit_hash,
            request_hash=work.request_hash,
            stage="execution_claimed",
            predecessor_stage_hash=chain[-1].stage_hash,
            payload={
                "arm": work.arm,
                "execution_claim_receipt_hash": claim["receipt_hash"],
                "model_call_authorization_count": body[
                    "model_call_authorization_count"
                ],
                "operator_call_authorization_count": body[
                    "operator_call_authorization_count"
                ],
                "retry_authorized": False,
                "replay_authorized": False,
            },
        )
    except Exception as exc:
        # The no-clobber claim remains consumed.  Deleting it here would make a
        # crash window replayable, so fail closed and retain the evidence.
        raise ControlsRuntimeError(
            "control claim was consumed but its stage could not close"
        ) from exc
    return claim


@dataclass(frozen=True)
class ControlBackendResultV1:
    arm: ControlArm
    work_unit_hash: str
    request_hash: str
    observation_hash: str
    valid: bool
    success: bool
    score: float
    model_calls: int
    operator_calls: int
    online_judge_calls: int
    output_sha256: str | None
    candidate_output_hash_match: bool | None

    def verify(self, work: ControlWorkUnitV1) -> None:
        expected_model = 1 if work.arm == "skill_only" else 0
        expected_operator = 1 if work.arm == "operator_only" else 0
        if (
            self.arm != work.arm
            or self.work_unit_hash != work.work_unit_hash
            or self.request_hash != work.request_hash
            or not _is_sha256(self.observation_hash)
            or not isinstance(self.valid, bool)
            or not isinstance(self.success, bool)
            or isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or self.model_calls != expected_model
            or self.operator_calls != expected_operator
            or self.online_judge_calls != 0
        ):
            raise ControlsRuntimeError("control backend result drifted")
        if self.arm == "operator_only":
            if not _is_sha256(self.output_sha256) or not isinstance(
                self.candidate_output_hash_match, bool
            ):
                raise ControlsRuntimeError(
                    "operator-only result lacks causal output evidence"
                )
        elif (
            self.output_sha256 is not None
            or self.candidate_output_hash_match is not None
        ):
            raise ControlsRuntimeError("skill-only result crossed operator state")

    def safe_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "work_unit_hash": self.work_unit_hash,
            "request_hash": self.request_hash,
            "observation_hash": self.observation_hash,
            "valid": self.valid,
            "success": self.success,
            "score": float(self.score),
            "model_calls": self.model_calls,
            "operator_calls": self.operator_calls,
            "online_judge_calls": self.online_judge_calls,
            "output_sha256": self.output_sha256,
            "candidate_output_hash_match": self.candidate_output_hash_match,
            "raw_content_persisted": False,
        }


class ControlBackendV1(Protocol):
    def run_control(
        self,
        *,
        work: ControlWorkUnitV1,
        state_root: Path,
        trace_id: str,
    ) -> ControlBackendResultV1: ...


ControlBackendFactoryV1 = Callable[[ControlWorkUnitV1], ControlBackendV1]


@dataclass(frozen=True)
class ControlExecutionV1:
    plan: ControlPlanV1
    results: tuple[ControlBackendResultV1, ...]
    backend_instance_count: int
    barrier_participant_count: int
    maximum_active_backend_calls: int
    all_futures_submitted_before_results_read: bool = True
    retry_count: int = 0

    @property
    def model_call_count(self) -> int:
        return sum(row.model_calls for row in self.results)

    @property
    def operator_call_count(self) -> int:
        return sum(row.operator_calls for row in self.results)

    def safe_payload(self) -> dict[str, Any]:
        rows = [row.safe_payload() for row in self.results]
        return {
            "control_plan_hash": self.plan.plan_hash,
            "physical_control_execution_count": len(rows),
            "backend_instance_count": self.backend_instance_count,
            "backend_reused": False,
            "barrier_participant_count": self.barrier_participant_count,
            "maximum_workers": CONTROL_MAX_WORKERS,
            "maximum_active_backend_calls": self.maximum_active_backend_calls,
            "all_futures_submitted_before_results_read": (
                self.all_futures_submitted_before_results_read
            ),
            "model_call_count": self.model_call_count,
            "operator_call_count": self.operator_call_count,
            "offline_verifier_call_count": len(rows),
            "retry_count": self.retry_count,
            "replay_count": 0,
            "resampling_used": False,
            "performance_gate_bound": False,
            "results": rows,
            "result_set_hash": stable_hash(rows),
            "raw_content_persisted": False,
        }


def execute_control_plan_once_v1(
    *,
    plan: ControlPlanV1,
    worker_root: str | Path,
    backend_factory: ControlBackendFactoryV1,
) -> ControlExecutionV1:
    """Claim, submit and execute all 16 incremental controls exactly once."""

    plan.verify()
    root = Path(worker_root)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ControlsRuntimeError("control worker root is not a regular tree")
    root.mkdir(parents=True, exist_ok=True)

    backend_rows: list[tuple[ControlWorkUnitV1, ControlBackendV1, Path]] = []
    backend_ids: set[int] = set()
    for work in plan.work_units:
        state = root / work.work_unit_hash / "durable"
        initialize_control_state_v1(state_root=state, work=work)
        authorize_control_execution_once_v1(state_root=state, work=work)
        backend = backend_factory(work)
        if id(backend) in backend_ids:
            raise ControlsRuntimeError("control backend instance was reused")
        backend_ids.add(id(backend))
        backend_rows.append((work, backend, state))

    barrier = threading.Barrier(CONTROL_WORK_UNIT_COUNT)
    activity_lock = threading.Lock()
    barrier_threads: set[int] = set()
    active = 0
    maximum_active = 0

    def run_one(
        work: ControlWorkUnitV1,
        backend: ControlBackendV1,
        state: Path,
    ) -> ControlBackendResultV1:
        nonlocal active, maximum_active
        with activity_lock:
            barrier_threads.add(threading.get_ident())
        barrier.wait()
        with activity_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            result = backend.run_control(
                work=work,
                state_root=state,
                trace_id="sec13f-controls-v1:" + work.work_unit_hash[:20],
            )
            result.verify(work)
            return result
        finally:
            with activity_lock:
                active -= 1

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=CONTROL_MAX_WORKERS
    ) as executor:
        futures = tuple(
            executor.submit(run_one, work, backend, state)
            for work, backend, state in backend_rows
        )
        # This join intentionally occurs after the complete 16-future tuple is
        # formed.  Moving result() into submission would violate preregistration.
        results = tuple(future.result() for future in futures)
    ordered = tuple(sorted(results, key=lambda row: row.work_unit_hash))
    if (
        len(ordered) != CONTROL_WORK_UNIT_COUNT
        or len(barrier_threads) != CONTROL_WORK_UNIT_COUNT
        or sum(row.model_calls for row in ordered) != CONTROL_MAX_MODEL_CALLS
        or sum(row.operator_calls for row in ordered) != CONTROL_ITEM_COUNT
    ):
        raise ControlsRuntimeError("control executor did not close frozen grid")
    return ControlExecutionV1(
        plan=plan,
        results=ordered,
        backend_instance_count=len(backend_ids),
        barrier_participant_count=len(barrier_threads),
        maximum_active_backend_calls=maximum_active,
    )


def validate_prior_measurement_reuse_v1(
    report: Mapping[str, Any],
    *,
    targets: Sequence[ControlTargetBindingV1],
    expected_report_hash: str,
) -> dict[str, Any]:
    """Validate hash-only RAW/full reuse without opening trial artifacts."""

    _require_sha256(expected_report_hash, "expected report hash")
    body = dict(report)
    declared = body.pop("report_hash", None)
    if declared != expected_report_hash or stable_hash(body) != declared:
        raise ControlsRuntimeError("prior measurement report self hash mismatch")
    # Replication C legitimately contains a safe plan receipt.  Do not reject
    # that field name merely because raw plans are forbidden; its own explicit
    # non-persistence flag and the rest of the report remain checked below.
    for key, value in report.items():
        if key != "plan":
            _reject_raw_payload({key: value})
    safe_plan = report.get("plan")
    if not isinstance(safe_plan, Mapping) or safe_plan.get(
        "raw_content_persisted"
    ) is not False:
        # Small unit fixtures may omit the safe plan entirely.  The formal
        # caller binds its plan hash separately, but a present plan must prove
        # that no raw content was persisted.
        if safe_plan is not None:
            raise ControlsRuntimeError("prior report plan is not hash-only")
    else:
        _reject_raw_payload(safe_plan)
    if (
        report.get("runner_version")
        != "financial_sec13f_contract_fresh_runner_v2"
        or report.get("execution_completed") is not True
        or report.get("evidence_valid") is not True
        or not _is_sha256(report.get("plan_hash"))
        or report.get("physical_model_call_count") != 16
        or report.get("raw_model_call_count") != 8
        or report.get("candidate_model_call_count") != 8
        or report.get("model_replay_count") != 0
        or report.get("retry_count") != 0
        or report.get("resampling_used") is not False
        or report.get("sealed_content_accessed") is not False
        or report.get("answers_payload_persisted") is not False
        or report.get("raw_plan_persisted") is not False
    ):
        raise ControlsRuntimeError("prior measurement report is not reusable")
    results = report.get("results")
    pairs = results.get("pairs") if isinstance(results, Mapping) else None
    normalized_targets = tuple(targets)
    if (
        len(normalized_targets) != CONTROL_ITEM_COUNT
        or not isinstance(pairs, list)
        or len(pairs) != CONTROL_ITEM_COUNT
        or results.get("pair_set_hash") != stable_hash(pairs)
        or results.get("invalid_pair_count") != 0
        or results.get("raw_successes") != sum(
            target.prior_raw_success for target in normalized_targets
        )
        or results.get("candidate_successes") != sum(
            target.prior_candidate_success for target in normalized_targets
        )
    ):
        raise ControlsRuntimeError("prior pair result set drifted")
    by_item = {
        str(row.get("item_id_hash")): row
        for row in pairs
        if isinstance(row, Mapping)
    }
    if len(by_item) != CONTROL_ITEM_COUNT:
        raise ControlsRuntimeError("prior pair identities are not unique")
    reuse_rows: list[dict[str, Any]] = []
    for target in normalized_targets:
        target.verify()
        row = by_item.get(target.item_id_hash)
        if (
            not isinstance(row, Mapping)
            or row.get("fold_id") != target.fold_id
            or row.get("pair_id") != target.prior_pair_id
            or row.get("raw_observation_hash")
            != target.prior_raw_observation_hash
            or row.get("candidate_observation_hash")
            != target.prior_candidate_observation_hash
            or row.get("raw_valid") is not True
            or row.get("candidate_valid") is not True
            or row.get("raw_success") != target.prior_raw_success
            or row.get("candidate_success")
            != target.prior_candidate_success
        ):
            raise ControlsRuntimeError("prior observation reuse identity drifted")
        reuse_rows.extend(
            (
                {
                    "arm": "raw",
                    "item_id_hash": target.item_id_hash,
                    "observation_hash": target.prior_raw_observation_hash,
                    "valid": True,
                    "success": target.prior_raw_success,
                },
                {
                    "arm": "full",
                    "item_id_hash": target.item_id_hash,
                    "observation_hash": (
                        target.prior_candidate_observation_hash
                    ),
                    "valid": True,
                    "success": target.prior_candidate_success,
                },
            )
        )
    reuse_rows.sort(key=lambda row: (row["item_id_hash"], row["arm"]))
    receipt = {
        "reuse_policy": (
            "replication_c_completed_raw_and_full_without_reexecution_v1"
        ),
        "prior_report_hash": declared,
        "prior_plan_hash": report["plan_hash"],
        "reused_observation_count": 16,
        "executions_performed": 0,
        "model_calls_performed": 0,
        "operator_calls_performed": 0,
        "offline_verifier_calls_performed": 0,
        "rows": reuse_rows,
        "row_set_hash": stable_hash(reuse_rows),
        "raw_content_persisted": False,
    }
    return {**receipt, "receipt_hash": stable_hash(receipt)}


_QUERY_RECEIPT_FIELDS = frozenset(
    {
        "receipt_version",
        "operator_version",
        "candidate_id",
        "asset_manifest_hash",
        "contract_hash",
        "operator_source_sha256",
        "plan_hash",
        "numeric_engine",
        "input_file_receipts",
        "input_set_hash",
        "pre_output_exists",
        "pre_output_sha256",
        "post_output_sha256",
        "output_changed",
        "answer_key_set_hash",
        "answers_payload_persisted_in_receipt",
        "raw_entity_persisted_in_receipt",
        "network_calls",
        "model_calls",
        "verifier_content_accessed",
        "gold_content_accessed",
        "pack_content_accessed",
        "receipt_hash",
    }
)


def validate_operator_only_query_receipt_v1(
    receipt: Mapping[str, Any],
    *,
    expected_plan_hash: str,
    expected_candidate_id: str,
    expected_asset_manifest_hash: str,
    expected_contract_hash: str,
    expected_operator_source_sha256: str,
) -> dict[str, Any]:
    """Validate a physical typed-operator receipt without answer content."""

    for value, label in (
        (expected_plan_hash, "typed plan hash"),
        (expected_candidate_id, "candidate id"),
        (expected_asset_manifest_hash, "asset manifest hash"),
        (expected_contract_hash, "contract hash"),
        (expected_operator_source_sha256, "operator source hash"),
    ):
        _require_sha256(value, label)
    _reject_raw_payload(receipt)
    if set(receipt) != _QUERY_RECEIPT_FIELDS:
        raise ControlsRuntimeError("operator query receipt schema drifted")
    body = dict(receipt)
    declared = body.pop("receipt_hash", None)
    input_rows = receipt.get("input_file_receipts")
    expected_inputs = (
        ("previous", "COVERPAGE.tsv"),
        ("previous", "INFOTABLE.tsv"),
        ("current", "COVERPAGE.tsv"),
        ("current", "INFOTABLE.tsv"),
    )
    if (
        declared != payload_hash(body)
        or receipt.get("receipt_version") != QUERY_RECEIPT_VERSION
        or receipt.get("operator_version") != OPERATOR_VERSION
        or receipt.get("candidate_id") != expected_candidate_id
        or receipt.get("asset_manifest_hash")
        != expected_asset_manifest_hash
        or receipt.get("contract_hash") != expected_contract_hash
        or receipt.get("operator_source_sha256")
        != expected_operator_source_sha256
        or receipt.get("plan_hash") != expected_plan_hash
        or receipt.get("numeric_engine") != NUMERIC_ENGINE
        or not isinstance(input_rows, list)
        or len(input_rows) != len(expected_inputs)
        or receipt.get("input_set_hash") != payload_hash(input_rows)
        or not isinstance(receipt.get("pre_output_exists"), bool)
        or not isinstance(receipt.get("output_changed"), bool)
        or not _is_sha256(receipt.get("post_output_sha256"))
        or not _is_sha256(receipt.get("answer_key_set_hash"))
        or receipt.get("answers_payload_persisted_in_receipt") is not False
        or receipt.get("raw_entity_persisted_in_receipt") is not False
        or receipt.get("network_calls") != 0
        or receipt.get("model_calls") != 0
        or receipt.get("verifier_content_accessed") is not False
        or receipt.get("gold_content_accessed") is not False
        or receipt.get("pack_content_accessed") is not False
    ):
        raise ControlsRuntimeError("operator query receipt identity drifted")
    for row, (role, table) in zip(input_rows, expected_inputs):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"role", "table", "size_bytes", "file_sha256"}
            or row.get("role") != role
            or row.get("table") != table
            or isinstance(row.get("size_bytes"), bool)
            or not isinstance(row.get("size_bytes"), int)
            or int(row["size_bytes"]) <= 0
            or not _is_sha256(row.get("file_sha256"))
        ):
            raise ControlsRuntimeError("operator input receipt row drifted")
    pre_exists = bool(receipt["pre_output_exists"])
    pre_hash = receipt.get("pre_output_sha256")
    if (pre_exists and not _is_sha256(pre_hash)) or (
        not pre_exists and pre_hash is not None
    ):
        raise ControlsRuntimeError("operator pre-output receipt drifted")
    return dict(receipt)


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ControlsRuntimeError("control artifact is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _SkillOnlyOfflineVerifierProxyV1:
    """Disconnect the completed agent container before offline verification."""

    def __init__(
        self,
        delegate: Any,
        *,
        backend: "DurableSkillOnlyBackendV1",
        network_name: str,
    ) -> None:
        if not network_name.strip():
            raise ControlsRuntimeError("skill-only verifier network is missing")
        self.delegate = delegate
        self.backend = backend
        self.network_name = network_name
        self._seen: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    @property
    def _host_delegate(self) -> Any:
        host = getattr(self.delegate, "delegate", None)
        if host is None or not callable(getattr(host, "run", None)):
            raise ControlsRuntimeError(
                "skill-only verifier cannot reach the Docker delegate"
            )
        return host

    def _run_host(self, command: Sequence[str], *, label: str) -> Any:
        completed = self._host_delegate.run(
            [str(value) for value in command],
            check=False,
            capture_output=True,
            text=True,
        )
        returncode = getattr(completed, "returncode", None)
        if isinstance(returncode, bool) or returncode != 0:
            raise ControlsRuntimeError(label)
        return completed

    def _assert_disconnected(self, container: str, *, phase: str) -> None:
        inspected = self._run_host(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .NetworkSettings.Networks}}",
                container,
            ],
            label=f"skill-only verifier {phase} network inspection failed",
        )
        raw = str(getattr(inspected, "stdout", "") or "").strip()
        try:
            networks = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ControlsRuntimeError(
                f"skill-only verifier {phase} network receipt is malformed"
            ) from exc
        if networks != {}:
            raise ControlsRuntimeError(
                f"skill-only verifier network remained attached {phase}"
            )

    def _write_network_receipt(
        self,
        *,
        path: Path,
        container: str,
        phase: str,
    ) -> dict[str, Any]:
        return atomic_write_hashed_json_v2(
            path,
            {
                "controls_runtime_version": CONTROLS_RUNTIME_VERSION,
                "arm": "skill_only",
                "phase": phase,
                "container_name_hash": stable_hash(
                    {"container_name": container}
                ),
                "disconnected_network_name_hash": stable_hash(
                    {"network_name": self.network_name}
                ),
                "attached_network_count": 0,
                "verifier_network": "none",
                "model_secret_env_available_to_verifier": False,
                "raw_content_persisted": False,
            },
            hash_field="receipt_hash",
            refuse_existing=True,
        )

    def run(self, args: Any, *positional: Any, **kwargs: Any) -> Any:
        command = list(args) if isinstance(args, (list, tuple)) else args
        if not (
            isinstance(command, list)
            and len(command) >= 4
            and command[:2] == ["docker", "exec"]
            and "/tests/test.sh" in {str(value) for value in command[3:]}
        ):
            return self.delegate.run(command, *positional, **kwargs)
        container = str(command[2])
        if container in self._seen:
            raise ControlsRuntimeError("skill-only verifier replay is forbidden")
        self._seen.add(container)
        self.backend._checkpoint_raw_before_verifier_v2()
        self._run_host(
            [
                "docker",
                "network",
                "disconnect",
                "--force",
                self.network_name,
                container,
            ],
            label="skill-only verifier network disconnect failed",
        )
        self._assert_disconnected(container, phase="before")
        before = self._write_network_receipt(
            path=(
                self.backend.durable_state_root
                / SKILL_ONLY_VERIFIER_NETWORK_BEFORE_FILENAME
            ),
            container=container,
            phase="before_verifier",
        )
        self.backend._verifier_network_before_receipt_hash = before[
            "receipt_hash"
        ]
        completed = self.delegate.run(command, *positional, **kwargs)
        self._assert_disconnected(container, phase="after")
        after = self._write_network_receipt(
            path=(
                self.backend.durable_state_root
                / SKILL_ONLY_VERIFIER_NETWORK_AFTER_FILENAME
            ),
            container=container,
            phase="after_verifier",
        )
        self.backend._verifier_network_after_receipt_hash = after[
            "receipt_hash"
        ]
        return completed


class DurableSkillOnlyBackendV1(
    _DurableBackendMixinV2,
    SkillLearnSubprocessBackend,
):
    """Run candidate SKILL policy-on with the post-agent operator absent."""

    def __init__(
        self,
        *args: Any,
        control_work: ControlWorkUnitV1,
        durable_state_root: str | Path,
        **kwargs: Any,
    ) -> None:
        control_work.verify()
        if control_work.arm != "skill_only" or control_work.request is None:
            raise ControlsRuntimeError("skill-only backend received wrong arm")
        super().__init__(*args, **kwargs)
        self.control_work = control_work
        self.durable_state_root = Path(durable_state_root).resolve()
        self.durable_work_unit_hash = control_work.work_unit_hash
        self.durable_request_hash = control_work.request_hash
        self.durable_arm = "skill_only"
        self.expected_program_id = control_work.request.program_id
        self.expected_program_set_hash = control_work.request.program_set_hash
        self.expected_treatment_hash = control_work.request.treatment_hash
        self.expected_external_skill_source_receipt_hash = (
            control_work.request.external_skill_source_receipt_hash
        )
        self._active_request: SkillLearnTrialRequest | None = None
        self._verifier_network_before_receipt_hash: str | None = None
        self._verifier_network_after_receipt_hash: str | None = None

    def _durable_chain(self) -> tuple[Any, ...]:
        return load_durable_stage_chain_v2(
            self.durable_state_root,
            stage_order=CONTROL_STAGE_ORDER_V1,
            work_unit_hash=self.durable_work_unit_hash,
            request_hash=self.durable_request_hash,
        )

    def _transition_next(
        self, stage: str, payload: Mapping[str, Any]
    ) -> Any:
        chain = self._durable_chain()
        return transition_durable_stage_v2(
            self.durable_state_root,
            stage_order=CONTROL_STAGE_ORDER_V1,
            work_unit_hash=self.durable_work_unit_hash,
            request_hash=self.durable_request_hash,
            stage=stage,
            predecessor_stage_hash=(chain[-1].stage_hash if chain else None),
            payload=payload,
        )

    def _trial_path(self, request: SkillLearnTrialRequest) -> Path:
        if not isinstance(self.trials_dir, Path):
            raise ControlsRuntimeError(
                "skill-only backend requires an explicit trials directory"
            )
        return (
            self.trials_dir
            / "assumption-agent-v2-challenger"
            / request.family
            / request.item_id
            / request.trial_id
        )

    def _checkpoint_raw_before_verifier_v2(self) -> None:
        """Proxy callback: close agent then prove the operator is knocked out."""

        request = self._active_request
        if not isinstance(request, SkillLearnTrialRequest):
            raise ControlsRuntimeError(
                "skill-only verifier started without active request"
            )
        if [row.stage for row in self._durable_chain()] != list(
            CONTROL_STAGE_ORDER_V1[:3]
        ):
            raise ControlsRuntimeError(
                "skill-only verifier started at unexpected durable stage"
            )
        self._transition_next(
            "agent_completed",
            self._agent_completion_payload(
                request,
                reconciled_after_backend_return=False,
            ),
        )
        self._transition_next(
            "operator_completed",
            {
                "arm": "skill_only",
                "applicable": False,
                "operator_enabled": False,
                "operator_calls": 0,
                "operator_knockout_enforced": True,
                "persisted_before_verifier": True,
                "typed_plan_executed": False,
            },
        )

    @contextmanager
    def _verifier_isolation(
        self,
        runner: Any,
        *,
        agent_runtime_volume: str | None = None,
        egress_policy: DockerEgressPolicy,
        offline_verifier_runtime: OfflineVerifierRuntime | None = None,
        trace_id: str = "sec13f-controls-skill-only",
    ) -> Iterator[None]:
        with super()._verifier_isolation(
            runner,
            agent_runtime_volume=agent_runtime_volume,
            egress_policy=egress_policy,
            offline_verifier_runtime=offline_verifier_runtime,
            trace_id=trace_id,
        ):
            base_proxy = runner.subprocess
            runner.subprocess = _SkillOnlyOfflineVerifierProxyV1(
                base_proxy,
                backend=self,
                network_name=egress_policy.network_name,
            )
            try:
                yield
            finally:
                runner.subprocess = base_proxy

    def _complete_control_observation(
        self,
        request: SkillLearnTrialRequest,
        observation: SkillLearnTrialObservation,
    ) -> None:
        if [row.stage for row in self._durable_chain()] != list(
            CONTROL_STAGE_ORDER_V1[:5]
        ):
            raise ControlsRuntimeError(
                "skill-only backend returned before causal checkpoint"
            )
        verifier = self._trial_path(request) / "verifier"
        reward = verifier / "reward.txt"
        ctrf = verifier / "ctrf.json"
        before_network = self._verifier_network_before_receipt_hash
        after_network = self._verifier_network_after_receipt_hash
        if not _is_sha256(before_network) or not _is_sha256(after_network):
            raise ControlsRuntimeError(
                "skill-only verifier network-none evidence is incomplete"
            )
        self._transition_next(
            "verifier_completed",
            {
                "offline": True,
                "verifier_network": "none",
                "verifier_network_before_receipt_hash": before_network,
                "verifier_network_after_receipt_hash": after_network,
                "reward_sha256": _sha256_file(reward),
                "ctrf_sha256": _sha256_file(ctrf),
                "online_judge_calls": 0,
                "operator_calls_before_verifier": 0,
            },
        )
        receipt = atomic_write_hashed_json_v2(
            self.durable_state_root / "observation.json",
            {
                "request_hash": request.request_hash,
                "observation": observation.to_dict(),
                "observation_hash": observation.observation_hash,
                "arm": "skill_only",
                "secret_value_persisted": False,
                "raw_content_persisted": False,
            },
            hash_field="receipt_hash",
            refuse_existing=True,
        )
        self._transition_next(
            "observation_finalized",
            {
                "observation_hash": observation.observation_hash,
                "observation_receipt_hash": receipt["receipt_hash"],
                "valid": observation.valid,
                "success": observation.success,
                "score": observation.score,
                "model_calls": 1,
                "operator_calls": 0,
                "online_judge_calls": 0,
            },
        )

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        work = self.control_work
        if (
            work.arm != "skill_only"
            or work.request is None
            or request != work.request
            or request.request_hash != work.request_hash
            or request.variant is not TrialVariant.POLICY_ON
            or request.program_id != self.expected_program_id
            or request.program_set_hash != self.expected_program_set_hash
            or request.treatment_hash != self.expected_treatment_hash
            or request.external_skill_source_receipt_hash
            != self.expected_external_skill_source_receipt_hash
            or skill_source_dir is None
        ):
            raise ControlsRuntimeError("skill-only arm identity or source drifted")
        self._active_request = request
        try:
            observation = super().run(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
            self._complete_control_observation(request, observation)
            return observation
        finally:
            self._active_request = None

    def run_control(
        self,
        *,
        work: ControlWorkUnitV1,
        state_root: Path,
        trace_id: str,
    ) -> ControlBackendResultV1:
        if (
            work != self.control_work
            or state_root.resolve() != self.durable_state_root
            or work.request is None
        ):
            raise ControlsRuntimeError("skill-only executor/backend identity drifted")
        observation = self.run(
            work.request,
            skill_source_dir=work.skill_source_dir,
            trace_id=trace_id,
        )
        return ControlBackendResultV1(
            arm="skill_only",
            work_unit_hash=work.work_unit_hash,
            request_hash=work.request_hash,
            observation_hash=observation.observation_hash,
            valid=observation.valid,
            success=observation.success,
            score=observation.score,
            model_calls=1,
            operator_calls=0,
            online_judge_calls=0,
            output_sha256=None,
            candidate_output_hash_match=None,
        )


__all__ = [
    "CONTROL_EXECUTION_CLAIM_FILENAME",
    "CONTROL_STAGE_ORDER_V1",
    "CONTROLS_RUNTIME_VERSION",
    "ControlBackendResultV1",
    "ControlExecutionV1",
    "ControlPlanV1",
    "ControlTargetBindingV1",
    "ControlWorkUnitV1",
    "ControlsRuntimeError",
    "DurableSkillOnlyBackendV1",
    "authorize_control_execution_once_v1",
    "build_control_plan_v1",
    "execute_control_plan_once_v1",
    "initialize_control_state_v1",
    "validate_operator_only_query_receipt_v1",
    "validate_prior_measurement_reuse_v1",
]
