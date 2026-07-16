from __future__ import annotations

"""Fixed four-pair/eight-call plan for the Replication-C sealed evaluation."""

import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any, Callable, Literal, Protocol, Sequence

from assumption_agent.benchmarks.skilllearn_compiler import NO_SKILL_TREATMENT_HASH
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.models import SplitName, stable_hash

from replication_runtime.financial_semantic_v2.plan import FixedPeriodOutTreatmentV2


PLAN_VERSION = "financial_sec13f_replication_c_sealed_four_pair_plan_v1"
PAIR_COUNT = 4
WORK_UNIT_COUNT = 8
MAXIMUM_WORKERS = 8
FAMILY = "financial-analysis"


class SealedPlanError(ValueError):
    """The sealed execution grid drifted."""


@dataclass(frozen=True)
class SealedTargetV1:
    item_id: str
    replicate: int
    family: str = FAMILY

    @property
    def fold_id(self) -> str:
        # Compatibility with the frozen descriptive-result helper.  Sealed
        # replicates are one final-test stratum, not development folds.
        return "sealed-test"

    def verify(self) -> None:
        if (
            not self.item_id.strip()
            or self.family != FAMILY
            or isinstance(self.replicate, bool)
            or self.replicate not in range(4)
        ):
            raise SealedPlanError("sealed target identity is invalid")


Arm = Literal["raw", "candidate"]


@dataclass(frozen=True)
class SealedWorkUnitV1:
    arm: Arm
    target: SealedTargetV1
    request: SkillLearnTrialRequest
    skill_source_dir: Path | None = field(default=None, compare=False, repr=False)

    @property
    def work_unit_hash(self) -> str:
        return stable_hash(
            {
                "plan_version": PLAN_VERSION,
                "arm": self.arm,
                "item_id": self.target.item_id,
                "replicate": self.target.replicate,
                "pair_id": self.request.pair_id,
                "request_hash": self.request.request_hash,
            }
        )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "work_unit_hash": self.work_unit_hash,
            "arm": self.arm,
            "item_id_hash": stable_hash({"item_id": self.target.item_id}),
            "sealed_replicate": self.target.replicate,
            "pair_id": self.request.pair_id,
            "request_hash": self.request.request_hash,
            "candidate_source_required": self.skill_source_dir is not None,
            "retry_count": 0,
            "raw_content_persisted": False,
        }


@dataclass(frozen=True)
class SealedPlanV1:
    manifest_hash: str
    evaluator_epoch: str
    treatment: FixedPeriodOutTreatmentV2
    work_units: tuple[SealedWorkUnitV1, ...]

    @property
    def plan_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        rows = [work.safe_payload() for work in self.work_units]
        return {
            "plan_version": PLAN_VERSION,
            "manifest_hash": self.manifest_hash,
            "evaluator_epoch": self.evaluator_epoch,
            "recipe_id": self.treatment.recipe_id,
            "program_set_hash": self.treatment.program_set_hash,
            "period_out_treatment_id": self.treatment.period_out_treatment_id,
            "external_skill_source_receipt_hash": self.treatment.external_skill_source_receipt_hash,
            "sealed_pair_count": PAIR_COUNT,
            "raw_execution_count": PAIR_COUNT,
            "candidate_execution_count": PAIR_COUNT,
            "physical_work_unit_count": WORK_UNIT_COUNT,
            "maximum_workers": MAXIMUM_WORKERS,
            "model_inference_slots": MAXIMUM_WORKERS,
            "arms": ["raw", "candidate"],
            "split": SplitName.TEST.value,
            "retry_policy": "none",
            "retry_count": 0,
            "replay_authorized": False,
            "resampling_authorized": False,
            "provider_switch_authorized": False,
            "offline_evaluation_only": True,
            "online_judge_calls": 0,
            "official_hipporag": False,
            "official_hipporag_status": "not_applicable_nonexecuted",
            "official_hipporag_execution_count": 0,
            "single_execution": True,
            "performance_gate_bound": False,
            "work_units": rows,
            "work_unit_set_hash": stable_hash({"rows": rows}),
            "raw_content_persisted": False,
        }

    def verify(self) -> None:
        self.treatment.verify()
        if (
            len(self.manifest_hash) != 64
            or len(self.work_units) != WORK_UNIT_COUNT
            or len({work.work_unit_hash for work in self.work_units}) != WORK_UNIT_COUNT
        ):
            raise SealedPlanError("sealed plan cardinality drifted")
        grouped: dict[tuple[str, str], set[str]] = {}
        for work in self.work_units:
            work.target.verify()
            if work.request.split is not SplitName.TEST:
                raise SealedPlanError("sealed request is not test-only")
            grouped.setdefault((work.target.item_id, work.request.pair_id), set()).add(work.arm)
        if len(grouped) != PAIR_COUNT or any(arms != {"raw", "candidate"} for arms in grouped.values()):
            raise SealedPlanError("sealed work units are not four complete pairs")


def build_sealed_plan_v1(
    *,
    targets: Sequence[SealedTargetV1],
    manifest_hash: str,
    evaluator_epoch: str,
    treatment: FixedPeriodOutTreatmentV2,
    agent_id: str,
    model: str,
    max_steps: int,
    codex_agent_execution_policy_hash: str,
) -> SealedPlanV1:
    treatment.verify()
    normalized = tuple(targets)
    if len(normalized) != PAIR_COUNT or len({row.item_id for row in normalized}) != PAIR_COUNT:
        raise SealedPlanError("sealed evaluation requires four unique targets")
    work_units: list[SealedWorkUnitV1] = []
    for target in normalized:
        target.verify()
        pair_id = "sealed-pair-" + stable_hash(
            {
                "manifest_hash": manifest_hash,
                "item_id": target.item_id,
                "replicate": target.replicate,
                "repeat": 0,
            }
        )[:20]
        shared = {
            "item_id": target.item_id,
            "family": target.family,
            "split": SplitName.TEST,
            "evaluator_epoch": evaluator_epoch,
            "pair_id": pair_id,
            "repeat": 0,
            "agent_id": agent_id,
            "model": model,
            "max_steps": max_steps,
            "manifest_hash": manifest_hash,
            "codex_agent_execution_policy_hash": codex_agent_execution_policy_hash,
        }
        raw = SkillLearnTrialRequest(
            **shared,
            variant=TrialVariant.POLICY_OFF,
            treatment_hash=NO_SKILL_TREATMENT_HASH,
        )
        candidate = SkillLearnTrialRequest(
            **shared,
            variant=TrialVariant.POLICY_ON,
            program_id=treatment.recipe_id,
            program_set_hash=treatment.program_set_hash,
            treatment_hash=treatment.period_out_treatment_id,
            external_skill_source_receipt_hash=treatment.external_skill_source_receipt_hash,
        )
        work_units.extend(
            (
                SealedWorkUnitV1("raw", target, raw),
                SealedWorkUnitV1(
                    "candidate", target, candidate, treatment.candidate_skill_source
                ),
            )
        )
    plan = SealedPlanV1(
        manifest_hash=manifest_hash,
        evaluator_epoch=evaluator_epoch,
        treatment=treatment,
        work_units=tuple(sorted(work_units, key=lambda row: (row.target.replicate, row.arm))),
    )
    plan.verify()
    return plan


class Backend(Protocol):
    def run(self, request: SkillLearnTrialRequest, *, skill_source_dir: Path | None, trace_id: str) -> Any: ...


@dataclass(frozen=True)
class SealedWorkResultV1:
    work: SealedWorkUnitV1
    observation: Any = field(compare=False, repr=False)


@dataclass(frozen=True)
class SealedPairResultV1:
    target: SealedTargetV1
    pair_id: str
    raw_observation: Any = field(compare=False, repr=False)
    candidate_observation: Any = field(compare=False, repr=False)


@dataclass(frozen=True)
class SealedExecutionV1:
    plan: SealedPlanV1
    work_results: tuple[SealedWorkResultV1, ...]
    pair_results: tuple[SealedPairResultV1, ...]
    maximum_active_backend_calls: int

    def safe_payload(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan.plan_hash,
            "sealed_pair_count": len(self.pair_results),
            "physical_execution_count": len(self.work_results),
            "backend_instance_count": len(self.work_results),
            "backend_reused": False,
            "barrier_participant_count": len(self.work_results),
            "maximum_workers": MAXIMUM_WORKERS,
            "maximum_active_backend_calls": self.maximum_active_backend_calls,
            "all_futures_submitted_before_results_read": True,
            "retry_count": 0,
            "replay_count": 0,
            "resampling_used": False,
            "provider_switch_used": False,
            "official_hipporag_execution_count": 0,
            "performance_gate_bound": False,
        }


def execute_sealed_plan_v1(
    *,
    plan: SealedPlanV1,
    backend_factory: Callable[[SealedWorkUnitV1], Backend],
) -> SealedExecutionV1:
    """Submit all eight one-shot calls before joining any result."""

    plan.verify()
    rows = [(work, backend_factory(work)) for work in plan.work_units]
    if len({id(backend) for _, backend in rows}) != WORK_UNIT_COUNT:
        raise SealedPlanError("sealed backend instance was reused")
    barrier = threading.Barrier(WORK_UNIT_COUNT)
    lock = threading.Lock()
    active = maximum_active = 0

    def run_one(work: SealedWorkUnitV1, backend: Backend) -> SealedWorkResultV1:
        nonlocal active, maximum_active
        barrier.wait()
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            observation = backend.run(
                work.request,
                skill_source_dir=work.skill_source_dir,
                trace_id="financial-semantic-v2:" + work.work_unit_hash[:20],
            )
            return SealedWorkResultV1(work, observation)
        finally:
            with lock:
                active -= 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAXIMUM_WORKERS) as executor:
        futures = tuple(executor.submit(run_one, work, backend) for work, backend in rows)
        results = tuple(future.result() for future in futures)
    grouped: dict[str, dict[str, SealedWorkResultV1]] = {}
    for result in results:
        grouped.setdefault(result.work.request.pair_id, {})[result.work.arm] = result
    pairs = tuple(
        SealedPairResultV1(
            target=arms["raw"].work.target,
            pair_id=pair_id,
            raw_observation=arms["raw"].observation,
            candidate_observation=arms["candidate"].observation,
        )
        for pair_id, arms in sorted(grouped.items())
    )
    if len(results) != WORK_UNIT_COUNT or len(pairs) != PAIR_COUNT:
        raise SealedPlanError("sealed executor did not complete the fixed grid")
    return SealedExecutionV1(plan, results, pairs, maximum_active)
