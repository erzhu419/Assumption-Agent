from __future__ import annotations

"""Frozen 8-pair execution planning for financial-semantic replication v2.

This module owns no provider circuit, retry policy, evaluator, or promotion
decision.  It only constructs the sixteen physical RAW/candidate requests and
executes each request once behind a sixteen-party start barrier.
"""

import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from assumption_agent.benchmarks.skilllearn_compiler import (
    NO_SKILL_TREATMENT_HASH,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.models import SplitName, stable_hash


FINANCIAL_SEMANTIC_V2_PLAN_VERSION = (
    "financial_semantic_period_out_eight_pair_plan_v2"
)
MEASUREMENT_PAIR_COUNT = 8
PHYSICAL_WORK_UNIT_COUNT = MEASUREMENT_PAIR_COUNT * 2
MEASUREMENT_MAX_WORKERS = PHYSICAL_WORK_UNIT_COUNT
FINANCIAL_FAMILY = "financial-analysis"


class FinancialSemanticV2PlanError(ValueError):
    """The frozen measurement grid or its execution contract drifted."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class MeasurementTargetV2:
    """One preregistered measurement item and its immutable fold identity."""

    item_id: str
    fold_id: str
    family: str = FINANCIAL_FAMILY

    def verify(self) -> None:
        if (
            not self.item_id.strip()
            or not self.fold_id.strip()
            or self.family != FINANCIAL_FAMILY
        ):
            raise FinancialSemanticV2PlanError(
                "measurement target identity is invalid"
            )


@dataclass(frozen=True)
class FixedPeriodOutTreatmentV2:
    """Opaque identities for the unchanged candidate and new evaluator binding."""

    recipe_id: str
    program_set_hash: str
    period_out_treatment_id: str
    external_skill_source_receipt_hash: str
    candidate_skill_source: Path = field(compare=False, repr=False)

    def verify(self) -> None:
        hashes = (
            self.recipe_id,
            self.program_set_hash,
            self.period_out_treatment_id,
            self.external_skill_source_receipt_hash,
        )
        if not all(_is_sha256(value) for value in hashes):
            raise FinancialSemanticV2PlanError(
                "fixed period-out treatment identity is malformed"
            )
        if self.program_set_hash != stable_hash(
            {"recipe_ids": [self.recipe_id]}
        ):
            raise FinancialSemanticV2PlanError(
                "program set does not bind the supplied fixed recipe"
            )
        if self.period_out_treatment_id == NO_SKILL_TREATMENT_HASH:
            raise FinancialSemanticV2PlanError(
                "candidate treatment cannot equal the RAW treatment"
            )


WorkArm = Literal["raw", "candidate"]


@dataclass(frozen=True)
class MeasurementWorkUnitV2:
    arm: WorkArm
    target: MeasurementTargetV2
    request: SkillLearnTrialRequest
    skill_source_dir: Path | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    @property
    def work_unit_hash(self) -> str:
        return stable_hash(
            {
                "plan_version": FINANCIAL_SEMANTIC_V2_PLAN_VERSION,
                "arm": self.arm,
                "item_id": self.target.item_id,
                "fold_id": self.target.fold_id,
                "pair_id": self.request.pair_id,
                "request_hash": self.request.request_hash,
            }
        )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "work_unit_hash": self.work_unit_hash,
            "arm": self.arm,
            "item_id_hash": stable_hash({"item_id": self.target.item_id}),
            "fold_id_hash": stable_hash({"fold_id": self.target.fold_id}),
            "pair_id": self.request.pair_id,
            "request_hash": self.request.request_hash,
            "candidate_source_required": self.skill_source_dir is not None,
            "retry_count": 0,
            "raw_content_persisted": False,
        }


@dataclass(frozen=True)
class MeasurementPlanV2:
    manifest_hash: str
    evaluator_epoch: str
    treatment: FixedPeriodOutTreatmentV2
    work_units: tuple[MeasurementWorkUnitV2, ...]

    @property
    def plan_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        rows = [work.safe_payload() for work in self.work_units]
        return {
            "plan_version": FINANCIAL_SEMANTIC_V2_PLAN_VERSION,
            "manifest_hash": self.manifest_hash,
            "evaluator_epoch": self.evaluator_epoch,
            "recipe_id": self.treatment.recipe_id,
            "program_set_hash": self.treatment.program_set_hash,
            "period_out_treatment_id": (
                self.treatment.period_out_treatment_id
            ),
            "external_skill_source_receipt_hash": (
                self.treatment.external_skill_source_receipt_hash
            ),
            "measurement_pair_count": MEASUREMENT_PAIR_COUNT,
            "raw_execution_count": MEASUREMENT_PAIR_COUNT,
            "candidate_execution_count": MEASUREMENT_PAIR_COUNT,
            "physical_work_unit_count": PHYSICAL_WORK_UNIT_COUNT,
            "projection_count": 0,
            "official_hipporag": False,
            "official_hipporag_status": "not_applicable_nonexecuted",
            "official_hipporag_execution_count": 0,
            "maximum_workers": MEASUREMENT_MAX_WORKERS,
            "retry_policy": "none",
            "retry_count": 0,
            "backend_factory_owns_provider_circuit": True,
            "descriptive_only": True,
            "performance_gate_bound": False,
            "promotion_authorized": False,
            "work_units": rows,
            "work_unit_set_hash": stable_hash({"rows": rows}),
            "raw_content_persisted": False,
        }

    def verify(self) -> None:
        self.treatment.verify()
        if (
            not _is_sha256(self.manifest_hash)
            or not self.evaluator_epoch.strip()
            or len(self.work_units) != PHYSICAL_WORK_UNIT_COUNT
            or len({work.work_unit_hash for work in self.work_units})
            != PHYSICAL_WORK_UNIT_COUNT
        ):
            raise FinancialSemanticV2PlanError(
                "measurement plan identity or cardinality drifted"
            )

        grouped: dict[
            tuple[str, str, str], dict[WorkArm, MeasurementWorkUnitV2]
        ] = {}
        for work in self.work_units:
            work.target.verify()
            grouped.setdefault(
                (
                    work.target.item_id,
                    work.target.fold_id,
                    work.request.pair_id,
                ),
                {},
            )[work.arm] = work
        if len(grouped) != MEASUREMENT_PAIR_COUNT or any(
            set(arms) != {"raw", "candidate"}
            for arms in grouped.values()
        ):
            raise FinancialSemanticV2PlanError(
                "measurement work units do not form eight complete pairs"
            )
        if len({item_id for item_id, _, _ in grouped}) != (
            MEASUREMENT_PAIR_COUNT
        ):
            raise FinancialSemanticV2PlanError(
                "measurement items must be unique"
            )

        for (item_id, _fold_id, pair_id), arms in grouped.items():
            raw = arms["raw"]
            candidate = arms["candidate"]
            raw_request = raw.request
            candidate_request = candidate.request
            shared_identity_matches = (
                raw.target == candidate.target
                and raw_request.item_id == item_id
                and candidate_request.item_id == item_id
                and raw_request.family == FINANCIAL_FAMILY
                and candidate_request.family == FINANCIAL_FAMILY
                and raw_request.pair_id == pair_id
                and candidate_request.pair_id == pair_id
                and raw_request.split is SplitName.VALIDATION
                and candidate_request.split is SplitName.VALIDATION
                and raw_request.evaluator_epoch == self.evaluator_epoch
                and candidate_request.evaluator_epoch == self.evaluator_epoch
                and raw_request.repeat == candidate_request.repeat == 0
                and raw_request.agent_id == candidate_request.agent_id
                and raw_request.model == candidate_request.model
                and raw_request.max_steps == candidate_request.max_steps
                and raw_request.manifest_hash
                == candidate_request.manifest_hash
                == self.manifest_hash
                and raw_request.codex_agent_execution_policy_hash
                == candidate_request.codex_agent_execution_policy_hash
            )
            raw_identity_matches = (
                raw.arm == "raw"
                and raw.skill_source_dir is None
                and raw_request.variant is TrialVariant.POLICY_OFF
                and raw_request.treatment_hash
                == NO_SKILL_TREATMENT_HASH
                and raw_request.program_id is None
                and raw_request.program_set_hash == ""
                and raw_request.external_skill_source_receipt_hash == ""
            )
            candidate_identity_matches = (
                candidate.arm == "candidate"
                and candidate.skill_source_dir
                == self.treatment.candidate_skill_source
                and candidate_request.variant is TrialVariant.POLICY_ON
                and candidate_request.program_id
                == self.treatment.recipe_id
                and candidate_request.program_set_hash
                == self.treatment.program_set_hash
                and candidate_request.treatment_hash
                == self.treatment.period_out_treatment_id
                and candidate_request.external_skill_source_receipt_hash
                == self.treatment.external_skill_source_receipt_hash
            )
            if not (
                shared_identity_matches
                and raw_identity_matches
                and candidate_identity_matches
            ):
                raise FinancialSemanticV2PlanError(
                    "paired RAW/candidate request identity drifted"
                )


def build_measurement_plan_v2(
    *,
    targets: Sequence[MeasurementTargetV2],
    manifest_hash: str,
    evaluator_epoch: str,
    treatment: FixedPeriodOutTreatmentV2,
    agent_id: str,
    model: str,
    max_steps: int,
    codex_agent_execution_policy_hash: str,
) -> MeasurementPlanV2:
    """Build exactly eight RAW/candidate pairs without projections."""

    treatment.verify()
    normalized_targets = tuple(targets)
    if len(normalized_targets) != MEASUREMENT_PAIR_COUNT:
        raise FinancialSemanticV2PlanError(
            "period-out measurement requires exactly eight targets"
        )
    for target in normalized_targets:
        target.verify()
    if len({target.item_id for target in normalized_targets}) != (
        MEASUREMENT_PAIR_COUNT
    ):
        raise FinancialSemanticV2PlanError(
            "period-out measurement target IDs must be unique"
        )
    if (
        not _is_sha256(manifest_hash)
        or not _is_sha256(codex_agent_execution_policy_hash)
        or not evaluator_epoch.strip()
        or not agent_id.strip()
        or not model.strip()
        or isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps <= 0
    ):
        raise FinancialSemanticV2PlanError(
            "measurement execution identity is invalid"
        )

    work_units: list[MeasurementWorkUnitV2] = []
    for target in normalized_targets:
        pair_id = "period-out-pair-" + stable_hash(
            {
                "manifest_hash": manifest_hash,
                "item_id": target.item_id,
                "fold_id": target.fold_id,
                "repeat": 0,
            }
        )[:20]
        shared: dict[str, Any] = {
            "item_id": target.item_id,
            "family": target.family,
            "split": SplitName.VALIDATION,
            "evaluator_epoch": evaluator_epoch,
            "pair_id": pair_id,
            "repeat": 0,
            "agent_id": agent_id,
            "model": model,
            "max_steps": max_steps,
            "manifest_hash": manifest_hash,
            "codex_agent_execution_policy_hash": (
                codex_agent_execution_policy_hash
            ),
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
            external_skill_source_receipt_hash=(
                treatment.external_skill_source_receipt_hash
            ),
        )
        work_units.extend(
            (
                MeasurementWorkUnitV2(
                    arm="raw",
                    target=target,
                    request=raw,
                ),
                MeasurementWorkUnitV2(
                    arm="candidate",
                    target=target,
                    request=candidate,
                    skill_source_dir=treatment.candidate_skill_source,
                ),
            )
        )

    plan = MeasurementPlanV2(
        manifest_hash=manifest_hash,
        evaluator_epoch=evaluator_epoch,
        treatment=treatment,
        work_units=tuple(
            sorted(
                work_units,
                key=lambda work: (
                    work.target.item_id,
                    work.target.fold_id,
                    work.arm,
                ),
            )
        ),
    )
    plan.verify()
    return plan


class MeasurementBackendV2(Protocol):
    """Minimal backend boundary used by the concurrency executor."""

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> Any: ...


BackendFactoryV2 = Callable[[MeasurementWorkUnitV2], MeasurementBackendV2]


@dataclass(frozen=True)
class MeasurementWorkResultV2:
    work: MeasurementWorkUnitV2
    observation: Any = field(compare=False, repr=False)


@dataclass(frozen=True)
class MeasurementPairResultV2:
    target: MeasurementTargetV2
    pair_id: str
    raw_observation: Any = field(compare=False, repr=False)
    candidate_observation: Any = field(compare=False, repr=False)


@dataclass(frozen=True)
class MeasurementExecutionV2:
    plan: MeasurementPlanV2
    work_results: tuple[MeasurementWorkResultV2, ...]
    pair_results: tuple[MeasurementPairResultV2, ...]
    backend_instance_count: int
    barrier_participant_count: int
    maximum_active_backend_calls: int

    def safe_payload(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan.plan_hash,
            "measurement_pair_count": len(self.pair_results),
            "physical_execution_count": len(self.work_results),
            "backend_instance_count": self.backend_instance_count,
            "backend_reused": False,
            "barrier_participant_count": self.barrier_participant_count,
            "maximum_workers": MEASUREMENT_MAX_WORKERS,
            "maximum_active_backend_calls": (
                self.maximum_active_backend_calls
            ),
            "all_futures_submitted_before_results_read": True,
            "backend_factory_owns_provider_circuit": True,
            "retry_count": 0,
            "projection_count": 0,
            "official_hipporag_execution_count": 0,
            "descriptive_only": True,
            "performance_gate_bound": False,
            "promotion_authorized": False,
        }


def execute_measurement_plan_v2(
    *,
    plan: MeasurementPlanV2,
    backend_factory: BackendFactoryV2,
) -> MeasurementExecutionV2:
    """Execute every physical arm once with a sixteen-worker start barrier.

    The factory is called once per work unit before the executor starts.  It
    therefore owns provider-circuit isolation.  All futures are submitted
    before this function reads any result, and no failed future is retried.
    """

    plan.verify()
    backend_rows: list[
        tuple[MeasurementWorkUnitV2, MeasurementBackendV2]
    ] = []
    backend_ids: set[int] = set()
    for work in plan.work_units:
        backend = backend_factory(work)
        backend_id = id(backend)
        if backend_id in backend_ids:
            raise FinancialSemanticV2PlanError(
                "backend factory reused an instance across work units"
            )
        backend_ids.add(backend_id)
        backend_rows.append((work, backend))

    barrier = threading.Barrier(PHYSICAL_WORK_UNIT_COUNT)
    activity_lock = threading.Lock()
    active = 0
    maximum_active = 0
    barrier_threads: set[int] = set()

    def run_one(
        work: MeasurementWorkUnitV2,
        backend: MeasurementBackendV2,
    ) -> MeasurementWorkResultV2:
        nonlocal active, maximum_active
        with activity_lock:
            barrier_threads.add(threading.get_ident())
        barrier.wait()
        with activity_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            observation = backend.run(
                work.request,
                skill_source_dir=work.skill_source_dir,
                trace_id=(
                    "financial-semantic-v2:"
                    + work.work_unit_hash[:20]
                ),
            )
            return MeasurementWorkResultV2(
                work=work,
                observation=observation,
            )
        finally:
            with activity_lock:
                active -= 1

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MEASUREMENT_MAX_WORKERS
    ) as executor:
        futures = tuple(
            executor.submit(run_one, work, backend)
            for work, backend in backend_rows
        )
        # Do not move result reads into the submission expression: the complete
        # 16-work grid must be visible to the executor before any join.
        work_results = tuple(future.result() for future in futures)

    ordered_results = tuple(
        sorted(
            work_results,
            key=lambda result: result.work.work_unit_hash,
        )
    )
    grouped: dict[
        tuple[str, str, str], dict[WorkArm, MeasurementWorkResultV2]
    ] = {}
    for result in ordered_results:
        grouped.setdefault(
            (
                result.work.target.item_id,
                result.work.target.fold_id,
                result.work.request.pair_id,
            ),
            {},
        )[result.work.arm] = result
    pair_results = tuple(
        MeasurementPairResultV2(
            target=arms["raw"].work.target,
            pair_id=pair_id,
            raw_observation=arms["raw"].observation,
            candidate_observation=arms["candidate"].observation,
        )
        for (_item_id, _fold_id, pair_id), arms in sorted(grouped.items())
    )
    if (
        len(ordered_results) != PHYSICAL_WORK_UNIT_COUNT
        or len(pair_results) != MEASUREMENT_PAIR_COUNT
        or len(barrier_threads) != PHYSICAL_WORK_UNIT_COUNT
    ):
        raise FinancialSemanticV2PlanError(
            "measurement executor did not complete the frozen grid"
        )
    return MeasurementExecutionV2(
        plan=plan,
        work_results=ordered_results,
        pair_results=pair_results,
        backend_instance_count=len(backend_ids),
        barrier_participant_count=len(barrier_threads),
        maximum_active_backend_calls=maximum_active,
    )
