from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

import pytest

from assumption_agent.benchmarks.skilllearn_compiler import (
    NO_SKILL_TREATMENT_HASH,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import TrialVariant
from assumption_agent.models import stable_hash
from replication_runtime.financial_semantic_v2.plan import (
    FinancialSemanticV2PlanError,
    FixedPeriodOutTreatmentV2,
    MEASUREMENT_MAX_WORKERS,
    MEASUREMENT_PAIR_COUNT,
    MeasurementTargetV2,
    PHYSICAL_WORK_UNIT_COUNT,
    build_measurement_plan_v2,
    execute_measurement_plan_v2,
)


def _sha(label: str) -> str:
    return stable_hash({"label": label})


def _treatment() -> FixedPeriodOutTreatmentV2:
    recipe_id = _sha("recipe")
    return FixedPeriodOutTreatmentV2(
        recipe_id=recipe_id,
        program_set_hash=stable_hash({"recipe_ids": [recipe_id]}),
        period_out_treatment_id=_sha("period-out-treatment"),
        external_skill_source_receipt_hash=_sha("source-receipt"),
        candidate_skill_source=Path("/frozen/candidate"),
    )


def _plan():
    return build_measurement_plan_v2(
        targets=tuple(
            MeasurementTargetV2(
                item_id=f"sec13f-period-out-{index:02d}",
                fold_id=f"measurement-fold-{index:02d}",
            )
            for index in range(MEASUREMENT_PAIR_COUNT)
        ),
        manifest_hash=_sha("manifest"),
        evaluator_epoch="financial-semantic-sec13f-periodout-test",
        treatment=_treatment(),
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        codex_agent_execution_policy_hash=_sha("execution-policy"),
    )


def test_builds_exactly_eight_physical_pairs_without_projection_or_hippo() -> None:
    plan = _plan()
    plan.verify()

    assert len(plan.work_units) == PHYSICAL_WORK_UNIT_COUNT == 16
    assert len({work.work_unit_hash for work in plan.work_units}) == 16
    assert sum(work.arm == "raw" for work in plan.work_units) == 8
    assert sum(work.arm == "candidate" for work in plan.work_units) == 8

    by_item: dict[str, dict[str, object]] = {}
    for work in plan.work_units:
        by_item.setdefault(work.target.item_id, {})[work.arm] = work
    assert len(by_item) == 8
    for arms in by_item.values():
        raw = arms["raw"]
        candidate = arms["candidate"]
        assert raw.target == candidate.target
        assert raw.request.pair_id == candidate.request.pair_id
        assert raw.request.item_id == candidate.request.item_id
        assert raw.request.variant is TrialVariant.POLICY_OFF
        assert raw.request.treatment_hash == NO_SKILL_TREATMENT_HASH
        assert raw.skill_source_dir is None
        assert candidate.request.variant is TrialVariant.POLICY_ON
        assert candidate.request.program_id == plan.treatment.recipe_id
        assert (
            candidate.request.program_set_hash
            == plan.treatment.program_set_hash
        )
        assert (
            candidate.request.treatment_hash
            == plan.treatment.period_out_treatment_id
        )
        assert (
            candidate.request.external_skill_source_receipt_hash
            == plan.treatment.external_skill_source_receipt_hash
        )
        assert (
            candidate.skill_source_dir
            == plan.treatment.candidate_skill_source
        )

    payload = plan.safe_payload()
    assert payload["projection_count"] == 0
    assert payload["official_hipporag"] is False
    assert payload["official_hipporag_execution_count"] == 0
    assert payload["retry_count"] == 0
    assert payload["descriptive_only"] is True
    assert payload["performance_gate_bound"] is False
    assert payload["promotion_authorized"] is False


def test_builder_rejects_wrong_cardinality_duplicate_items_and_bad_program_set() -> None:
    targets = tuple(
        MeasurementTargetV2(item_id=f"item-{index}", fold_id="fold")
        for index in range(7)
    )
    with pytest.raises(FinancialSemanticV2PlanError, match="exactly eight"):
        build_measurement_plan_v2(
            targets=targets,
            manifest_hash=_sha("manifest"),
            evaluator_epoch="epoch",
            treatment=_treatment(),
            agent_id="codex",
            model="gpt-5.4-mini",
            max_steps=100,
            codex_agent_execution_policy_hash=_sha("policy"),
        )

    duplicated = tuple(
        MeasurementTargetV2(
            item_id="duplicate" if index == 7 else f"item-{index}",
            fold_id=f"fold-{index}",
        )
        for index in range(8)
    )
    duplicated = (*duplicated[:-1], MeasurementTargetV2("item-0", "other"))
    with pytest.raises(FinancialSemanticV2PlanError, match="must be unique"):
        build_measurement_plan_v2(
            targets=duplicated,
            manifest_hash=_sha("manifest"),
            evaluator_epoch="epoch",
            treatment=_treatment(),
            agent_id="codex",
            model="gpt-5.4-mini",
            max_steps=100,
            codex_agent_execution_policy_hash=_sha("policy"),
        )

    treatment = _treatment()
    bad_treatment = FixedPeriodOutTreatmentV2(
        recipe_id=treatment.recipe_id,
        program_set_hash=_sha("wrong-program-set"),
        period_out_treatment_id=treatment.period_out_treatment_id,
        external_skill_source_receipt_hash=(
            treatment.external_skill_source_receipt_hash
        ),
        candidate_skill_source=treatment.candidate_skill_source,
    )
    with pytest.raises(FinancialSemanticV2PlanError, match="fixed recipe"):
        bad_treatment.verify()


@dataclass(frozen=True)
class _Observation:
    success: bool
    valid: bool = True


class _Backend:
    def __init__(
        self,
        *,
        serial: int,
        factory_count: list[int],
        calls: list[tuple[int, str, Path | None]],
        calls_lock: threading.Lock,
        provider_circuit: object,
    ) -> None:
        self.serial = serial
        self.factory_count = factory_count
        self.calls = calls
        self.calls_lock = calls_lock
        self.provider_circuit = provider_circuit

    def run(self, request, *, skill_source_dir, trace_id):
        # Every backend is constructed before the 16-party executor barrier is
        # released and before any backend result can be observed.
        assert self.factory_count[0] == PHYSICAL_WORK_UNIT_COUNT
        assert trace_id.startswith("financial-semantic-v2:")
        with self.calls_lock:
            self.calls.append(
                (self.serial, request.request_hash, skill_source_dir)
            )
        return _Observation(success=request.variant is TrialVariant.POLICY_ON)


def test_executor_uses_unique_backends_one_call_each_and_sixteen_party_barrier() -> None:
    plan = _plan()
    factory_count = [0]
    calls: list[tuple[int, str, Path | None]] = []
    calls_lock = threading.Lock()
    circuits: list[object] = []
    backends: list[_Backend] = []

    def factory(_work):
        factory_count[0] += 1
        circuit = object()
        circuits.append(circuit)
        backend = _Backend(
            serial=factory_count[0],
            factory_count=factory_count,
            calls=calls,
            calls_lock=calls_lock,
            provider_circuit=circuit,
        )
        backends.append(backend)
        return backend

    result = execute_measurement_plan_v2(
        plan=plan,
        backend_factory=factory,
    )

    assert factory_count[0] == 16
    assert len(backends) == len({id(backend) for backend in backends}) == 16
    assert len(circuits) == len({id(circuit) for circuit in circuits}) == 16
    assert len(calls) == 16
    assert len({serial for serial, _, _ in calls}) == 16
    assert len({request_hash for _, request_hash, _ in calls}) == 16
    assert result.backend_instance_count == 16
    assert result.barrier_participant_count == 16
    assert len(result.work_results) == 16
    assert len(result.pair_results) == 8
    assert MEASUREMENT_MAX_WORKERS == 16

    payload = result.safe_payload()
    assert payload["all_futures_submitted_before_results_read"] is True
    assert payload["backend_factory_owns_provider_circuit"] is True
    assert payload["retry_count"] == 0
    assert payload["projection_count"] == 0
    assert payload["descriptive_only"] is True


def test_executor_rejects_backend_reuse_before_starting_any_work() -> None:
    plan = _plan()
    calls: list[object] = []

    class ReusedBackend:
        def run(self, request, *, skill_source_dir, trace_id):
            calls.append(request)
            return _Observation(success=False)

    reused = ReusedBackend()
    with pytest.raises(FinancialSemanticV2PlanError, match="reused"):
        execute_measurement_plan_v2(
            plan=plan,
            backend_factory=lambda _work: reused,
        )
    assert calls == []


def test_executor_never_retries_a_failed_physical_work_unit() -> None:
    plan = _plan()
    call_counts: dict[str, int] = {}
    lock = threading.Lock()
    serial = [0]

    class SometimesFailingBackend:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def run(self, request, *, skill_source_dir, trace_id):
            with lock:
                call_counts[request.request_hash] = (
                    call_counts.get(request.request_hash, 0) + 1
                )
            if self.fail:
                raise RuntimeError("one physical call failed")
            return _Observation(success=False)

    def factory(_work):
        serial[0] += 1
        return SometimesFailingBackend(fail=serial[0] == 1)

    with pytest.raises(RuntimeError, match="one physical call failed"):
        execute_measurement_plan_v2(
            plan=plan,
            backend_factory=factory,
        )
    assert len(call_counts) == PHYSICAL_WORK_UNIT_COUNT
    assert set(call_counts.values()) == {1}
