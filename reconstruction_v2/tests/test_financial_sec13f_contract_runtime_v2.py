from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import threading
from typing import Any

import pytest

from assumption_agent.benchmarks.financial_sec13f_contract_integration_v2 import (
    SharedFinancialSec13FContractPlannerV2,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.models import SplitName, stable_hash
from replication_runtime.financial_sec13f_contract_v2.backends import (
    DurableFinancialSec13FContractBackendV2,
    DurableRawSubprocessBackendV2,
    FinancialSemanticReplicationBackendError,
)
from replication_runtime.financial_sec13f_contract_v2 import runner
from replication_runtime.financial_sec13f_contract_v2.treatment import (
    ContractTreatmentError,
    build_evaluation_treatment_v2,
    load_fixed_contract_candidate_v2,
    validate_evaluation_treatment_v2,
)
from replication_runtime.financial_semantic_v2.pack import payload_hash
from replication_runtime.financial_semantic_v2.plan import (
    FixedPeriodOutTreatmentV2,
    MeasurementTargetV2,
    build_measurement_plan_v2,
)


PROJECT = Path(__file__).resolve().parents[1]


def _hash(label: str) -> str:
    return stable_hash({"contract-runtime-test": label})


def _measurement_plan() -> Any:
    candidate = load_fixed_contract_candidate_v2(PROJECT)
    return build_measurement_plan_v2(
        targets=tuple(
            MeasurementTargetV2(
                item_id=f"financial-contract-{index}",
                fold_id=f"measurement-fold-{index % 4}",
            )
            for index in range(8)
        ),
        manifest_hash=_hash("manifest"),
        evaluator_epoch="contract-runtime-test-v2",
        treatment=FixedPeriodOutTreatmentV2(
            recipe_id=candidate.recipe_id,
            program_set_hash=candidate.program_set_hash,
            period_out_treatment_id=_hash("period-out-treatment"),
            external_skill_source_receipt_hash=(
                candidate.external_skill_source_receipt_hash
            ),
            candidate_skill_source=candidate.candidate_skill_source,
        ),
        agent_id="codex",
        model="offline-test-model",
        max_steps=100,
        codex_agent_execution_policy_hash=_hash("execution-policy"),
    )


def test_fixed_candidate_binds_skill_asset_and_operator() -> None:
    candidate = load_fixed_contract_candidate_v2(PROJECT)
    candidate.verify()
    planner = SharedFinancialSec13FContractPlannerV2(
        asset_path=candidate.operator_asset_path
    )

    assert planner.asset["candidate_id"] == candidate.candidate_id
    assert (
        planner.asset["operator_source_sha256"]
        == candidate.operator_source_sha256
    )
    assert planner.asset["manifest_hash"] == candidate.asset_manifest_hash
    assert candidate.safe_payload(PROJECT)["operator_is_candidate_content"]


def test_evaluation_treatment_rejects_self_consistent_extra_field() -> None:
    candidate = load_fixed_contract_candidate_v2(PROJECT)
    value = build_evaluation_treatment_v2(
        candidate=candidate,
        execution_source_closure_hash=_hash("closure"),
        measurement_view_hash=_hash("view"),
        benchmark_tree_hash=_hash("tree"),
    )
    assert validate_evaluation_treatment_v2(
        value, candidate=candidate
    ) == value["binding_hash"]

    value["prompt_override"] = "forbidden"
    body = dict(value)
    body.pop("binding_hash")
    value["binding_hash"] = stable_hash(body)
    with pytest.raises(ContractTreatmentError, match="fields drifted"):
        validate_evaluation_treatment_v2(value, candidate=candidate)


def test_contract_backend_rejects_program_set_drift_before_execution(
    tmp_path: Path,
) -> None:
    candidate = load_fixed_contract_candidate_v2(PROJECT)
    treatment_hash = _hash("treatment")
    request = SkillLearnTrialRequest(
        item_id="financial-contract-fixture",
        family="financial-analysis",
        split=SplitName.VALIDATION,
        variant=TrialVariant.POLICY_ON,
        evaluator_epoch="contract-runtime-test",
        pair_id="pair-test",
        repeat=0,
        agent_id="codex",
        model="offline-test-model",
        max_steps=100,
        manifest_hash=_hash("manifest"),
        program_id=candidate.recipe_id,
        program_set_hash=candidate.program_set_hash,
        treatment_hash=treatment_hash,
        external_skill_source_receipt_hash=(
            candidate.external_skill_source_receipt_hash
        ),
    )
    backend = object.__new__(DurableFinancialSec13FContractBackendV2)
    backend.durable_request_hash = request.request_hash
    backend.expected_program_id = candidate.recipe_id
    backend.expected_program_set_hash = _hash("wrong-program-set")
    backend.expected_treatment_hash = treatment_hash
    backend.expected_external_skill_source_receipt_hash = (
        candidate.external_skill_source_receipt_hash
    )

    with pytest.raises(
        FinancialSemanticReplicationBackendError,
        match="identity or source drifted",
    ):
        backend.run(
            request,
            skill_source_dir=tmp_path,
            trace_id="must-not-execute",
        )


def test_integration_never_copies_answer_payload_to_host() -> None:
    path = (
        PROJECT
        / "assumption_agent"
        / "benchmarks"
        / "financial_sec13f_contract_integration_v2.py"
    )
    source = path.read_text(encoding="utf-8")
    assert 'f"{container_name}:/root/answers.json"' not in source
    assert '"answers_payload": answers_payload' not in source
    assert '"sha256sum",\n                    "/root/answers.json"' in source
    assert '"python3",\n                    "-B"' in source
    assert '"raw_entity_persisted_in_durable_evidence": False' in source


def test_plan_freeze_cross_binds_source_tree_and_exact_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = load_fixed_contract_candidate_v2(PROJECT)
    view_hash = _hash("fresh-view")
    source_hash = _hash("execution-source")
    tree_hash = _hash("benchmark-tree")
    view = {
        "measurement_view_hash": view_hash,
        "measurement_items": [
            {"item_id": f"financial-contract-{index}", "fold": index % 4}
            for index in range(8)
        ],
    }
    monkeypatch.setattr(runner, "verify_measurement_view", lambda value: value)
    protocol = SimpleNamespace(
        payload={
            "agent_id": "codex",
            "model": "offline-test-model",
            "max_steps": 100,
        },
        codex_agent_execution_policy=SimpleNamespace(
            policy_hash=_hash("execution-policy")
        ),
    )
    evaluation = build_evaluation_treatment_v2(
        candidate=candidate,
        execution_source_closure_hash=source_hash,
        measurement_view_hash=view_hash,
        benchmark_tree_hash=tree_hash,
    )
    expected = build_measurement_plan_v2(
        targets=tuple(
            MeasurementTargetV2(
                item_id=str(item["item_id"]),
                fold_id=f"measurement-fold-{int(item['fold'])}",
            )
            for item in view["measurement_items"]
        ),
        manifest_hash=view_hash,
        evaluator_epoch="contract-runtime-test-v2",
        treatment=FixedPeriodOutTreatmentV2(
            recipe_id=candidate.recipe_id,
            program_set_hash=candidate.program_set_hash,
            period_out_treatment_id=evaluation["period_out_treatment_id"],
            external_skill_source_receipt_hash=(
                candidate.external_skill_source_receipt_hash
            ),
            candidate_skill_source=candidate.candidate_skill_source,
        ),
        agent_id="codex",
        model="offline-test-model",
        max_steps=100,
        codex_agent_execution_policy_hash=_hash("execution-policy"),
    )
    freeze = {
        "execution_source_closure": {"closure_hash": source_hash},
        "materialization": {"benchmark_tree_hash": tree_hash},
        "treatment": {
            "evaluation_binding": evaluation,
            "recipe_id": candidate.recipe_id,
            "program_set_hash": candidate.program_set_hash,
            "external_skill_source_receipt_hash": (
                candidate.external_skill_source_receipt_hash
            ),
            "evaluator_epoch": "contract-runtime-test-v2",
        },
        "plan": {
            "plan_hash": expected.plan_hash,
            "safe_payload": expected.safe_payload(),
        },
    }

    observed = runner.build_plan_from_freeze_v2(
        measurement_view=view,
        execution_freeze=freeze,
        candidate=candidate,
        protocol=protocol,
    )
    assert observed.plan_hash == expected.plan_hash

    cross_bound = copy.deepcopy(freeze)
    cross_bound["treatment"]["evaluation_binding"] = (
        build_evaluation_treatment_v2(
            candidate=candidate,
            execution_source_closure_hash=_hash("other-execution-source"),
            measurement_view_hash=view_hash,
            benchmark_tree_hash=tree_hash,
        )
    )
    with pytest.raises(runner.ContractRunnerError, match="identity drifted"):
        runner.build_plan_from_freeze_v2(
            measurement_view=view,
            execution_freeze=cross_bound,
            candidate=candidate,
            protocol=protocol,
        )

    extra_plan_field = copy.deepcopy(freeze)
    extra_plan_field["plan"]["raw_plan"] = {"entity": "forbidden"}
    with pytest.raises(runner.ContractRunnerError, match="plan differs"):
        runner.build_plan_from_freeze_v2(
            measurement_view=view,
            execution_freeze=extra_plan_field,
            candidate=candidate,
            protocol=protocol,
        )


def test_reused_executor_submits_sixteen_distinct_barrier_workers() -> None:
    plan = _measurement_plan()
    backend_barrier = threading.Barrier(16)
    calls: list[str] = []
    calls_lock = threading.Lock()
    backend_ids: set[int] = set()

    class BarrierBackend:
        def __init__(self, work: Any) -> None:
            self.work = work

        def run(
            self,
            request: Any,
            *,
            skill_source_dir: Path | None,
            trace_id: str,
        ) -> str:
            assert request.request_hash == self.work.request.request_hash
            assert skill_source_dir == self.work.skill_source_dir
            assert trace_id == (
                "financial-semantic-v2:" + self.work.work_unit_hash[:20]
            )
            backend_barrier.wait(timeout=10)
            with calls_lock:
                calls.append(self.work.work_unit_hash)
            return self.work.arm

    def factory(work: Any) -> BarrierBackend:
        backend = BarrierBackend(work)
        backend_ids.add(id(backend))
        return backend

    execution = runner.execute_measurement_plan_v2(
        plan=plan,
        backend_factory=factory,
    )

    assert len(backend_ids) == 16
    assert len(calls) == len(set(calls)) == 16
    assert execution.backend_instance_count == 16
    assert execution.barrier_participant_count == 16
    assert execution.maximum_active_backend_calls == 16
    assert execution.safe_payload()["all_futures_submitted_before_results_read"]


def test_recovery_wrapper_rejects_cross_arm_delegates(tmp_path: Path) -> None:
    plan = _measurement_plan()
    raw_work = next(work for work in plan.work_units if work.arm == "raw")
    candidate_work = next(
        work for work in plan.work_units if work.arm == "candidate"
    )
    raw_delegate = object.__new__(DurableRawSubprocessBackendV2)
    candidate_delegate = object.__new__(
        DurableFinancialSec13FContractBackendV2
    )
    candidate_bindings = {
        "expected_plan_hash": _hash("contract-plan"),
        "expected_program_id": plan.treatment.recipe_id,
        "expected_treatment_hash": plan.treatment.period_out_treatment_id,
        "expected_external_source_receipt_hash": (
            plan.treatment.external_skill_source_receipt_hash
        ),
    }

    with pytest.raises(runner.ContractRunnerError, match="candidate.*RAW"):
        runner.ContractRecoveryBoundBackendV2(
            delegate=raw_delegate,
            work=candidate_work,
            state_root=tmp_path / "candidate-state",
            trial_root=tmp_path / "candidate-trial",
            expected_process_scope="per_process",
            **candidate_bindings,
        )
    with pytest.raises(runner.ContractRunnerError, match="RAW.*candidate"):
        runner.ContractRecoveryBoundBackendV2(
            delegate=candidate_delegate,
            work=raw_work,
            state_root=tmp_path / "raw-state",
            trial_root=tmp_path / "raw-trial",
            expected_process_scope="per_process",
        )

    candidate_wrapper = runner.ContractRecoveryBoundBackendV2(
        delegate=candidate_delegate,
        work=candidate_work,
        state_root=tmp_path / "candidate-state",
        trial_root=tmp_path / "candidate-trial",
        expected_process_scope="per_process",
        **candidate_bindings,
    )
    raw_wrapper = runner.ContractRecoveryBoundBackendV2(
        delegate=raw_delegate,
        work=raw_work,
        state_root=tmp_path / "raw-state",
        trial_root=tmp_path / "raw-trial",
        expected_process_scope="per_process",
    )
    assert candidate_wrapper.work.arm == "candidate"
    assert raw_wrapper.work.arm == "raw"


def test_prewarm_schema_matches_hygienic_runtime_and_item_set() -> None:
    item_ids = [f"financial-contract-{index}" for index in range(8)]
    rows = [{"item_id": item_id} for item_id in item_ids]
    tree_hash = _hash("benchmark-tree")
    body = {
        "prewarm_version": runner.PREWARM_VERSION,
        "measurement_view_hash": _hash("fresh-view"),
        "benchmark_tree_hash": tree_hash,
        "pre_prewarm_tree_hash": tree_hash,
        "post_prewarm_tree_hash": tree_hash,
        "benchmark_tree_unchanged": True,
        "python_dont_write_bytecode": True,
        "python_dont_write_bytecode_env": "1",
        "formal_execution_cache_only": True,
        "formal_image_cache_only": True,
        "formal_offline_verifier_cache_only": True,
        "formal_verifier_network": "none",
        "model_calls": 0,
        "online_judge_calls": 0,
        "sealed_task_count": 0,
        "sealed_content_accessed": False,
        "secret_value_persisted": False,
        "item_count": 8,
        "formal_cache_rows": rows,
        "formal_cache_row_set_hash": payload_hash(rows),
    }
    prewarm = {**body, "prewarm_hash": payload_hash(body)}

    observed = runner._prewarm_by_item_v2(
        prewarm=prewarm,
        measurement_view_hash=_hash("fresh-view"),
        benchmark_tree_hash=tree_hash,
        expected_item_ids=item_ids,
    )
    assert set(observed) == set(item_ids)

    stale = {**prewarm, "prewarm_version": "stale-v1"}
    stale_body = dict(stale)
    stale_body.pop("prewarm_hash")
    stale["prewarm_hash"] = payload_hash(stale_body)
    with pytest.raises(runner.ContractRunnerError, match="policy drifted"):
        runner._prewarm_by_item_v2(
            prewarm=stale,
            measurement_view_hash=_hash("fresh-view"),
            benchmark_tree_hash=tree_hash,
            expected_item_ids=item_ids,
        )


def test_durable_evidence_rejects_nested_raw_plan_or_answers() -> None:
    safe = {
        "plan_hash": _hash("plan"),
        "query_receipt": {
            "answer_key_set_hash": _hash("answer-keys"),
            "answers_payload_persisted_in_receipt": False,
            "raw_entity_persisted_in_receipt": False,
        },
        "answers_payload_persisted": False,
        "raw_instruction_persisted": False,
    }
    runner._assert_no_raw_contract_payload_v2(safe)

    for raw in (
        {"query_receipt": {"answers_payload": {"q1_answer": 1}}},
        {"nested": [{"plan": {"operations": []}}]},
        {"nested": {"instruction": "raw instruction"}},
    ):
        with pytest.raises(
            runner.ContractRunnerError,
            match="raw payload content",
        ):
            runner._assert_no_raw_contract_payload_v2(raw)


def test_benchmark_tree_is_rehashed_and_execution_has_final_check(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    payload = benchmark / "payload.txt"
    payload.write_text("before\n", encoding="utf-8")
    expected = runner.measurement_benchmark_tree_receipt_v2(benchmark)[
        "tree_hash"
    ]
    assert (
        runner._require_benchmark_tree_hash_v2(
            benchmark,
            expected_tree_hash=expected,
            stage="test",
        )
        == expected
    )

    payload.write_text("after\n", encoding="utf-8")
    with pytest.raises(runner.ContractRunnerError, match="tree drifted"):
        runner._require_benchmark_tree_hash_v2(
            benchmark,
            expected_tree_hash=expected,
            stage="test",
        )

    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert 'stage="measurement execution completion"' in source


def test_resume_tree_rejects_nested_symlink(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (output / "worker_state").symlink_to(target, target_is_directory=True)

    with pytest.raises(runner.ContractRunnerError, match="link or special"):
        runner._require_regular_output_tree_v2(output)
