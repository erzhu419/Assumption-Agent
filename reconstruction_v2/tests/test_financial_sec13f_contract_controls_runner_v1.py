from __future__ import annotations

import json
from pathlib import Path

from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from assumption_agent.models import stable_hash
from replication_runtime.financial_sec13f_contract_v2.controls import (
    ControlBackendResultV1,
    ControlExecutionV1,
)
from replication_runtime.financial_sec13f_contract_v2.controls_runner import (
    _failure_disposition_v1,
    build_control_plan_from_replication_c_v1,
    descriptive_controls_results_v1,
)
from replication_runtime.financial_sec13f_contract_v2.treatment import (
    load_fixed_contract_candidate_v2,
)


PROJECT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> dict[str, object]:
    value = json.loads((PROJECT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _prepared():
    protocol = PaperProtocol.read(
        PROJECT / "manifests/skilllearn_paper_protocol_v3_20_ruoli_gpt54mini.json"
    )
    return build_control_plan_from_replication_c_v1(
        preregistration=_read(
            "manifests/financial_sec13f_contract_v2_controls_preregistration_v1.json"
        ),
        replication_c_execution_freeze=_read(
            "manifests/financial_sec13f_contract_v2_replication_c_execution_freeze_v1.json"
        ),
        replication_c_report=_read(
            "artifacts/financial_sec13f_contract_v2_replication_c_measurement_run_v1/measurement.report.json"
        ),
        measurement_view=_read(
            "manifests/financial_sec13f_contract_v2_replication_c_measurement_view_v1.json"
        ),
        candidate=load_fixed_contract_candidate_v2(PROJECT),
        protocol=protocol,
    )


def test_rebuilds_exact_eight_target_sixteen_work_control_plan() -> None:
    prepared = _prepared()
    plan = prepared.plan
    assert len(prepared.targets) == 8
    assert len(plan.work_units) == 16
    assert sum(work.arm == "skill_only" for work in plan.work_units) == 8
    assert sum(work.arm == "operator_only" for work in plan.work_units) == 8
    assert len({work.target.item_id_hash for work in plan.work_units}) == 8
    assert prepared.reuse_receipt["reused_observation_count"] == 16
    assert prepared.reuse_receipt["executions_performed"] == 0
    assert prepared.reuse_receipt["model_calls_performed"] == 0
    safe_text = json.dumps(plan.safe_payload(), sort_keys=True)
    assert '"instruction"' not in safe_text
    assert '"query"' not in safe_text
    assert '"answers"' not in safe_text
    assert '"gold"' not in safe_text
    assert plan.skill_only_treatment_hash != plan.candidate_treatment_hash
    assert plan.operator_only_treatment_hash != plan.candidate_treatment_hash
    assert plan.skill_only_treatment_hash != plan.operator_only_treatment_hash


def test_four_arm_report_is_descriptive_hash_only_and_fold_complete() -> None:
    prepared = _prepared()
    results: list[ControlBackendResultV1] = []
    for work in prepared.plan.work_units:
        success = work.arm == "operator_only"
        results.append(
            ControlBackendResultV1(
                arm=work.arm,
                work_unit_hash=work.work_unit_hash,
                request_hash=work.request_hash,
                observation_hash=stable_hash(
                    {"mock-control-observation": work.work_unit_hash}
                ),
                valid=True,
                success=success,
                score=1.0 if success else 0.0,
                model_calls=1 if work.arm == "skill_only" else 0,
                operator_calls=1 if work.arm == "operator_only" else 0,
                online_judge_calls=0,
                output_sha256=(
                    work.target.candidate_output_sha256
                    if work.arm == "operator_only"
                    else None
                ),
                candidate_output_hash_match=(
                    True if work.arm == "operator_only" else None
                ),
            )
        )
    execution = ControlExecutionV1(
        plan=prepared.plan,
        results=tuple(sorted(results, key=lambda row: row.work_unit_hash)),
        backend_instance_count=16,
        barrier_participant_count=16,
        maximum_active_backend_calls=16,
    )
    report = descriptive_controls_results_v1(
        plan=prepared.plan,
        reuse_receipt=prepared.reuse_receipt,
        execution=execution,
    )
    assert report["record_count"] == 32
    assert [row["arm"] for row in report["arms"]] == [
        "raw",
        "full",
        "skill_only",
        "operator_only",
    ]
    assert [row["contrast"] for row in report["contrasts"]] == [
        "full_minus_raw",
        "skill_only_minus_raw",
        "full_minus_skill_only",
        "operator_only_minus_raw",
        "full_minus_operator_only",
    ]
    assert len(report["folds"]) == 4
    assert all(row["item_count"] == 2 for row in report["folds"])
    assert len(report["item_relations"]) == 8
    assert report["performance_thresholds"] == []
    assert report["performance_gate_bound"] is False
    text = json.dumps(report, sort_keys=True)
    assert '"instruction"' not in text
    assert '"query"' not in text
    assert '"answers"' not in text
    assert '"gold"' not in text


def test_invalid_incremental_result_is_not_relabelled_failure_or_success() -> None:
    prepared = _prepared()
    invalid_work = next(
        work for work in prepared.plan.work_units if work.arm == "skill_only"
    )
    results: list[ControlBackendResultV1] = []
    for work in prepared.plan.work_units:
        invalid = work.work_unit_hash == invalid_work.work_unit_hash
        success = work.arm == "operator_only"
        results.append(
            ControlBackendResultV1(
                arm=work.arm,
                work_unit_hash=work.work_unit_hash,
                request_hash=work.request_hash,
                observation_hash=stable_hash(
                    {"mock-invalid-control-observation": work.work_unit_hash}
                ),
                valid=not invalid,
                success=success,
                score=1.0 if success else 0.0,
                model_calls=1 if work.arm == "skill_only" else 0,
                operator_calls=1 if work.arm == "operator_only" else 0,
                online_judge_calls=0,
                output_sha256=(
                    work.target.candidate_output_sha256
                    if work.arm == "operator_only"
                    else None
                ),
                candidate_output_hash_match=(
                    True if work.arm == "operator_only" else None
                ),
            )
        )
    execution = ControlExecutionV1(
        plan=prepared.plan,
        results=tuple(sorted(results, key=lambda row: row.work_unit_hash)),
        backend_instance_count=16,
        barrier_participant_count=16,
        maximum_active_backend_calls=16,
    )
    report = descriptive_controls_results_v1(
        plan=prepared.plan,
        reuse_receipt=prepared.reuse_receipt,
        execution=execution,
    )
    skill = next(row for row in report["arms"] if row["arm"] == "skill_only")
    invalid_row = next(
        row
        for row in skill["rows"]
        if row["item_id_hash"] == invalid_work.target.item_id_hash
    )
    assert invalid_row["valid"] is False
    assert invalid_row["success"] is None
    assert invalid_row["score"] is None
    contrast = next(
        row
        for row in report["contrasts"]
        if row["contrast"] == "skill_only_minus_raw"
    )
    relation = next(
        row
        for row in contrast["rows"]
        if row["item_id_hash"] == invalid_work.target.item_id_hash
    )
    assert relation["paired_valid"] is False
    assert relation["delta"] is None
    assert relation["relation"] == "invalid"


def test_failure_disposition_matches_preregistered_execution_boundary() -> None:
    assert _failure_disposition_v1(True) == "executed_incomplete_no_retry"
    assert _failure_disposition_v1(False) == "execution_failed_no_retry"
