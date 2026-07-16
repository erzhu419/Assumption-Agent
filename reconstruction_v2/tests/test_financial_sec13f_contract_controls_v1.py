from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from assumption_agent.benchmarks.financial_sec13f_contract_operator_v2 import (
    NUMERIC_ENGINE,
    OPERATOR_VERSION,
    QUERY_RECEIPT_VERSION,
    payload_hash,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import TrialVariant
from assumption_agent.models import stable_hash
from replication_runtime.financial_sec13f_contract_v2.controls import (
    CONTROL_EXECUTION_CLAIM_FILENAME,
    CONTROL_STAGE_ORDER_V1,
    ControlBackendResultV1,
    ControlTargetBindingV1,
    ControlsRuntimeError,
    DurableSkillOnlyBackendV1,
    _SkillOnlyOfflineVerifierProxyV1,
    authorize_control_execution_once_v1,
    build_control_plan_v1,
    execute_control_plan_once_v1,
    initialize_control_state_v1,
    validate_operator_only_query_receipt_v1,
    validate_prior_measurement_reuse_v1,
)
from replication_runtime.financial_semantic_v2.durable_state import (
    load_durable_stage_chain_v2,
)


def _hash(label: str) -> str:
    return stable_hash({"controls-test": label})


def _targets() -> tuple[ControlTargetBindingV1, ...]:
    return tuple(
        ControlTargetBindingV1(
            item_id=f"financial-control-{index}",
            fold_id=f"measurement-fold-{index // 2}",
            prior_pair_id=f"prior-pair-{index}",
            prior_raw_observation_hash=_hash(f"raw-observation-{index}"),
            prior_candidate_observation_hash=_hash(
                f"candidate-observation-{index}"
            ),
            prior_raw_success=False,
            prior_candidate_success=True,
            candidate_output_sha256=_hash(f"candidate-output-{index}"),
            typed_plan_hash=_hash(f"typed-plan-{index}"),
            extraction_receipt_hash=_hash(f"extraction-{index}"),
        )
        for index in range(8)
    )


def _plan() -> Any:
    return build_control_plan_v1(
        targets=_targets(),
        controls_preregistration_hash=_hash("preregistration"),
        prior_measurement_report_hash=_hash("measurement-report"),
        prior_measurement_plan_hash=_hash("measurement-plan"),
        evaluator_epoch="sec13f-controls-test-v1",
        candidate_recipe_id=_hash("recipe"),
        candidate_program_set_hash=stable_hash(
            {"recipe_ids": [_hash("recipe")]}
        ),
        candidate_treatment_hash=_hash("candidate-treatment"),
        skill_only_treatment_hash=_hash("skill-only-treatment"),
        operator_only_treatment_hash=_hash("operator-only-treatment"),
        external_skill_source_receipt_hash=_hash("skill-source"),
        candidate_skill_source=Path("/tmp/candidate-skill"),
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        codex_agent_execution_policy_hash=_hash("execution-policy"),
    )


def test_control_plan_is_eight_skill_only_plus_eight_operator_only() -> None:
    plan = _plan()
    assert len(plan.work_units) == 16
    assert sum(row.arm == "skill_only" for row in plan.work_units) == 8
    assert sum(row.arm == "operator_only" for row in plan.work_units) == 8
    assert len({row.work_unit_hash for row in plan.work_units}) == 16
    assert all(
        row.request is not None
        and row.request.variant is TrialVariant.POLICY_ON
        and row.skill_source_dir == Path("/tmp/candidate-skill")
        for row in plan.work_units
        if row.arm == "skill_only"
    )
    assert all(
        row.request is None
        and row.skill_source_dir is None
        for row in plan.work_units
        if row.arm == "operator_only"
    )
    safe = plan.safe_payload()
    assert safe["prior_observation_reuse_count"] == 16
    assert safe["prior_observation_execution_count"] == 0
    assert safe["physical_control_execution_count"] == 16
    assert safe["skill_only_model_call_count"] == 8
    assert safe["operator_only_model_call_count"] == 0
    assert safe["performance_gate_bound"] is False


def test_control_claim_is_atomic_and_never_authorizes_replay(
    tmp_path: Path,
) -> None:
    work = _plan().work_units[0]
    state = tmp_path / "state"
    initialize_control_state_v1(state_root=state, work=work)
    claim = authorize_control_execution_once_v1(
        state_root=state,
        work=work,
    )
    assert claim["model_call_authorization_count"] == (
        1 if work.arm == "skill_only" else 0
    )
    assert claim["retry_authorized"] is False
    assert claim["replay_authorized"] is False
    assert (state / CONTROL_EXECUTION_CLAIM_FILENAME).is_file()
    with pytest.raises((FileExistsError, ControlsRuntimeError)):
        authorize_control_execution_once_v1(
            state_root=state,
            work=work,
        )


def test_control_executor_submits_all_sixteen_before_reading_results(
    tmp_path: Path,
) -> None:
    plan = _plan()
    backend_barrier = threading.Barrier(16)
    entered: list[str] = []
    lock = threading.Lock()

    class FakeBackend:
        def __init__(self, work: Any) -> None:
            self.work = work

        def run_control(
            self,
            *,
            work: Any,
            state_root: Path,
            trace_id: str,
        ) -> ControlBackendResultV1:
            assert work == self.work
            assert trace_id.endswith(work.work_unit_hash[:20])
            with lock:
                entered.append(work.work_unit_hash)
            backend_barrier.wait(timeout=5)
            return ControlBackendResultV1(
                arm=work.arm,
                work_unit_hash=work.work_unit_hash,
                request_hash=work.request_hash,
                observation_hash=_hash(f"observation-{work.work_unit_hash}"),
                valid=True,
                success=work.arm == "operator_only",
                score=1.0 if work.arm == "operator_only" else 0.0,
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

    execution = execute_control_plan_once_v1(
        plan=plan,
        worker_root=tmp_path / "workers",
        backend_factory=FakeBackend,
    )
    assert len(entered) == 16
    assert execution.backend_instance_count == 16
    assert execution.barrier_participant_count == 16
    assert execution.all_futures_submitted_before_results_read
    assert execution.model_call_count == 8
    assert execution.operator_call_count == 8
    assert execution.retry_count == 0
    for work in plan.work_units:
        chain = load_durable_stage_chain_v2(
            tmp_path / "workers" / work.work_unit_hash / "durable",
            stage_order=CONTROL_STAGE_ORDER_V1,
            work_unit_hash=work.work_unit_hash,
            request_hash=work.request_hash,
        )
        assert [row.stage for row in chain[:3]] == [
            "planned",
            "typed_plan_ready",
            "execution_claimed",
        ]


def test_prior_measurement_reuse_accepts_only_complete_safe_report() -> None:
    pairs = [
        {
            "item_id_hash": target.item_id_hash,
            "fold_id": target.fold_id,
            "pair_id": target.prior_pair_id,
            "raw_observation_hash": target.prior_raw_observation_hash,
            "candidate_observation_hash": (
                target.prior_candidate_observation_hash
            ),
            "raw_valid": True,
            "candidate_valid": True,
            "raw_success": target.prior_raw_success,
            "candidate_success": target.prior_candidate_success,
            "delta": 1,
            "relation": "gain",
        }
        for target in _targets()
    ]
    body = {
        "runner_version": "financial_sec13f_contract_fresh_runner_v2",
        "execution_completed": True,
        "evidence_valid": True,
        "plan_hash": _hash("prior-plan"),
        "physical_model_call_count": 16,
        "raw_model_call_count": 8,
        "candidate_model_call_count": 8,
        "model_replay_count": 0,
        "retry_count": 0,
        "resampling_used": False,
        "sealed_content_accessed": False,
        "answers_payload_persisted": False,
        "raw_plan_persisted": False,
        "results": {
            "pairs": pairs,
            "pair_set_hash": stable_hash(pairs),
            "invalid_pair_count": 0,
            "raw_successes": 0,
            "candidate_successes": 8,
        },
    }
    report = {**body, "report_hash": stable_hash(body)}
    receipt = validate_prior_measurement_reuse_v1(
        report,
        targets=_targets(),
        expected_report_hash=report["report_hash"],
    )
    assert receipt["reused_observation_count"] == 16
    assert receipt["executions_performed"] == 0
    assert receipt["model_calls_performed"] == 0

    unsafe = copy.deepcopy(report)
    unsafe["answers_payload"] = {"q1_answer": 1}
    unsafe_body = dict(unsafe)
    unsafe_body.pop("report_hash")
    unsafe["report_hash"] = stable_hash(unsafe_body)
    with pytest.raises(ControlsRuntimeError, match="raw payload"):
        validate_prior_measurement_reuse_v1(
            unsafe,
            targets=_targets(),
            expected_report_hash=unsafe["report_hash"],
        )


def test_operator_only_query_receipt_rejects_answer_content() -> None:
    input_rows = [
        {
            "role": role,
            "table": table,
            "size_bytes": index + 1,
            "file_sha256": _hash(f"{role}-{table}"),
        }
        for index, (role, table) in enumerate(
            (
                ("previous", "COVERPAGE.tsv"),
                ("previous", "INFOTABLE.tsv"),
                ("current", "COVERPAGE.tsv"),
                ("current", "INFOTABLE.tsv"),
            )
        )
    ]
    body = {
        "receipt_version": QUERY_RECEIPT_VERSION,
        "operator_version": OPERATOR_VERSION,
        "candidate_id": _hash("candidate"),
        "asset_manifest_hash": _hash("asset"),
        "contract_hash": _hash("contract"),
        "operator_source_sha256": _hash("operator-source"),
        "plan_hash": _hash("typed-plan"),
        "numeric_engine": NUMERIC_ENGINE,
        "input_file_receipts": input_rows,
        "input_set_hash": payload_hash(input_rows),
        "pre_output_exists": False,
        "pre_output_sha256": None,
        "post_output_sha256": _hash("output"),
        "output_changed": True,
        "answer_key_set_hash": _hash("answer-keys"),
        "answers_payload_persisted_in_receipt": False,
        "raw_entity_persisted_in_receipt": False,
        "network_calls": 0,
        "model_calls": 0,
        "verifier_content_accessed": False,
        "gold_content_accessed": False,
        "pack_content_accessed": False,
    }
    receipt = {**body, "receipt_hash": payload_hash(body)}
    assert validate_operator_only_query_receipt_v1(
        receipt,
        expected_plan_hash=_hash("typed-plan"),
        expected_candidate_id=_hash("candidate"),
        expected_asset_manifest_hash=_hash("asset"),
        expected_contract_hash=_hash("contract"),
        expected_operator_source_sha256=_hash("operator-source"),
    )["post_output_sha256"] == _hash("output")

    unsafe = copy.deepcopy(receipt)
    unsafe["answers"] = {"q1_answer": 1}
    unsafe_body = dict(unsafe)
    unsafe_body.pop("receipt_hash")
    unsafe["receipt_hash"] = payload_hash(unsafe_body)
    with pytest.raises(ControlsRuntimeError, match="raw payload"):
        validate_operator_only_query_receipt_v1(
            unsafe,
            expected_plan_hash=_hash("typed-plan"),
            expected_candidate_id=_hash("candidate"),
            expected_asset_manifest_hash=_hash("asset"),
            expected_contract_hash=_hash("contract"),
            expected_operator_source_sha256=_hash("operator-source"),
        )


def test_skill_only_backend_rejects_non_policy_on_before_model(
    tmp_path: Path,
) -> None:
    work = next(row for row in _plan().work_units if row.arm == "skill_only")
    assert work.request is not None
    wrong_request = copy.copy(work.request)
    object.__setattr__(wrong_request, "variant", TrialVariant.POLICY_OFF)
    backend = object.__new__(DurableSkillOnlyBackendV1)
    backend.control_work = work
    backend.expected_program_id = work.request.program_id
    backend.expected_program_set_hash = work.request.program_set_hash
    backend.expected_treatment_hash = work.request.treatment_hash
    backend.expected_external_skill_source_receipt_hash = (
        work.request.external_skill_source_receipt_hash
    )
    with pytest.raises(ControlsRuntimeError, match="identity"):
        backend.run(
            wrong_request,
            skill_source_dir=tmp_path,
            trace_id="must-not-execute",
        )


def test_skill_only_verifier_disconnects_network_before_offline_tests(
    tmp_path: Path,
) -> None:
    order: list[tuple[str, ...]] = []

    class Host:
        def run(self, command: list[str], **_: Any) -> Any:
            row = tuple(command)
            order.append(row)
            stdout = "{}\n" if row[:2] == ("docker", "inspect") else ""
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    class Isolation:
        def __init__(self) -> None:
            self.delegate = Host()

        def run(self, command: list[str], *_: Any, **__: Any) -> Any:
            order.append(("OFFLINE_VERIFIER", *tuple(command)))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    checkpointed: list[bool] = []
    backend = SimpleNamespace(
        durable_state_root=tmp_path,
        _verifier_network_before_receipt_hash=None,
        _verifier_network_after_receipt_hash=None,
        _checkpoint_raw_before_verifier_v2=lambda: checkpointed.append(True),
    )
    proxy = _SkillOnlyOfflineVerifierProxyV1(
        Isolation(),
        backend=backend,
        network_name="frozen-provider-network",
    )
    result = proxy.run(
        ["docker", "exec", "trial-container", "bash", "/tests/test.sh"]
    )
    assert result.returncode == 0 and checkpointed == [True]
    disconnect = next(
        index
        for index, row in enumerate(order)
        if row[:3] == ("docker", "network", "disconnect")
    )
    verifier = next(
        index for index, row in enumerate(order) if row[0] == "OFFLINE_VERIFIER"
    )
    assert disconnect < verifier
    assert len(
        [row for row in order if row[:2] == ("docker", "inspect")]
    ) == 2
    assert backend._verifier_network_before_receipt_hash
    assert backend._verifier_network_after_receipt_hash
    for name in (
        "skill_only.verifier_network.before.json",
        "skill_only.verifier_network.after.json",
    ):
        payload = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert payload["verifier_network"] == "none"
        assert payload["attached_network_count"] == 0
