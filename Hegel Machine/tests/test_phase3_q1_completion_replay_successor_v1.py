from __future__ import annotations

import ast
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from hegel_machine import phase3_q1_completion_replay_successor_v1 as wire


def _sha(label: str) -> str:
    return "sha256:" + sha256(label.encode("ascii")).hexdigest()


def _predecessor(tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "project"
    run = tmp_path / "q1-start-v7r1-run"
    ledger = tmp_path / "q1-start-v7r1-ledger"
    checkpoints = run / "checkpoints"
    return {
        "run_id": "1" * 32,
        "run_root": str(run),
        "ledger_root": str(ledger),
        "contract_path": str(project / "config" / "phase3_q1_start_v7.json"),
        "contract_raw_sha256": _sha("v7r1-contract"),
        "capsule_path": str(run / "capsule.json"),
        "capsule_raw_sha256": _sha("v7r1-capsule-raw"),
        "capsule_digest": _sha("v7r1-capsule"),
        "authorization_path": str(ledger / "owner.authorization.json"),
        "authorization_raw_sha256": _sha("v7r1-authorization-raw"),
        "authorization_digest": _sha("v7r1-authorization"),
        "consumption_path": str(ledger / "campaign.consumed.json"),
        "consumption_raw_sha256": _sha("v7r1-consumption-raw"),
        "consumption_record_digest": _sha("v7r1-consumption"),
        "run_spec_path": str(run / "run-spec.json"),
        "run_spec_raw_sha256": _sha("v7r1-run-spec"),
        "python_intent_path": str(checkpoints / "01-python-intent.json"),
        "python_intent_raw_sha256": _sha("v7r1-python-intent"),
        "python_complete_path": str(checkpoints / "01-python-complete.json"),
        "python_complete_raw_sha256": _sha("v7r1-python-complete"),
        "rust_intent_path": str(checkpoints / "02-rust-intent.json"),
        "rust_intent_raw_sha256": _sha("v7r1-rust-intent"),
        "rust_complete_path": str(checkpoints / "02-rust-complete.json"),
        "rust_complete_raw_sha256": _sha("v7r1-rust-complete"),
        "host_intent_path": str(checkpoints / "03-host-intent.json"),
        "host_intent_raw_sha256": _sha("v7r1-host-intent"),
        "host_complete_path": str(checkpoints / "03-host-complete.json"),
        "host_complete_raw_sha256": _sha("v7r1-host-complete"),
        "supervisor_outcome_path": str(checkpoints / "supervisor-outcome.json"),
        "supervisor_outcome_raw_sha256": _sha("v7r1-supervisor-outcome"),
        "controller_outcome_path": str(run / "campaign.outcome.json"),
        "controller_outcome_raw_sha256": _sha("v7r1-controller-outcome"),
        "host_result_path": str(run / "host-output" / "host-result.json"),
        "host_result_raw_sha256": _sha("v7r1-host-result"),
        "terminal_path": str(ledger / "campaign.terminal.json"),
        "terminal_raw_sha256": _sha("v7r1-terminal-raw"),
        "terminal_record_digest": _sha("v7r1-terminal"),
        "status": wire.V7R1_TERMINAL_STATUS,
        "failure_code": wire.V7R1_FAILURE_CODE,
        "failure_detail": wire.V7R1_FAILURE_DETAIL,
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "actor_start_count_by_actor": {"host": 1, "python": 1, "rust": 1},
        "completed_stages": ["python", "rust", "host"],
        "q1_state": "NOT_RUN",
        "q1_gate_count": wire.Q1_PRESTART_GATE_COUNT,
        "q1_gate_mask": wire.Q1_PRESTART_GATE_MASK,
        "q1_output_slots": deepcopy(wire.NULL_OUTPUT_SLOTS),
        "terminal_no_resume": True,
    }


def _contract_value(tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "project"
    run = tmp_path / "q1-completion-successor-v8c-run"
    ledger = tmp_path / "q1-completion-successor-v8c-ledger"
    predecessor = _predecessor(tmp_path)
    prior_run = Path(predecessor["run_root"])  # type: ignore[arg-type]
    return {
        "schema_version": wire.CONTRACT_SCHEMA,
        "contract_id": wire.CONTRACT_ID,
        "status": "IMPLEMENTED_NOT_EXECUTED",
        "action_id": wire.ACTION_ID,
        "fixed_paths": {
            "project_root": str(project),
            "campaign_lock_path": str(tmp_path / "q1-completion-successor-v8c.lock"),
            "run_root": str(run),
            "ledger_root": str(ledger),
            "capsule_path": str(run / "capsule.json"),
            "authorization_path": str(ledger / "owner.authorization.json"),
            "consumption_path": str(ledger / "campaign.consumed.json"),
            "pending_path": str(run / "campaign.pending.json"),
            "outcome_path": str(run / "campaign.outcome.json"),
            "terminal_path": str(ledger / "campaign.terminal.json"),
            "artifact_path": str(
                project / "artifacts" / "phase3_q1_completion_successor_v8c.cbor"
            ),
        },
        "predecessor_v7r1": predecessor,
        "read_only_replay": {
            "python_output_root": str(prior_run / "python-output"),
            "rust_output_root": str(prior_run / "rust-output"),
            "host_output_root": str(prior_run / "host-output"),
            "access_mode": "READ_ONLY_REPLAY",
            "mutation_allowed": False,
            "reuse_as_actor_work_product_allowed": False,
        },
        "execution_policy": deepcopy(wire.EXECUTION_POLICY),
        "authority_boundary": deepcopy(wire.AUTHORITY_BOUNDARY),
    }


def _loaded_contract(tmp_path: Path) -> wire.LoadedContract:
    path = tmp_path / "completion-v8c-contract.json"
    path.write_bytes(wire.canonical_json_bytes(_contract_value(tmp_path)))
    return wire.load_contract(path)


def _capsule_inputs() -> dict[str, object]:
    rows = [
        {
            "relative_path": "src/hegel_machine/phase3_q1_completion_replay_successor_v1.py",
            "byte_count": 123,
            "raw_sha256": _sha("core-source"),
        },
        {
            "relative_path": "tools/phase3_q1_completion_successor_v8c.py",
            "byte_count": 456,
            "raw_sha256": _sha("controller-source"),
        },
    ]
    slots = [_sha(f"slot-{index}") for index in range(1, 9)]
    return {
        "schema_version": wire.CAPSULE_INPUTS_SCHEMA,
        "source_closure_rows": rows,
        "source_closure_root": wire.source_closure_root(rows),
        "run_id": "2" * 32,
        "created_at_unix_seconds": 1_788_170_000,
        "expected_completion_artifact_raw_sha256": _sha("completion-artifact"),
        "expected_completion_receipt_root": slots[-1],
        "expected_q1_output_slots": slots,
    }


def _state(tmp_path: Path) -> tuple[
    wire.LoadedContract,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    contract = _loaded_contract(tmp_path)
    capsule = wire.build_capsule(contract, _capsule_inputs())
    authorization = wire.build_authorization(contract, capsule)
    consumption = wire.build_consumption(contract, capsule, authorization)
    return contract, capsule, authorization, consumption


def test_complete_state_round_trips_with_zero_execution(tmp_path: Path) -> None:
    contract, capsule, authorization, consumption = _state(tmp_path)
    assert wire.validate_capsule(capsule, contract) == capsule
    assert wire.validate_authorization(authorization, contract, capsule) == authorization
    assert (
        wire.validate_consumption(consumption, contract, capsule, authorization)
        == consumption
    )
    assert consumption["predecessor_actor_start_count_by_actor"] == {
        "host": 1,
        "python": 1,
        "rust": 1,
    }
    assert consumption["completion_successor_actor_start_count_by_actor"] == {
        "host": 0,
        "python": 0,
        "rust": 0,
    }
    terminal = wire.build_terminal(
        contract,
        capsule,
        authorization,
        consumption,
        success=True,
        q1_output_slots=capsule["expected_q1_output_slots"],  # type: ignore[arg-type]
        artifact_raw_sha256=capsule[  # type: ignore[arg-type]
            "expected_completion_artifact_raw_sha256"
        ],
        read_only_replay_completed=True,
    )
    assert (
        wire.validate_terminal(
            terminal, contract, capsule, authorization, consumption
        )
        == terminal
    )
    assert terminal["status"] == wire.COMPLETE
    outcome = terminal["outcome"]
    assert outcome["q1_state"] == "COMPLETE"  # type: ignore[index]
    assert outcome["q1_gate_count"] == 20  # type: ignore[index]
    assert outcome["q1_gate_mask"] == 1_048_575  # type: ignore[index]
    assert len(outcome["q1_output_slots"]) == 8  # type: ignore[index]
    assert all(outcome["q1_output_slots"])  # type: ignore[index]
    assert terminal["status"] == "Q1_COMPLETE_20_OF_20"
    assert outcome["completion_receipt_root"] == capsule[  # type: ignore[index]
        "expected_completion_receipt_root"
    ]
    assert outcome["q2_state"] == "NOT_RUN"  # type: ignore[index]
    assert outcome["target_truth_accessed"] is False  # type: ignore[index]
    assert outcome["split_accessed"] is False  # type: ignore[index]
    assert outcome["role_evaluation_performed"] is False  # type: ignore[index]
    assert outcome["m3_formal_roots"] is None  # type: ignore[index]
    assert outcome["outside_certificate_issued"] is False  # type: ignore[index]
    assert outcome["popen_count"] == 0  # type: ignore[index]
    assert outcome["actor_start_count"] == 0  # type: ignore[index]
    assert outcome["docker_invocation_count"] == 0  # type: ignore[index]
    assert outcome["predecessor_actor_start_count_by_actor"] == {  # type: ignore[index]
        "host": 1,
        "python": 1,
        "rust": 1,
    }
    assert outcome["completion_successor_actor_start_count_by_actor"] == {  # type: ignore[index]
        "host": 0,
        "python": 0,
        "rust": 0,
    }


def test_preconsumption_pending_is_same_digest_and_unconsumed(tmp_path: Path) -> None:
    contract, capsule, authorization, _ = _state(tmp_path)
    for bound_authorization in (None, authorization):
        pending = wire.build_pending(
            contract,
            capsule,
            phase="PRECONSUMPTION_REPLAY",
            condition="LOCAL_INPUT_UNAVAILABLE",
            detail="one bound read-only input is temporarily unavailable",
            authorization=bound_authorization,
        )
        assert (
            wire.validate_pending(
                pending, contract, capsule, authorization=bound_authorization
            )
            == pending
        )
        assert pending["resume_capsule_digest"] == capsule["capsule_digest"]
        assert pending["authorization_consumption_count"] == 0
        assert pending["logical_campaign_count"] == 0
        assert pending["consumption_record_digest"] is None
        assert pending["predecessor_actor_start_count_by_actor"] == {
            "host": 1,
            "python": 1,
            "rust": 1,
        }
        assert pending["completion_successor_actor_start_count_by_actor"] == {
            "host": 0,
            "python": 0,
            "rust": 0,
        }


def test_postconsumption_failure_is_terminal_no_resume(tmp_path: Path) -> None:
    contract, capsule, authorization, consumption = _state(tmp_path)
    terminal = wire.build_terminal(
        contract,
        capsule,
        authorization,
        consumption,
        success=False,
        failure_code="Q1_COMPLETION_REPLAY_VALIDATION_FAILED",
        failure_detail="exact replay output differs",
        read_only_replay_completed=False,
    )
    assert (
        wire.validate_terminal(
            terminal, contract, capsule, authorization, consumption
        )
        == terminal
    )
    assert terminal["status"] == wire.TERMINAL_NO_RESUME
    assert terminal["artifact_raw_sha256"] is None
    outcome = terminal["outcome"]
    assert outcome["terminal_no_resume"] is True  # type: ignore[index]
    assert outcome["q1_state"] == "NOT_RUN"  # type: ignore[index]
    assert outcome["q1_gate_count"] == 14  # type: ignore[index]
    assert outcome["q1_output_slots"] == [None] * 8  # type: ignore[index]
    assert outcome["q2_state"] == "NOT_RUN"  # type: ignore[index]
    assert outcome["completion_receipt_root"] is None  # type: ignore[index]
    assert outcome["target_truth_accessed"] is False  # type: ignore[index]
    assert outcome["split_accessed"] is False  # type: ignore[index]
    assert outcome["role_evaluation_performed"] is False  # type: ignore[index]
    assert outcome["m3_formal_roots"] is None  # type: ignore[index]
    assert outcome["outside_certificate_issued"] is False  # type: ignore[index]


def test_capsule_inputs_bind_source_artifact_receipt_and_slots(tmp_path: Path) -> None:
    contract = _loaded_contract(tmp_path)
    inputs = _capsule_inputs()
    capsule = wire.build_capsule(contract, inputs)
    authorization = wire.build_authorization(contract, capsule)
    assert authorization["source_closure_root"] == inputs["source_closure_root"]
    assert (
        authorization["expected_completion_artifact_raw_sha256"]
        == inputs["expected_completion_artifact_raw_sha256"]
    )
    assert (
        authorization["expected_completion_receipt_root"]
        == inputs["expected_completion_receipt_root"]
    )
    assert authorization["expected_q1_output_slots"] == inputs["expected_q1_output_slots"]
    changed = deepcopy(inputs)
    changed["source_closure_root"] = _sha("wrong-root")
    with pytest.raises(wire.Q1CompletionReplayValidationError, match="source closure root"):
        wire.build_capsule(contract, changed)
    changed = deepcopy(inputs)
    changed["expected_completion_receipt_root"] = _sha("wrong-receipt")
    with pytest.raises(wire.Q1CompletionReplayValidationError, match="slot 8"):
        wire.build_capsule(contract, changed)


def test_success_cannot_publish_unbound_artifact_or_slots(tmp_path: Path) -> None:
    contract, capsule, authorization, consumption = _state(tmp_path)
    with pytest.raises(
        wire.Q1CompletionReplayValidationError,
        match="artifact differs",
    ):
        wire.build_terminal(
            contract,
            capsule,
            authorization,
            consumption,
            success=True,
            q1_output_slots=capsule["expected_q1_output_slots"],  # type: ignore[arg-type]
            artifact_raw_sha256=_sha("wrong-artifact"),
            read_only_replay_completed=True,
        )
    changed_slots = list(capsule["expected_q1_output_slots"])  # type: ignore[arg-type]
    changed_slots[0] = _sha("wrong-slot")
    with pytest.raises(wire.Q1CompletionReplayValidationError, match="slots differ"):
        wire.build_terminal(
            contract,
            capsule,
            authorization,
            consumption,
            success=True,
            q1_output_slots=changed_slots,
            artifact_raw_sha256=capsule[  # type: ignore[arg-type]
                "expected_completion_artifact_raw_sha256"
            ],
            read_only_replay_completed=True,
        )


def test_contract_forbids_v7r1_mutation_execution_and_path_overlap(
    tmp_path: Path,
) -> None:
    value = _contract_value(tmp_path)
    changed = deepcopy(value)
    changed["read_only_replay"]["mutation_allowed"] = True  # type: ignore[index]
    with pytest.raises(wire.Q1CompletionReplayValidationError, match="read-only replay"):
        wire.validate_contract(changed)
    changed = deepcopy(value)
    changed["execution_policy"]["maximum_popen_count"] = 1  # type: ignore[index]
    with pytest.raises(wire.Q1CompletionReplayValidationError, match="execution policy"):
        wire.validate_contract(changed)
    changed = deepcopy(value)
    changed["fixed_paths"]["run_root"] = changed["predecessor_v7r1"][  # type: ignore[index]
        "run_root"
    ]
    with pytest.raises(wire.Q1CompletionReplayValidationError):
        wire.validate_contract(changed)


def test_v7r1_complete_lineage_is_in_capsule_and_authorization(tmp_path: Path) -> None:
    contract, capsule, authorization, _ = _state(tmp_path)
    predecessor = capsule["predecessor_v7r1"]
    for stage in ("python", "rust", "host"):
        assert predecessor[f"{stage}_intent_raw_sha256"]  # type: ignore[index]
        assert predecessor[f"{stage}_complete_raw_sha256"]  # type: ignore[index]
    assert predecessor["contract_raw_sha256"]  # type: ignore[index]
    assert predecessor["run_spec_raw_sha256"]  # type: ignore[index]
    assert predecessor["supervisor_outcome_raw_sha256"]  # type: ignore[index]
    assert predecessor["host_result_raw_sha256"]  # type: ignore[index]
    assert predecessor["terminal_record_digest"]  # type: ignore[index]
    assert authorization["v7r1_contract_raw_sha256"] == predecessor[  # type: ignore[index]
        "contract_raw_sha256"
    ]
    assert authorization["v7r1_host_intent_raw_sha256"] == predecessor[  # type: ignore[index]
        "host_intent_raw_sha256"
    ]
    assert (
        authorization["v7r1_terminal_record_digest"]
        == predecessor["terminal_record_digest"]  # type: ignore[index]
    )
    changed = deepcopy(capsule)
    changed["predecessor_v7r1"]["host_result_raw_sha256"] = _sha("tampered")  # type: ignore[index]
    with pytest.raises(wire.Q1CompletionReplayValidationError, match="capsule replay"):
        wire.validate_capsule(changed, contract)


def test_core_has_no_process_or_container_execution_surface() -> None:
    source_path = Path(wire.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "subprocess" not in imported
    assert "docker" not in imported
    assert wire.EXECUTION_POLICY["maximum_popen_count"] == 0
    assert wire.EXECUTION_POLICY["maximum_actor_start_count"] == 0
    assert wire.EXECUTION_POLICY["maximum_docker_invocation_count"] == 0
    assert wire.EXECUTION_POLICY["exact_partial_finalization_recovery_allowed"] is True
    assert wire.EXECUTION_POLICY["postconsumption_pending_allowed"] is False
    assert wire.AUTHORITY_BOUNDARY["m3_formal_roots"] is None
    assert wire.AUTHORITY_BOUNDARY["outside_certificate_issued"] is False
