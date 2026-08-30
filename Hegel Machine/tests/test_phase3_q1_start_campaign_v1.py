from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

from hegel_machine import phase3_q1_start_campaign_v1 as campaign


def _sha(value: int) -> str:
    return "sha256:" + f"{value:02x}" * 32


def _contract_value(tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "project"
    run = tmp_path / "run"
    ledger = tmp_path / "ledger"
    return {
        "schema_version": campaign.CONTRACT_SCHEMA_VERSION,
        "contract_id": campaign.CONTRACT_ID,
        "status": "IMPLEMENTED_NOT_EXECUTED",
        "action_id": campaign.ACTION_ID,
        "capsule_inputs_schema_version": campaign.CAPSULE_INPUTS_SCHEMA_VERSION,
        "fixed_paths": {
            "project_root": str(project),
            "campaign_lock_path": str(tmp_path / "q1-start.lock"),
            "run_root": str(run),
            "ledger_root": str(ledger),
            "capsule_path": str(run / "capsule.json"),
            "authorization_path": str(ledger / "authorization.json"),
            "consumption_path": str(ledger / "consumption.json"),
            "checkpoint_root": str(run / "checkpoints"),
            "outcome_path": str(run / "outcome.json"),
            "terminal_path": str(ledger / "terminal.json"),
            "artifact_path": str(project / "artifacts" / "q1-complete.json"),
            "rust_binary_path": str(run / "hegel-q1-formal-archive-materializer"),
            "admission_path": str(run / "admission.json"),
            "run_spec_path": str(run / "run-spec.json"),
            "python_source_root": str(run / "python-source"),
            "rust_source_root": str(run / "rust-source"),
            "host_source_root": str(run / "host-source"),
            "python_output_path": str(run / "python-output"),
            "python_scratch_path": str(run / "python-scratch"),
            "rust_output_path": str(run / "rust-output"),
            "rust_scratch_path": str(run / "rust-scratch"),
            "host_output_path": str(run / "host-output"),
            "host_scratch_path": str(run / "host-scratch"),
        },
        "controller_source_closure_relative_paths": [
            "src/hegel_machine/phase3_q1_start_campaign_v1.py",
            "src/hegel_machine/phase3_q1_start_completion_wire_v1.py",
            "tools/phase3_q1_start_v1.py",
        ],
        "predecessor": {
            "prestart_artifact_path": str(tmp_path / "prestart" / "artifact.json"),
            "prestart_artifact_raw_sha256": _sha(1),
            "prestart_capsule_path": str(tmp_path / "prestart" / "capsule.json"),
            "prestart_capsule_digest": _sha(2),
            "prestart_authorization_path": str(
                tmp_path / "prestart" / "authorization.json"
            ),
            "prestart_authorization_digest": _sha(3),
            "prestart_consumption_path": str(
                tmp_path / "prestart" / "consumption.json"
            ),
            "prestart_consumption_record_digest": _sha(4),
            "formal_source_commit": "11" * 20,
            "formal_source_tree": "22" * 20,
            "run_genesis_root": _sha(5),
            "prestart_admission_receipt_root": _sha(6),
            "actor_implementation_roots": {
                "host": _sha(7),
                "python": _sha(8),
                "rust": _sha(9),
            },
            "q1_state": "NOT_RUN",
            "q1_gate_count": 14,
            "q1_gate_mask": 0x003FFF,
            "q1_output_slots": [None] * 8,
        },
        "runtime": {
            "docker_path": "/usr/bin/docker",
            "supervisor_python_path": "/usr/bin/python3.10",
            "python_image": campaign.PYTHON_IMAGE,
            "rust_image": campaign.RUST_IMAGE,
            "seccomp_profile_path": str(run / "seccomp-profile.json"),
            "seccomp_profile_raw_sha256": _sha(10),
            "python_cpuset": "0-11",
            "rust_cpuset": "12-23",
            "host_cpuset": "0-11",
            "endpoint_memory_bytes": 2_147_483_648,
            "endpoint_output_bytes": 16_777_216,
            "endpoint_scratch_bytes": 4_194_304,
            "endpoint_wall_seconds": 3600,
            "host_memory_bytes": 67_108_864,
            "host_output_bytes": 2_097_152,
            "host_scratch_bytes": 1,
            "host_wall_seconds": 3600,
            "pids_limit": 128,
            "minimum_available_memory_bytes": 4_294_967_296,
            "minimum_available_storage_bytes": 134_217_728,
        },
        "execution_policy": {
            "actor_order": ["python", "rust", "host"],
            "maximum_start_count_by_actor": {"host": 1, "python": 1, "rust": 1},
            "maximum_concurrently_live_actor_containers": 1,
            "clean_checkpoint_resume_allowed": True,
            "checkpoint_resume_repeats_completed_actor": False,
            "intent_or_partial_without_exact_completion_terminal_no_resume": True,
            "preconsumption_pending_consumes_authorization": False,
            "preconsumption_pending_starts_actor": False,
            "network_allowed": False,
        },
        "authority_boundary": {
            "q1_start_authorized_by_future_exact_capsule": True,
            "q1_state_before": "NOT_RUN",
            "q1_gate_count_before": 14,
            "q1_gate_mask_before": 0x003FFF,
            "q1_success_state": "COMPLETE",
            "q1_success_gate_count": 20,
            "q1_success_gate_mask": 0x0FFFFF,
            "q2_state": "NOT_RUN",
            "m3_formal_roots": None,
            "role_evaluation_performed": False,
            "outside_certificate_issued": False,
            "adoption_allowed": False,
            "promotion_allowed": False,
            "current_pointer_update_allowed": False,
            "active_transition_allowed": False,
        },
    }


@dataclass(frozen=True)
class CampaignState:
    contract_path: Path
    contract: campaign.LoadedContractV1
    capsule: dict[str, object]
    authorization: dict[str, object]
    consumption: dict[str, object]


def _state(tmp_path: Path) -> CampaignState:
    contract_path = tmp_path / "contract.json"
    contract_path.write_bytes(
        campaign.canonical_json_bytes_v1(_contract_value(tmp_path))
    )
    loaded = campaign.load_contract_v1(contract_path)
    source_rows = []
    for relative in loaded.value["controller_source_closure_relative_paths"]:
        payload = ("source:" + relative + "\n").encode("ascii")
        source_rows.append(
            {
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": campaign.raw_sha256_v1(payload),
            }
        )
    rust_payload = b"synthetic formal Rust binary\n"
    inputs = {
        "schema_version": campaign.CAPSULE_INPUTS_SCHEMA_VERSION,
        "controller_source_identity": {
            "source_commit": "33" * 20,
            "source_tree": "44" * 20,
            "source_closure_root": campaign.source_closure_root_v1(source_rows),
        },
        "controller_source_closure_rows": source_rows,
        "rust_binary": {
            "path": loaded.value["fixed_paths"]["rust_binary_path"],
            "byte_count": len(rust_payload),
            "raw_sha256": campaign.raw_sha256_v1(rust_payload),
        },
        "run_id": "55" * 16,
        "created_at_unix_seconds": 1_788_100_000,
    }
    capsule = campaign.build_capsule_v1(loaded, inputs)
    authorization = campaign.build_authorization_v1(loaded, capsule)
    consumption = campaign.build_consumption_v1(
        loaded, capsule, authorization
    )
    return CampaignState(
        contract_path,
        loaded,
        capsule,
        authorization,
        consumption,
    )


def _outcome(
    state: CampaignState,
    status: str,
) -> dict[str, object]:
    if status == campaign.COMPLETE:
        return {
            "status": status,
            "capsule_digest": state.capsule["capsule_digest"],
            "actor_start_count_by_actor": {"host": 1, "python": 1, "rust": 1},
            "network_accessed": False,
            "target_truth_accessed": False,
            "split_accessed": False,
            "role_evaluation_performed": False,
            "q1_state": "COMPLETE",
            "q1_gate_count": 20,
            "q1_gate_mask": 0x0FFFFF,
            "q1_output_slots": [_sha(20 + index) for index in range(8)],
            "completion_receipt_root": _sha(40),
        }
    return {
        "status": status,
        "capsule_digest": state.capsule["capsule_digest"],
        "actor_start_count_by_actor": {"host": 0, "python": 1, "rust": 0},
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 0x003FFF,
        "q1_output_slots": [None] * 8,
        "failure_code": (
            "FORMAL_RESOURCE_GUARD_HIT"
            if status == campaign.INCONCLUSIVE
            else "MEASUREMENT_OUTCOME_UNKNOWN"
        ),
    }


def test_canonical_contract_load_round_trip(tmp_path: Path) -> None:
    state = _state(tmp_path)
    raw = state.contract_path.read_bytes()

    assert campaign.strict_json_bytes_v1(raw, "contract") == state.contract.value
    assert state.contract.raw == raw
    assert state.contract.raw_sha256 == campaign.raw_sha256_v1(raw)
    assert state.contract.value["action_id"] == campaign.ACTION_ID


def test_contract_rejects_noncanonical_and_bool_integer_alias(
    tmp_path: Path,
) -> None:
    value = _contract_value(tmp_path)
    pretty = __import__("json").dumps(value, indent=2).encode("ascii")
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="canonical JSON object",
    ):
        campaign.strict_json_bytes_v1(pretty, "contract")

    changed = deepcopy(value)
    changed["runtime"]["pids_limit"] = True  # type: ignore[index]
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="runtime pids_limit differs",
    ):
        campaign.validate_contract_v1(changed)


def test_contract_rejects_path_and_policy_drift(tmp_path: Path) -> None:
    changed = _contract_value(tmp_path)
    changed["fixed_paths"]["terminal_path"] = str(  # type: ignore[index]
        tmp_path / "wrong-ledger" / "terminal.json"
    )
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="terminal path must belong",
    ):
        campaign.validate_contract_v1(changed)

    changed = _contract_value(tmp_path)
    changed["execution_policy"]["actor_order"] = [  # type: ignore[index]
        "rust",
        "python",
        "host",
    ]
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="execution policy differs",
    ):
        campaign.validate_contract_v1(changed)


def test_capsule_build_validate_and_canonical_round_trip(tmp_path: Path) -> None:
    state = _state(tmp_path)
    encoded = campaign.canonical_json_bytes_v1(state.capsule)

    assert campaign.validate_capsule_v1(
        campaign.strict_json_bytes_v1(encoded, "capsule"),
        state.contract,
    ) == state.capsule
    assert state.capsule["status"] == campaign.CAPSULE_READY
    assert state.capsule["q1_state_before"] == "NOT_RUN"
    assert state.capsule["q1_gate_count_before"] == 14
    assert state.capsule["intended_success_gate_count"] == 20


def test_capsule_rejects_source_binary_and_field_mutations(tmp_path: Path) -> None:
    state = _state(tmp_path)

    changed = deepcopy(state.capsule)
    changed["controller_source_closure_rows"][0]["raw_sha256"] = _sha(90)  # type: ignore[index]
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="source closure root differs",
    ):
        campaign.validate_capsule_v1(changed, state.contract)

    changed = deepcopy(state.capsule)
    changed["rust_binary"]["path"] = str(tmp_path / "other-binary")  # type: ignore[index]
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="Rust binary path or byte count differs",
    ):
        campaign.validate_capsule_v1(changed, state.contract)

    changed = deepcopy(state.capsule)
    changed["created_at_unix_seconds"] = True
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="created_at_unix_seconds",
    ):
        campaign.validate_capsule_v1(changed, state.contract)

    changed = deepcopy(state.capsule)
    changed["unexpected"] = "not admitted"
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="capsule replay differs",
    ):
        campaign.validate_capsule_v1(changed, state.contract)


def test_authorization_build_validate_and_mutation_rejection(tmp_path: Path) -> None:
    state = _state(tmp_path)

    assert campaign.validate_authorization_v1(
        state.authorization,
        state.contract,
        state.capsule,
    ) == state.authorization
    assert state.authorization["status"] == campaign.AUTHORIZATION_RECORDED
    assert state.authorization["authorization_consumption_limit"] == 1
    assert state.authorization["q1_start_authorized"] is True
    assert state.authorization["network_allowed"] is False

    changed = deepcopy(state.authorization)
    changed["network_allowed"] = True
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="authorization replay differs",
    ):
        campaign.validate_authorization_v1(
            changed,
            state.contract,
            state.capsule,
        )


def test_consumption_build_validate_and_mutation_rejection(tmp_path: Path) -> None:
    state = _state(tmp_path)

    assert campaign.validate_consumption_v1(
        state.consumption,
        state.contract,
        state.capsule,
        state.authorization,
    ) == state.consumption
    assert state.consumption["status"] == campaign.AUTHORIZATION_CONSUMED
    assert state.consumption["authorization_consumption_count"] == 1
    assert state.consumption["logical_campaign_count"] == 1
    assert state.consumption["actor_start_count_by_actor"] == {
        "host": 0,
        "python": 0,
        "rust": 0,
    }

    changed = deepcopy(state.consumption)
    changed["actor_start_count_by_actor"]["python"] = 1  # type: ignore[index]
    unsigned = deepcopy(changed)
    unsigned.pop("record_digest")
    changed["record_digest"] = campaign._digest(  # type: ignore[attr-defined]
        campaign.DOMAINS["consumption"],
        unsigned,
    )
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="consumption replay differs",
    ):
        campaign.validate_consumption_v1(
            changed,
            state.contract,
            state.capsule,
            state.authorization,
        )


def test_success_terminal_build_validate_round_trip(tmp_path: Path) -> None:
    state = _state(tmp_path)
    outcome = _outcome(state, campaign.COMPLETE)
    terminal = campaign.build_terminal_record_v1(
        state.contract,
        state.capsule,
        state.authorization,
        outcome,
        artifact_raw_sha256=_sha(70),
    )

    assert campaign.validate_terminal_record_v1(
        terminal,
        state.contract,
        state.capsule,
        state.authorization,
    ) == terminal
    assert terminal["status"] == campaign.COMPLETE
    assert terminal["artifact_raw_sha256"] == _sha(70)
    assert terminal["consumption_record_digest"] == state.consumption["record_digest"]
    assert terminal["authorization_consumption_count"] == 1
    assert terminal["outcome"] == outcome


@pytest.mark.parametrize(
    "status",
    [campaign.INCONCLUSIVE, campaign.OUTCOME_UNKNOWN],
)
def test_failure_terminal_shapes_bind_no_artifact(
    tmp_path: Path,
    status: str,
) -> None:
    state = _state(tmp_path)
    outcome = _outcome(state, status)
    terminal = campaign.build_terminal_record_v1(
        state.contract,
        state.capsule,
        state.authorization,
        outcome,
        artifact_raw_sha256=None,
    )

    assert campaign.validate_terminal_record_v1(
        terminal,
        state.contract,
        state.capsule,
        state.authorization,
    ) == terminal
    assert terminal["status"] == status
    assert terminal["artifact_raw_sha256"] is None
    assert terminal["outcome"]["q1_state"] == "NOT_RUN"  # type: ignore[index]
    assert terminal["outcome"]["q1_output_slots"] == [None] * 8  # type: ignore[index]


@pytest.mark.parametrize(
    "status",
    [campaign.PENDING_NOT_CONSUMED, campaign.PENDING_CLEAN_CHECKPOINT],
)
def test_pending_statuses_are_not_terminal_records(
    tmp_path: Path,
    status: str,
) -> None:
    state = _state(tmp_path)
    outcome = _outcome(state, campaign.INCONCLUSIVE)
    outcome["status"] = status

    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="terminal outcome status differs",
    ):
        campaign.build_terminal_record_v1(
            state.contract,
            state.capsule,
            state.authorization,
            outcome,
            artifact_raw_sha256=None,
        )


@pytest.mark.parametrize("after_authorization", [False, True])
def test_preconsumption_pending_build_validate_round_trip(
    tmp_path: Path,
    after_authorization: bool,
) -> None:
    state = _state(tmp_path)
    authorization_digest = (
        state.authorization["authorization_digest"]
        if after_authorization
        else None
    )
    pending = campaign.build_pending_record_v1(
        state.contract,
        state.capsule,
        status=campaign.PENDING_NOT_CONSUMED,
        phase="PATH_AND_STORAGE_ADMISSION",
        condition="PUBLICATION_DIRECTORY_UNAVAILABLE",
        detail="ledger root is not yet available",
        authorization_digest=authorization_digest,
    )

    assert campaign.validate_pending_record_v1(
        pending,
        state.contract,
        state.capsule,
    ) == pending
    assert pending["authorization_consumption_count"] == 0
    assert pending["logical_campaign_count"] == 0
    assert pending["completed_actors"] == []
    assert pending["actor_start_count_by_actor"] == {
        "host": 0,
        "python": 0,
        "rust": 0,
    }
    assert pending["q1_state"] == "NOT_RUN"
    assert pending["q1_gate_count"] == 14
    assert pending["q1_output_slots"] == [None] * 8
    assert pending["target_truth_accessed"] is False
    assert pending["split_accessed"] is False
    assert pending["role_evaluation_performed"] is False


@pytest.mark.parametrize(
    ("completed_actors", "expected_starts"),
    [
        ([], {"host": 0, "python": 0, "rust": 0}),
        (["python"], {"host": 0, "python": 1, "rust": 0}),
        (["python", "rust"], {"host": 0, "python": 1, "rust": 1}),
    ],
)
def test_clean_checkpoint_pending_build_validate_round_trip(
    tmp_path: Path,
    completed_actors: list[str],
    expected_starts: dict[str, int],
) -> None:
    state = _state(tmp_path)
    pending = campaign.build_pending_record_v1(
        state.contract,
        state.capsule,
        status=campaign.PENDING_CLEAN_CHECKPOINT,
        phase="ACTOR_SEQUENCE",
        condition="SUCCESSOR_CAPACITY_TEMPORARILY_UNAVAILABLE",
        detail="completed actor evidence is durable; next actor has not started",
        authorization_digest=state.authorization["authorization_digest"],
        consumption_record_digest=state.consumption["record_digest"],
        completed_actors=completed_actors,
    )

    assert campaign.validate_pending_record_v1(
        pending,
        state.contract,
        state.capsule,
    ) == pending
    assert pending["authorization_consumption_count"] == 1
    assert pending["logical_campaign_count"] == 1
    assert pending["completed_actors"] == completed_actors
    assert pending["actor_start_count_by_actor"] == expected_starts
    assert pending["network_accessed"] is False
    assert pending["target_truth_accessed"] is False
    assert pending["split_accessed"] is False
    assert pending["role_evaluation_performed"] is False


def test_pending_rejects_completed_work_drift_and_record_tamper(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)

    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="pre-consumption PENDING cannot bind completed work",
    ):
        campaign.build_pending_record_v1(
            state.contract,
            state.capsule,
            status=campaign.PENDING_NOT_CONSUMED,
            phase="PATH_ADMISSION",
            condition="FILE_UNAVAILABLE",
            detail="one local input is absent",
            completed_actors=["python"],
        )

    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="clean actor prefix",
    ):
        campaign.build_pending_record_v1(
            state.contract,
            state.capsule,
            status=campaign.PENDING_CLEAN_CHECKPOINT,
            phase="ACTOR_SEQUENCE",
            condition="NEXT_ACTOR_UNAVAILABLE",
            detail="invalid non-prefix checkpoint",
            authorization_digest=state.authorization["authorization_digest"],
            consumption_record_digest=state.consumption["record_digest"],
            completed_actors=["rust"],
        )

    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="pending authorization digest differs",
    ):
        campaign.build_pending_record_v1(
            state.contract,
            state.capsule,
            status=campaign.PENDING_NOT_CONSUMED,
            phase="PATH_ADMISSION",
            condition="FILE_UNAVAILABLE",
            detail="authorization binding belongs to another capsule",
            authorization_digest=_sha(90),
        )

    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="clean-checkpoint ledger chain differs",
    ):
        campaign.build_pending_record_v1(
            state.contract,
            state.capsule,
            status=campaign.PENDING_CLEAN_CHECKPOINT,
            phase="ACTOR_SEQUENCE",
            condition="NEXT_ACTOR_UNAVAILABLE",
            detail="consumption binding belongs to another campaign",
            authorization_digest=state.authorization["authorization_digest"],
            consumption_record_digest=_sha(91),
            completed_actors=["python"],
        )

    pending = campaign.build_pending_record_v1(
        state.contract,
        state.capsule,
        status=campaign.PENDING_NOT_CONSUMED,
        phase="PATH_ADMISSION",
        condition="FILE_UNAVAILABLE",
        detail="one local input is absent",
    )
    changed = deepcopy(pending)
    changed["network_accessed"] = True
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="pending record replay differs",
    ):
        campaign.validate_pending_record_v1(
            changed,
            state.contract,
            state.capsule,
        )


def test_terminal_rejects_realistic_authority_mutations(tmp_path: Path) -> None:
    state = _state(tmp_path)

    changed = _outcome(state, campaign.COMPLETE)
    changed["q1_output_slots"][3] = None  # type: ignore[index]
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="successful terminal Q1 authority differs",
    ):
        campaign.build_terminal_record_v1(
            state.contract,
            state.capsule,
            state.authorization,
            changed,
            artifact_raw_sha256=_sha(70),
        )

    changed = _outcome(state, campaign.COMPLETE)
    changed["actor_start_count_by_actor"]["rust"] = 2  # type: ignore[index]
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="terminal actor start count differs",
    ):
        campaign.build_terminal_record_v1(
            state.contract,
            state.capsule,
            state.authorization,
            changed,
            artifact_raw_sha256=_sha(70),
        )

    changed = _outcome(state, campaign.COMPLETE)
    changed["actor_start_count_by_actor"]["diagnostic"] = 1  # type: ignore[index]
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="terminal actor start registry differs",
    ):
        campaign.build_terminal_record_v1(
            state.contract,
            state.capsule,
            state.authorization,
            changed,
            artifact_raw_sha256=_sha(70),
        )

    changed = _outcome(state, campaign.INCONCLUSIVE)
    changed["network_accessed"] = True
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="terminal network boundary differs",
    ):
        campaign.build_terminal_record_v1(
            state.contract,
            state.capsule,
            state.authorization,
            changed,
            artifact_raw_sha256=None,
        )

    changed = _outcome(state, campaign.INCONCLUSIVE)
    changed["actor_start_count_by_actor"] = {
        "host": 0,
        "python": 0,
        "rust": 1,
    }
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="strict sequential prefix",
    ):
        campaign.build_terminal_record_v1(
            state.contract,
            state.capsule,
            state.authorization,
            changed,
            artifact_raw_sha256=None,
        )

    changed = _outcome(state, campaign.INCONCLUSIVE)
    changed["target_truth_accessed"] = True
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="terminal target-blind authority differs",
    ):
        campaign.build_terminal_record_v1(
            state.contract,
            state.capsule,
            state.authorization,
            changed,
            artifact_raw_sha256=None,
        )

    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="non-success terminal cannot bind",
    ):
        campaign.build_terminal_record_v1(
            state.contract,
            state.capsule,
            state.authorization,
            _outcome(state, campaign.OUTCOME_UNKNOWN),
            artifact_raw_sha256=_sha(70),
        )

    changed = _outcome(state, campaign.INCONCLUSIVE)
    changed["q1_state"] = "COMPLETE"
    changed["q1_gate_count"] = 20
    changed["q1_gate_mask"] = 0x0FFFFF
    changed["q1_output_slots"] = [_sha(80 + index) for index in range(8)]
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="non-success terminal cannot publish Q1 authority",
    ):
        campaign.build_terminal_record_v1(
            state.contract,
            state.capsule,
            state.authorization,
            changed,
            artifact_raw_sha256=None,
        )


def test_terminal_record_replay_rejects_digest_and_registry_tamper(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    terminal = campaign.build_terminal_record_v1(
        state.contract,
        state.capsule,
        state.authorization,
        _outcome(state, campaign.COMPLETE),
        artifact_raw_sha256=_sha(70),
    )

    changed = deepcopy(terminal)
    changed["record_digest"] = _sha(99)
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="terminal record replay differs",
    ):
        campaign.validate_terminal_record_v1(
            changed,
            state.contract,
            state.capsule,
            state.authorization,
        )

    changed = deepcopy(terminal)
    changed["consumption_record_digest"] = _sha(98)
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="terminal record replay differs",
    ):
        campaign.validate_terminal_record_v1(
            changed,
            state.contract,
            state.capsule,
            state.authorization,
        )

    changed = deepcopy(terminal)
    changed["retry_allowed"] = True
    with pytest.raises(
        campaign.Q1StartCampaignValidationError,
        match="terminal record registry differs",
    ):
        campaign.validate_terminal_record_v1(
            changed,
            state.contract,
            state.capsule,
            state.authorization,
        )
