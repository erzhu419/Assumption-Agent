from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

from hegel_machine import phase3_q1_prestart_successor_candidate_v2 as successor
from hegel_machine import phase3_q1_start_campaign_v2 as campaign


def _sha(value: int) -> str:
    return "sha256:" + f"{value:02x}" * 32


def _bundle(
    monkeypatch: pytest.MonkeyPatch,
    source_identity: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    legacy = [
        successor._legacy_gate_dict(row)  # type: ignore[attr-defined]
        for row in successor.LEGACY_GATE_ROWS_1_TO_10
    ]
    monkeypatch.setattr(
        successor,
        "_validate_base_artifact",
        lambda _payload: (deepcopy(legacy), {}),
    )
    monkeypatch.setattr(
        successor,
        "_validate_q1_start_v1_terminal",
        lambda _payload: {},
    )
    manifests: dict[str, object] = {}
    descriptors: dict[str, object] = {}
    mount_rows: list[dict[str, object]] = []
    for index, actor in enumerate(successor.ACTORS, 1):
        entrypoint = successor.ACTOR_ENTRYPOINT_RELATIVE_PATHS[actor]
        rows = [
            {
                "relative_path": entrypoint,
                "byte_count": 100 + index,
                "raw_sha256": _sha(40 + index),
            }
        ]
        manifests[actor] = {
            "source_manifest_root": successor.actor_source_manifest_root_v2(
                actor, rows
            ),
            "rows": rows,
        }
        implementation_id, action_id = successor.ACTOR_IMPLEMENTATION_REGISTRY[
            actor
        ]
        descriptors[actor] = {
            "implementation_id": implementation_id,
            "action_id": action_id,
            "entrypoint_relative_path": entrypoint,
            "entrypoint_raw_sha256": rows[0]["raw_sha256"],
            **deepcopy(successor.ACTOR_BOUND_ENVIRONMENT_ROOTS[actor]),
        }
        mount_rows.append(
            {
                "actor_id": actor,
                "read_only_source_relative_paths": [entrypoint],
                "target_truth_mounted": False,
                "split_mounted": False,
                "role_mounted": False,
            }
        )
    bundle = successor.derive_q1_prestart_successor_candidate_v2(
        b"synthetic base payload admitted by this campaign fixture",
        b"synthetic v1 terminal admitted by this campaign fixture",
        v6_scratch_observations=deepcopy(
            successor.EXPECTED_V6_SCRATCH_OBSERVATIONS
        ),
        corrected_resource_envelope=deepcopy(
            successor.EXPECTED_CORRECTED_RESOURCE_ENVELOPE
        ),
        source_identity=source_identity,
        actor_source_manifests=manifests,
        actor_implementation_descriptors=descriptors,
        mount_manifest={
            "schema_version": successor.MOUNT_MANIFEST_SCHEMA_VERSION,
            "rows": mount_rows,
        },
        input_absence_manifest={
            "schema_version": successor.INPUT_ABSENCE_SCHEMA_VERSION,
            "formal_input_keys": [],
            "target_truth_present": False,
            "split_present": False,
            "role_present": False,
        },
        formal_output_layout_root=successor.FORMAL_OUTPUT_LAYOUT_ROOT,
        run_id=bytes.fromhex("88" * 16),
        created_at_unix_seconds=1_788_100_100,
    )
    return bundle, manifests


def _contract_value(tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "project"
    run = tmp_path / "q1-start-v2-run"
    ledger = tmp_path / "q1-start-v2-ledger"
    v1_run = tmp_path / "q1-start-v1-run"
    v1_ledger = tmp_path / "q1-start-v1-ledger"
    return {
        "schema_version": campaign.CONTRACT_SCHEMA_VERSION,
        "contract_id": campaign.CONTRACT_ID,
        "status": "IMPLEMENTED_NOT_EXECUTED",
        "action_id": campaign.ACTION_ID,
        "capsule_inputs_schema_version": campaign.CAPSULE_INPUTS_SCHEMA_VERSION,
        "fixed_paths": {
            "project_root": str(project),
            "campaign_lock_path": str(tmp_path / "q1-start-v2.lock"),
            "run_root": str(run),
            "ledger_root": str(ledger),
            "capsule_path": str(run / "capsule.json"),
            "authorization_path": str(ledger / "authorization.json"),
            "consumption_path": str(ledger / "consumption.json"),
            "checkpoint_root": str(run / "checkpoints"),
            "outcome_path": str(run / "outcome.json"),
            "terminal_path": str(ledger / "terminal.json"),
            "artifact_path": str(project / "artifacts" / "q1-complete-v2.json"),
            "rust_binary_path": str(run / "hegel-q1-formal-v2"),
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
            "src/hegel_machine/phase3_q1_prestart_successor_candidate_v2.py",
            "src/hegel_machine/phase3_q1_start_campaign_v2.py",
            "tools/phase3_q1_start_supervisor_v2.py",
        ],
        "base_authoritative_predecessor": {
            "prestart_artifact_path": str(tmp_path / "base" / "artifact.json"),
            "prestart_artifact_raw_sha256": successor.BASE_ARTIFACT_RAW_SHA256,
            "prestart_capsule_path": str(tmp_path / "base" / "capsule.json"),
            "prestart_capsule_digest": successor.BASE_CAPSULE_DIGEST,
            "prestart_authorization_path": str(
                tmp_path / "base" / "authorization.json"
            ),
            "prestart_authorization_digest": successor.BASE_AUTHORIZATION_DIGEST,
            "prestart_consumption_path": str(
                tmp_path / "base" / "consumption.json"
            ),
            "prestart_consumption_record_digest": _sha(60),
            "formal_source_commit": successor.BASE_SOURCE_COMMIT,
            "formal_source_tree": successor.BASE_SOURCE_TREE,
            "run_genesis_root": successor.BASE_RUN_GENESIS_ROOT,
            "prestart_admission_receipt_root": successor.BASE_RECEIPT_ROOT,
            "actor_implementation_roots": deepcopy(
                successor.BASE_ACTOR_IMPLEMENTATION_ROOTS
            ),
            "q1_state": "NOT_RUN",
            "q1_gate_count": 14,
            "q1_gate_mask": 0x003FFF,
            "q1_output_slots": [None] * 8,
        },
        "failed_start_v1_predecessor": {
            "run_root": str(v1_run),
            "ledger_root": str(v1_ledger),
            "capsule_digest": successor.Q1_START_V1_CAPSULE_DIGEST,
            "authorization_digest": successor.Q1_START_V1_AUTHORIZATION_DIGEST,
            "consumption_record_digest": (
                successor.Q1_START_V1_CONSUMPTION_RECORD_DIGEST
            ),
            "terminal_path": str(v1_ledger / "campaign.terminal.json"),
            "terminal_raw_sha256": successor.Q1_START_V1_TERMINAL_RAW_SHA256,
            "terminal_record_digest": (
                successor.Q1_START_V1_TERMINAL_RECORD_DIGEST
            ),
            "status": campaign.FAILED_V1_STATUS,
            "failure_code": campaign.FAILED_V1_CODE,
            "failure_detail": campaign.FAILED_V1_DETAIL,
            "actor_start_count_by_actor": {"host": 0, "python": 1, "rust": 0},
            "q1_state": "NOT_RUN",
            "q1_gate_count": 14,
            "q1_gate_mask": 0x003FFF,
            "q1_output_slots": [None] * 8,
            "terminal_no_resume": True,
            "prior_python_action_conservatively_counted": True,
            "work_product_reuse_allowed": False,
        },
        "successor_prestart_bundle_descriptor": {
            "path": str(tmp_path / "successor" / "bundle.json"),
            "schema_version": successor.SCHEMA_VERSION,
            "status": successor.STATUS,
            "canonical_json_required": True,
            "endpoint_scratch_bytes": 134_217_728,
            "authorized": False,
            "authorization_consumption_count": 0,
            "campaign_execution_count": 0,
            "popen_count": 0,
        },
        "runtime": {
            "docker_path": "/usr/bin/docker",
            "supervisor_python_path": "/usr/bin/python3.10",
            "python_image": campaign.PYTHON_IMAGE,
            "rust_image": campaign.RUST_IMAGE,
            "seccomp_profile_path": str(run / "seccomp-profile.json"),
            "seccomp_profile_raw_sha256": _sha(61),
            "python_cpuset": "0-11",
            "rust_cpuset": "12-23",
            "host_cpuset": "0-11",
            "endpoint_memory_bytes": 2_147_483_648,
            "endpoint_output_bytes": 16_777_216,
            "endpoint_scratch_bytes": 134_217_728,
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
            "successor_bundle_replay_before_each_actor_intent": True,
            "failed_start_v1_resume_allowed": False,
            "failed_start_v1_work_product_reuse_allowed": False,
            "network_allowed": False,
        },
        "authority_boundary": {
            "q1_start_authorized_by_future_exact_capsule": True,
            "fresh_v2_actor_actions_require_exact_authorization": True,
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
    contract: campaign.LoadedContractV2
    bundle: dict[str, object]
    capsule: dict[str, object]
    authorization: dict[str, object]
    consumption: dict[str, object]


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CampaignState:
    contract_path = tmp_path / "contract.json"
    contract_path.write_bytes(campaign.canonical_json_bytes_v2(_contract_value(tmp_path)))
    contract = campaign.load_contract_v2(contract_path)
    rows: list[dict[str, object]] = []
    for relative in contract.value["controller_source_closure_relative_paths"]:
        payload = f"source:{relative}\n".encode("ascii")
        rows.append(
            {
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": campaign.raw_sha256_v2(payload),
            }
        )
    controller_source_identity = {
        "source_commit": "99" * 20,
        "source_tree": "aa" * 20,
        "source_closure_root": campaign.source_closure_root_v2(rows),
    }
    bundle, actor_source_manifests = _bundle(
        monkeypatch, controller_source_identity
    )
    rust_payload = b"fresh q1 start v2 Rust materializer\n"
    capsule = campaign.build_capsule_v2(
        contract,
        {
            "schema_version": campaign.CAPSULE_INPUTS_SCHEMA_VERSION,
            "controller_source_identity": controller_source_identity,
            "controller_source_closure_rows": rows,
            "rust_binary": {
                "path": contract.value["fixed_paths"]["rust_binary_path"],
                "byte_count": len(rust_payload),
                "raw_sha256": campaign.raw_sha256_v2(rust_payload),
            },
            "successor_prestart_bundle": bundle,
            "actor_source_manifests": actor_source_manifests,
            "run_id": "bb" * 16,
            "created_at_unix_seconds": 1_788_100_200,
        },
    )
    authorization = campaign.build_authorization_v2(contract, capsule)
    consumption = campaign.build_consumption_v2(contract, capsule, authorization)
    return CampaignState(contract, bundle, capsule, authorization, consumption)


def _outcome(state: CampaignState, status: str) -> dict[str, object]:
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
            "q1_output_slots": [_sha(70 + index) for index in range(8)],
            "completion_receipt_root": _sha(80),
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
        "failure_code": "Q1_START_ACTOR_NONZERO",
        "failure_detail": "python:1",
    }


def test_contract_and_capsule_bind_static_policy_then_full_bundle(
    state: CampaignState,
) -> None:
    policy = state.contract.value["successor_prestart_bundle_descriptor"]
    descriptor = state.capsule["successor_prestart_bundle_descriptor"]

    assert "raw_sha256" not in policy
    assert "bundle_root" not in policy
    assert "source_commit" not in policy
    assert descriptor["raw_sha256"] == campaign.raw_sha256_v2(
        successor.canonical_json_bytes_v2(state.bundle)
    )
    assert descriptor["bundle_root"] == state.bundle["bundle_root"]
    assert descriptor["source_commit"] == state.bundle["source_identity"][  # type: ignore[index]
        "source_commit"
    ]
    assert state.capsule["successor_prestart_bundle"] == state.bundle
    assert state.capsule["actor_source_manifests"] == state.bundle[
        "actor_source_manifests"
    ]
    assert state.capsule["runtime"]["endpoint_scratch_bytes"] == 134_217_728  # type: ignore[index]
    assert campaign.validate_capsule_v2(
        state.capsule, state.contract
    ) == state.capsule


def test_contract_rejects_scratch_drift_and_v1_path_overlap(tmp_path: Path) -> None:
    changed = _contract_value(tmp_path)
    changed["runtime"]["endpoint_scratch_bytes"] = 4_194_304  # type: ignore[index]
    with pytest.raises(
        campaign.Q1StartCampaignV2ValidationError,
        match="endpoint_scratch_bytes differs",
    ):
        campaign.validate_contract_v2(changed)

    changed = _contract_value(tmp_path)
    changed["failed_start_v1_predecessor"]["run_root"] = changed[  # type: ignore[index]
        "fixed_paths"
    ]["run_root"]
    with pytest.raises(
        campaign.Q1StartCampaignV2ValidationError,
        match="overlap the terminal v1",
    ):
        campaign.validate_contract_v2(changed)


def test_capsule_rejects_bundle_or_lineage_tamper(state: CampaignState) -> None:
    changed = deepcopy(state.capsule)
    changed["successor_prestart_bundle"]["bundle_root"] = _sha(90)  # type: ignore[index]
    with pytest.raises(
        campaign.Q1StartCampaignV2ValidationError,
        match="successor-prestart bundle rejected",
    ):
        campaign.validate_capsule_v2(changed, state.contract)

    divergent_source = deepcopy(state.capsule["controller_source_identity"])
    divergent_source["source_commit"] = "98" * 20
    with pytest.raises(
        campaign.Q1StartCampaignV2ValidationError,
        match="bundle source identity differs from controller source identity",
    ):
        campaign.build_capsule_v2(
            state.contract,
            {
                "schema_version": campaign.CAPSULE_INPUTS_SCHEMA_VERSION,
                "controller_source_identity": divergent_source,
                "controller_source_closure_rows": state.capsule[
                    "controller_source_closure_rows"
                ],
                "rust_binary": state.capsule["rust_binary"],
                "successor_prestart_bundle": state.bundle,
                "actor_source_manifests": state.capsule[
                    "actor_source_manifests"
                ],
                "run_id": state.capsule["run_id"],
                "created_at_unix_seconds": state.capsule[
                    "created_at_unix_seconds"
                ],
            },
        )

    changed_contract = deepcopy(state.contract.value)
    changed_contract["base_authoritative_predecessor"][  # type: ignore[index]
        "prestart_artifact_raw_sha256"
    ] = _sha(91)
    path = state.contract.path.parent / "lineage-drift.json"
    path.write_bytes(campaign.canonical_json_bytes_v2(changed_contract))
    loaded = campaign.load_contract_v2(path)
    with pytest.raises(
        campaign.Q1StartCampaignV2ValidationError,
        match="bundle lineage differs",
    ):
        campaign.build_capsule_v2(
            loaded,
            {
                "schema_version": campaign.CAPSULE_INPUTS_SCHEMA_VERSION,
                "controller_source_identity": state.capsule[
                    "controller_source_identity"
                ],
                "controller_source_closure_rows": state.capsule[
                    "controller_source_closure_rows"
                ],
                "rust_binary": state.capsule["rust_binary"],
                "successor_prestart_bundle": state.bundle,
                "actor_source_manifests": state.capsule[
                    "actor_source_manifests"
                ],
                "run_id": state.capsule["run_id"],
                "created_at_unix_seconds": state.capsule[
                    "created_at_unix_seconds"
                ],
            },
        )


def test_authorization_and_consumption_use_v2_once_scope(
    state: CampaignState,
) -> None:
    assert state.authorization["authorized_scope"] == campaign.AUTHORIZED_SCOPE
    assert state.authorization["authorization_consumption_limit"] == 1
    assert state.authorization["fresh_v2_actor_actions_authorized"] is True
    assert state.authorization["failed_start_v1_resume_authorized"] is False
    assert state.authorization["failed_start_v1_work_product_reuse_authorized"] is False
    assert state.consumption["authorization_consumption_count"] == 1
    assert state.consumption["logical_campaign_count"] == 1
    assert state.consumption["actor_start_count_by_actor"] == {
        "host": 0,
        "python": 0,
        "rust": 0,
    }

    changed = deepcopy(state.authorization)
    changed["authorized_scope"] = "TARGET_BLIND_FULL_NODE6_Q1_START_ONCE"
    with pytest.raises(
        campaign.Q1StartCampaignV2ValidationError,
        match="authorization replay differs",
    ):
        campaign.validate_authorization_v2(
            changed, state.contract, state.capsule
        )


def test_pending_boundaries_match_one_shot_actor_prefixes(
    state: CampaignState,
) -> None:
    pre = campaign.build_pending_record_v2(
        state.contract,
        state.capsule,
        status=campaign.PENDING_NOT_CONSUMED,
        phase="LOCAL_PATH_ADMISSION",
        condition="SUCCESSOR_BUNDLE_TEMPORARILY_UNAVAILABLE",
        detail="no authority consumed and no actor intent created",
        authorization_digest=state.authorization["authorization_digest"],
    )
    assert pre["authorization_consumption_count"] == 0
    assert pre["completed_actors"] == []

    for completed in ([], ["python"], ["python", "rust"]):
        pending = campaign.build_pending_record_v2(
            state.contract,
            state.capsule,
            status=campaign.PENDING_CLEAN_CHECKPOINT,
            phase="ACTOR_SEQUENCE",
            condition="NEXT_ACTOR_TEMPORARILY_UNAVAILABLE",
            detail="durable completed prefix; no next intent",
            authorization_digest=state.authorization["authorization_digest"],
            consumption_record_digest=state.consumption["record_digest"],
            completed_actors=completed,
        )
        assert pending["completed_actors"] == completed
        assert campaign.validate_pending_record_v2(
            pending, state.contract, state.capsule
        ) == pending

    with pytest.raises(
        campaign.Q1StartCampaignV2ValidationError,
        match="clean actor prefix",
    ):
        campaign.build_pending_record_v2(
            state.contract,
            state.capsule,
            status=campaign.PENDING_CLEAN_CHECKPOINT,
            phase="ACTOR_SEQUENCE",
            condition="HOST_ALREADY_INTENDED",
            detail="host intent forbids pending",
            authorization_digest=state.authorization["authorization_digest"],
            consumption_record_digest=state.consumption["record_digest"],
            completed_actors=["python", "rust", "host"],
        )


@pytest.mark.parametrize(
    "status",
    [campaign.INCONCLUSIVE, campaign.OUTCOME_UNKNOWN],
)
def test_terminal_no_resume_preserves_14_of_20(
    state: CampaignState,
    status: str,
) -> None:
    terminal = campaign.build_terminal_record_v2(
        state.contract,
        state.capsule,
        state.authorization,
        _outcome(state, status),
        artifact_raw_sha256=None,
    )
    assert terminal["status"] == status
    assert terminal["artifact_raw_sha256"] is None
    assert terminal["outcome"]["q1_gate_count"] == 14  # type: ignore[index]
    assert campaign.validate_terminal_record_v2(
        terminal,
        state.contract,
        state.capsule,
        state.authorization,
    ) == terminal


def test_success_terminal_requires_all_three_once_and_eight_outputs(
    state: CampaignState,
) -> None:
    terminal = campaign.build_terminal_record_v2(
        state.contract,
        state.capsule,
        state.authorization,
        _outcome(state, campaign.COMPLETE),
        artifact_raw_sha256=_sha(95),
    )
    assert terminal["status"] == campaign.COMPLETE
    assert len(terminal["outcome"]["q1_output_slots"]) == 8  # type: ignore[index]

    changed = _outcome(state, campaign.COMPLETE)
    changed["actor_start_count_by_actor"]["rust"] = 0  # type: ignore[index]
    with pytest.raises(
        campaign.Q1StartCampaignV2ValidationError,
        match="strict sequential prefix",
    ):
        campaign.build_terminal_record_v2(
            state.contract,
            state.capsule,
            state.authorization,
            changed,
            artifact_raw_sha256=_sha(95),
        )


def test_v2_domains_are_distinct_and_canonical_parser_rejects_duplicate() -> None:
    assert all(domain.endswith(b"/V2\x00") for domain in campaign.DOMAINS.values())
    payload = b'{"a":1,"a":2}\n'
    with pytest.raises(
        campaign.Q1StartCampaignV2ValidationError,
        match="duplicate JSON key",
    ):
        campaign.strict_json_bytes_v2(payload, "duplicate")
