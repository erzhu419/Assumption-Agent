from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hegel_machine import phase3_q2_admission_v1 as admission
from hegel_machine import phase3_q2_campaign_v1 as campaign
from hegel_machine import phase3_q2_role_membership_v1 as membership
from hegel_machine.phase3_q1_formal_archive_contract_v1 import Q1_OUTPUT_SLOT_NAMES


PREREGISTRATION_PATH = ROOT / "config/phase3_q2_role_membership_v1.json"


def _tag(value: int) -> str:
    return "sha256:" + bytes((value,)).hex() * 32


def _bytes(value: str) -> bytes:
    return bytes.fromhex(value[7:])


def _contract(tmp_path: Path) -> campaign.LoadedContract:
    prereg_raw = PREREGISTRATION_PATH.read_bytes()
    root = tmp_path.resolve()
    value = {
        "schema_version": campaign.CONTRACT_SCHEMA,
        "contract_id": campaign.CONTRACT_ID,
        "status": "IMPLEMENTED_NOT_EXECUTED",
        "action_id": campaign.ACTION_ID,
        "preregistration_path": str(PREREGISTRATION_PATH.resolve()),
        "preregistration_raw_sha256": campaign.raw_sha256(prereg_raw),
        "fixed_paths": {
            "project_root": str(root),
            "campaign_lock_path": str(root / "campaign.lock"),
            "run_root": str(root / "run"),
            "ledger_root": str(root / "ledger"),
            "capsule_path": str(root / "run/capsule.json"),
            "authorization_path": str(root / "ledger/authorization.json"),
            "consumption_path": str(root / "ledger/consumption.json"),
            "pending_path": str(root / "run/pending.json"),
            "terminal_path": str(root / "ledger/terminal.json"),
            "artifact_path": str(root / "artifact.cbor"),
            "trajectory_path": str(root / "run/trajectory.jsonl"),
        },
        "q1_inputs": {
            "terminal_path": str(root / "q1-terminal.json"),
            "completion_artifact_path": str(root / "q1-completion.cbor"),
            "archive_root": str(root / "q1-python-output"),
            "selected_endpoint_kind": "python",
            "access_mode": "READ_ONLY_PREDECESSOR",
        },
        "resource_envelope": admission.q2_preregistration_object_v1()[
            "resource_envelope"
        ],
        "execution_policy": campaign.EXECUTION_POLICY,
        "authority_boundary": campaign.AUTHORITY_BOUNDARY,
    }
    path = root / "contract.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return campaign.load_contract(path)


def _preflight_receipt() -> dict[str, object]:
    slots = admission.Q1_V8C_BINDING.ordered_q1_output_slots
    roles = admission.Q2_ROLE_COMMITMENTS
    return {
        "endpoint_kind": "python",
        "odd_class_count": 418,
        "sink_class_count": 89,
        "odd_universe_root": roles[0].universe_root,
        "sink_universe_root": roles[1].universe_root,
        "odd_signature_manifest_root": slots[0],
        "sink_signature_manifest_root": slots[2],
        "odd_saturation_state_root": slots[1],
        "sink_saturation_state_root": slots[3],
        "odd_class_archive_root": _tag(40),
        "sink_class_archive_root": _tag(41),
        "odd_class_stream_framed_bytes": 852_581,
        "sink_class_stream_framed_bytes": 55_084,
        "total_class_stream_framed_bytes": 907_665,
        "selected_reader_total_bytes": 919_757,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }


def _campaign_objects(
    tmp_path: Path,
) -> tuple[
    campaign.LoadedContract,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    admission.Q1CompletionAdmissionV1,
]:
    contract = _contract(tmp_path)
    preregistration = admission.q2_preregistration_object_v1()
    rows = [
        {
            "relative_path": "src/example.py",
            "byte_count": 1,
            "raw_sha256": _tag(1),
        }
    ]
    inputs = {
        "schema_version": campaign.CAPSULE_INPUTS_SCHEMA,
        "source_closure_rows": rows,
        "source_closure_root": campaign.source_closure_root(rows),
        "run_id": "12" * 16,
        "created_at_unix_seconds": 1,
        "preflight_receipt": _preflight_receipt(),
    }
    capsule = campaign.build_capsule(contract, preregistration, inputs)
    authorization = campaign.build_authorization(
        contract, preregistration, capsule
    )
    consumption = campaign.build_consumption(
        contract, preregistration, capsule, authorization
    )
    slots = tuple(
        (index, name, _bytes(root))
        for index, (name, root) in enumerate(
            zip(
                Q1_OUTPUT_SLOT_NAMES,
                admission.Q1_V8C_BINDING.ordered_q1_output_slots,
                strict=True,
            ),
            1,
        )
    )
    admitted = admission.Q1CompletionAdmissionV1(
        admission.Q1_V8C_BINDING, slots
    )
    return (
        contract,
        preregistration,
        capsule,
        authorization,
        consumption,
        admitted,
    )


def _complete_artifact(
    capsule: dict[str, object],
    authorization: dict[str, object],
    consumption: dict[str, object],
    admitted: admission.Q1CompletionAdmissionV1,
) -> membership.Q2MembershipArtifactV1:
    odd_commitment, sink_commitment = admission.Q2_ROLE_COMMITMENTS
    odd = membership.Q2RoleMembershipResultV1(
        odd_commitment,
        admitted.ordered_output_slots[0][2],
        admitted.ordered_output_slots[1][2],
        _bytes(_tag(40)),
        418,
        (),
        membership.OUTSIDE_CLOSURE,
    )
    sink = membership.Q2RoleMembershipResultV1(
        sink_commitment,
        admitted.ordered_output_slots[2][2],
        admitted.ordered_output_slots[3][2],
        _bytes(_tag(41)),
        89,
        (_bytes(_tag(42)),),
        membership.IN_CLOSURE,
    )
    return membership.build_q2_membership_artifact_v1(
        _bytes(capsule["capsule_digest"]),
        _bytes(authorization["authorization_digest"]),
        _bytes(consumption["record_digest"]),
        admitted,
        odd,
        sink,
        observed_wall_time_ns=1,
        observed_peak_rss_bytes=1,
    )


def test_complete_terminal_requires_exact_campaign_and_all_q1_roots(
    tmp_path: Path,
) -> None:
    contract, prereg, capsule, authorization, consumption, admitted = (
        _campaign_objects(tmp_path)
    )
    artifact = _complete_artifact(
        capsule, authorization, consumption, admitted
    )
    terminal = campaign.build_complete_terminal(
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        artifact,
    )
    assert terminal["status"] == membership.PASS_STATUS
    assert terminal["outcome"]["q2_state"] == "COMPLETE"
    assert campaign.validate_terminal(
        terminal,
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        artifact_payload=artifact.canonical_bytes,
    ) == terminal
    with pytest.raises(campaign.Q2CampaignValidationError, match="binding"):
        campaign.build_complete_terminal(
            contract,
            prereg,
            capsule,
            authorization,
            consumption,
            replace(artifact, q1_closure_bundle_root=bytes(32)),
        )


def test_invalidated_terminal_persists_conservative_action_denominators(
    tmp_path: Path,
) -> None:
    contract, prereg, capsule, authorization, consumption, _ = (
        _campaign_objects(tmp_path)
    )
    terminal = campaign.build_invalidated_terminal(
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        failure_phase="OUTSIDE_TARGET",
        failure_code="Q2_POSTCONSUMPTION_OUTCOME_UNKNOWN",
        failure_detail="interrupted after target-open intent",
        target_open_count_by_role={
            "OUTSIDE_TARGET": 1,
            "IN_LANGUAGE_NULL": 0,
        },
        role_evaluation_count_by_role={
            "OUTSIDE_TARGET": 0,
            "IN_LANGUAGE_NULL": 0,
        },
    )
    assert terminal["outcome"]["target_truth_accessed"] is True
    assert terminal["outcome"]["role_evaluation_performed"] is False
    assert campaign.validate_terminal(
        terminal,
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
    ) == terminal
    with pytest.raises(campaign.Q2CampaignValidationError, match="forbids"):
        campaign.validate_terminal(
            terminal,
            contract,
            prereg,
            capsule,
            authorization,
            consumption,
            artifact_payload=b"not allowed",
        )


def test_loaded_contract_and_preregistration_raw_bytes_must_replay(
    tmp_path: Path,
) -> None:
    contract, prereg, capsule, _, _, _ = _campaign_objects(tmp_path)
    contract.value["fixed_paths"]["run_root"] = str(tmp_path / "changed")
    with pytest.raises(campaign.Q2CampaignValidationError, match="no longer"):
        campaign.validate_capsule(capsule, contract, prereg)

    raw = PREREGISTRATION_PATH.read_bytes()
    digest = campaign.raw_sha256(raw)
    assert campaign.validate_preregistration_payload(raw, digest) == prereg
    with pytest.raises(campaign.Q2CampaignValidationError, match="raw bytes"):
        campaign.validate_preregistration_payload(raw + b" ", digest)


def test_authorization_explicitly_accepts_both_complete_dispositions(
    tmp_path: Path,
) -> None:
    _, _, _, authorization, _, _ = _campaign_objects(tmp_path)
    assert authorization["pass_terminal_authorized"] is True
    assert authorization["complete_fail_terminal_authorized"] is True
    assert authorization["partial_finalization_artifact_to_terminal_only"] is True
    assert authorization["target_open_limit_by_role"] == {
        "IN_LANGUAGE_NULL": 1,
        "OUTSIDE_TARGET": 1,
    }
