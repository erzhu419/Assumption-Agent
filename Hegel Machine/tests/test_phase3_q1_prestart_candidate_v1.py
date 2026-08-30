from __future__ import annotations

from dataclasses import replace

import pytest

from hegel_machine import phase3_q1_prestart_candidate_v1 as candidate
from hegel_machine import phase3_q1_prestart_wire_v1 as wire


def _root(value: int) -> bytes:
    return bytes((value,)) * 32


def _source() -> wire.Q1SourceIdentityV1:
    return wire.Q1SourceIdentityV1(b"c" * 20, b"t" * 20, _root(1))


def _actors() -> tuple[wire.Q1FormalActorIdentityV1, ...]:
    return tuple(
        wire.Q1FormalActorIdentityV1(
            actor_id=row[0],
            implementation_id=row[1],
            action_id=row[2],
            source_manifest_root=_root(10 + index),
            entrypoint_raw_sha256=_root(20 + index),
            dependency_closure_root=_root(30 + index),
            runtime_identity_root=_root(40 + index),
            offline_isolation_root=_root(50 + index),
        )
        for index, row in enumerate(wire.FORMAL_ACTOR_REGISTRY, 1)
    )


def _resource_candidate() -> dict[str, object]:
    names = (
        "maximum_raw_operator_applications",
        "maximum_behavior_classes",
        "maximum_visible_frontier_points",
        "maximum_visible_frontier_points_per_class",
        "maximum_continuation_bank_points",
        "maximum_continuation_bank_points_per_class",
        "maximum_work_queue_points",
        "maximum_saturation_rounds",
    )
    rows = []
    for signature_id, guard_rows in wire.FORMAL_RESOURCE_ENVELOPE[2]:
        rows.append(
            {
                "input_signature_id": signature_id,
                "formal_guard": {
                    name: value
                    for name, (_guard_id, value) in zip(names, guard_rows, strict=True)
                },
            }
        )
    actor_rows = wire.FORMAL_RESOURCE_ENVELOPE[3]
    endpoint = {
        "output_bytes": actor_rows[0][2][0][1],
        "scratch_bytes": actor_rows[0][2][1][1],
        "memory_bytes": actor_rows[0][2][2][1],
        "wall_time_seconds": actor_rows[0][2][3][1],
    }
    host = {
        "output_bytes": actor_rows[2][2][0][1],
        "scratch_bytes": actor_rows[2][2][1][1],
        "memory_bytes": actor_rows[2][2][2][1],
        "wall_time_seconds": actor_rows[2][2][3][1],
    }
    return {
        "dual_semantic_agreement_sha256": "sha256:" + "aa" * 32,
        "formal_resource_envelope": {
            "endpoint": endpoint,
            "trusted_host_replay": host,
        },
        "per_signature_formal_count_guards": rows,
    }


@pytest.fixture
def isolated_predecessor_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    q0_gates = [
        {
            "gate_id": gate_id,
            "name": f"Q0_GATE_{gate_id}",
            "passed": True,
            "pending_dual": False,
        }
        for gate_id in range(1, 15)
    ]
    monkeypatch.setattr(candidate, "_validate_q0_v1", lambda _payload: ({}, q0_gates))
    monkeypatch.setattr(candidate, "_validate_q05b_v1", lambda *_args: None)
    monkeypatch.setattr(candidate, "_validate_v6_terminal_v1", lambda _payload: None)
    monkeypatch.setattr(
        candidate._resource,
        "derive_formal_resource_amendment_v1",
        lambda *_args: _resource_candidate(),
    )


def _derive() -> dict[str, object]:
    return candidate.derive_q1_prestart_candidate_v1(
        b"q0",
        b"q05b-terminal",
        b"q05b-stdout",
        b"v6-terminal",
        b"v6-artifact",
        b"v6-python",
        b"v6-rust",
        source_identity=_source(),
        actor_identities=_actors(),
        run_id=b"fresh-run-id-v1!",
        created_at_unix_seconds=1_788_000_000,
        prestart_wire_contract_raw_sha256=_root(70),
        mount_manifest_root=_root(71),
    )


def test_candidate_derivation_builds_exact_14_of_20_without_publication(
    isolated_predecessor_replay: None,
) -> None:
    first = _derive()
    second = _derive()
    assert first == second
    assert first["status"] == candidate.STATUS
    assert first["candidate_only"] is True
    assert first["published"] is False
    assert first["q1_execution_started"] is False
    assert first["q1_state_before_authoritative_apply"] == "NOT_RUN"
    assert first["q1_gate_count_before_authoritative_apply"] == 0
    assert first["intended_q1_gate_count"] == 14
    assert first["intended_q1_gate_mask"] == 0x003FFF
    assert first["q1_output_slots"] == [None] * 8
    assert len(first["gate_evidence_rows"]) == 14

    genesis = wire.decode_run_genesis_v1(
        bytes.fromhex(first["run_genesis_cbor_hex"])
    )
    receipt = wire.decode_prestart_admission_receipt_v1(
        bytes.fromhex(first["prestart_admission_receipt_cbor_hex"])
    )
    assert first["run_genesis_root"] == "sha256:" + genesis.genesis_root.hex()
    assert first["prestart_admission_receipt_root"] == (
        "sha256:" + receipt.receipt_root.hex()
    )


def test_candidate_rejects_diagnostic_actor_alias(
    isolated_predecessor_replay: None,
) -> None:
    actors = list(_actors())
    with pytest.raises(wire.Q1PrestartWireError, match="identity/action"):
        actors[0] = replace(
            actors[0],
            action_id=b"full-node6-capacity-preflight-v1",
        )


def test_resource_candidate_must_equal_mechanical_envelope() -> None:
    exact = _resource_candidate()
    candidate._validate_resource_envelope_v1(exact)
    exact["formal_resource_envelope"]["endpoint"]["memory_bytes"] = 1
    with pytest.raises(candidate.Q1PrestartCandidateError, match="endpoint envelope"):
        candidate._validate_resource_envelope_v1(exact)
