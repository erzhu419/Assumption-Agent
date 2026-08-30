from __future__ import annotations

from dataclasses import replace

import pytest

from hegel_machine.phase3_q1_formal_archive_contract_v1 import Q1_TAG_REGISTRY
from hegel_machine.phase3_q1_prestart_wire_v1 import (
    FORMAL_ACTOR_REGISTRY,
    FORMAL_OUTPUT_LAYOUT_ROOT,
    FORMAL_OUTPUT_LAYOUT_ROWS,
    FORMAL_RESOURCE_ENVELOPE,
    PRESTART_TAG_REGISTRY,
    Q1_GATE_REGISTRY,
    Q1_GATE_REGISTRY_ROOT,
    Q1_PRESTART_GATE_COUNT,
    Q1_PRESTART_GATE_MASK,
    Q1_PRESTART_GATE_SUBJECT_REGISTRY,
    Q1PrestartAdmissionReceiptV1,
    Q1PrestartWireError,
    Q1FormalActorIdentityV1,
    Q1RunGenesisV1,
    Q1SourceIdentityV1,
    decode_prestart_admission_receipt_v1,
    decode_run_genesis_v1,
    gate_row_v1,
    q1_null_output_slots_v1,
    q1_prestart_authority_object_v1,
    verify_prestart_gate_evidence_v1,
)
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode


def _root(byte: int) -> bytes:
    return bytes((byte,)) * 32


def _genesis() -> Q1RunGenesisV1:
    source = Q1SourceIdentityV1(b"c" * 20, b"t" * 20, _root(1))
    implementations = (
        (b"python", _root(2)),
        (b"rust", _root(3)),
        (b"host", _root(4)),
    )
    return Q1RunGenesisV1(
        run_id=b"fresh-run-id-v1!",
        source_identity=source,
        q1_semantic_binding_root=_root(5),
        q1_projection_profile_root=_root(6),
        q0_14_of_14_receipt_root=_root(7),
        q05b_qualification_receipt_root=_root(8),
        full_v16_leaf_manifest_root=_root(9),
        final_implementation_root_rows=implementations,
        created_at_unix_seconds=1_788_000_000,
    )


def _subjects(gate_id: int, genesis: Q1RunGenesisV1) -> tuple[object, ...]:
    values: list[object] = []
    for index, (_, kind) in enumerate(
        Q1_PRESTART_GATE_SUBJECT_REGISTRY[gate_id - 1][1], 1
    ):
        if kind == b"root32":
            values.append(_root((gate_id * 11 + index) % 251 + 1))
        elif kind == b"bytes20":
            values.append(bytes((gate_id,)) * 20)
        elif kind == b"false":
            values.append(False)
        elif kind == b"formal_resource_envelope":
            values.append(FORMAL_RESOURCE_ENVELOPE)
        elif kind == b"implementation_rows":
            values.append(genesis.final_implementation_root_rows)
        elif kind == b"run_genesis_object":
            values.append(genesis.canonical_object())
        elif kind == b"null_output_slots":
            values.append(q1_null_output_slots_v1())
        else:  # pragma: no cover
            raise AssertionError(kind)
    if gate_id == 14:
        values[0] = genesis.genesis_root
    return tuple(values)


def _receipt() -> Q1PrestartAdmissionReceiptV1:
    genesis = _genesis()
    rows = tuple(
        gate_row_v1(genesis.source_identity, genesis, gate_id, _subjects(gate_id, genesis))
        for gate_id in range(1, Q1_PRESTART_GATE_COUNT + 1)
    )
    return Q1PrestartAdmissionReceiptV1(genesis, rows)


def test_control_plane_namespace_is_separate_from_frozen_archive_registry() -> None:
    archive_tags = {row[0] for row in Q1_TAG_REGISTRY}
    prestart_tags = {row[0] for row in PRESTART_TAG_REGISTRY}
    assert prestart_tags == {0x3B00, 0x3B01}
    assert archive_tags.isdisjoint(prestart_tags)
    assert max(archive_tags) == 0x370C


def test_formal_resource_envelope_is_exact_mechanical_candidate() -> None:
    signature_rows = FORMAL_RESOURCE_ENVELOPE[2]
    actor_rows = FORMAL_RESOURCE_ENVELOPE[3]
    assert signature_rows[0][1][-1] == (8, 7)
    assert signature_rows[1][1][3] == (4, 256)
    assert actor_rows[0][2] == actor_rows[1][2]
    assert actor_rows[0][2][2] == (11, 2_147_483_648)
    assert actor_rows[2][2] == (
        (9, 2_097_152), (10, 1), (11, 67_108_864), (12, 3600)
    )


def test_final_actor_identities_bind_action_closure_runtime_and_layout() -> None:
    identities = tuple(
        Q1FormalActorIdentityV1(
            actor_id=row[0],
            implementation_id=row[1],
            action_id=row[2],
            source_manifest_root=_root(20 + index),
            entrypoint_raw_sha256=_root(30 + index),
            dependency_closure_root=_root(40 + index),
            runtime_identity_root=_root(50 + index),
            offline_isolation_root=_root(60 + index),
        )
        for index, row in enumerate(FORMAL_ACTOR_REGISTRY, 1)
    )
    assert tuple(identity.actor_id for identity in identities) == (
        b"python", b"rust", b"host"
    )
    assert len({identity.implementation_root for identity in identities}) == 3
    assert all(identity.canonical_object()[10] == FORMAL_OUTPUT_LAYOUT_ROOT for identity in identities)
    assert FORMAL_OUTPUT_LAYOUT_ROWS[2][2].endswith(b"chunk-%06u.framed")

    with pytest.raises(Q1PrestartWireError, match="identity/action"):
        replace(identities[0], action_id=b"full-node6-capacity-preflight-v1")


def test_genesis_and_receipt_strict_round_trip() -> None:
    genesis = _genesis()
    assert decode_run_genesis_v1(genesis.canonical_bytes) == genesis
    receipt = _receipt()
    assert decode_prestart_admission_receipt_v1(receipt.canonical_bytes) == receipt
    verify_prestart_gate_evidence_v1(
        receipt,
        tuple(_subjects(gate_id, genesis) for gate_id in range(1, 15)),
    )
    assert len(receipt.receipt_root) == 32
    assert len(Q1_GATE_REGISTRY) == 20
    assert Q1_PRESTART_GATE_COUNT == 14
    assert Q1_PRESTART_GATE_MASK == 0x003FFF
    assert Q1_PRESTART_GATE_MASK.bit_count() == Q1_PRESTART_GATE_COUNT
    assert len(Q1_GATE_REGISTRY_ROOT) == 32


def test_authority_remains_not_run_with_eight_named_null_slots() -> None:
    authority = q1_prestart_authority_object_v1()
    assert authority[:5] == (0, 14, 0x003FFF, 20, 8)
    assert authority[5] == q1_null_output_slots_v1()
    assert tuple(row[0] for row in authority[5]) == tuple(range(1, 9))
    assert all(row[2] is None for row in authority[5])
    assert authority[6] is None
    assert authority[7] == 0
    assert authority[8] is None
    assert authority[9:] == (False, None, False, False, False, False, False, False, False)


def test_gate12_requires_exact_final_actor_rows() -> None:
    genesis = _genesis()
    subjects = list(_subjects(12, genesis))
    subjects[0] = (
        (b"python", _root(2)),
        (b"rust", _root(3)),
        (b"diagnostic-host", _root(4)),
    )
    with pytest.raises(Q1PrestartWireError, match="IMPLEMENTATION|actor"):
        gate_row_v1(genesis.source_identity, genesis, 12, tuple(subjects))


def test_gate13_requires_absence_booleans_and_gate14_exact_genesis() -> None:
    genesis = _genesis()
    gate13 = list(_subjects(13, genesis))
    gate13[3] = True
    with pytest.raises(Q1PrestartWireError, match="false"):
        gate_row_v1(genesis.source_identity, genesis, 13, tuple(gate13))

    gate14 = list(_subjects(14, genesis))
    gate14[0] = _root(99)
    with pytest.raises(Q1PrestartWireError, match="genesis root"):
        gate_row_v1(genesis.source_identity, genesis, 14, tuple(gate14))


def test_receipt_rejects_mutated_gate_mask_and_source_binding() -> None:
    receipt = _receipt()
    wire = list(receipt.canonical_object())
    wire[13] = 0x001FFF
    with pytest.raises(Q1PrestartWireError):
        decode_prestart_admission_receipt_v1(canonical_cbor_encode(tuple(wire)))

    other_source = Q1SourceIdentityV1(b"d" * 20, b"t" * 20, _root(1))
    with pytest.raises(Q1PrestartWireError, match="source identity/genesis"):
        gate_row_v1(other_source, receipt.genesis, 1, _subjects(1, receipt.genesis))

    subjects = [
        _subjects(gate_id, receipt.genesis) for gate_id in range(1, 15)
    ]
    first = list(subjects[0])
    first[0] = _root(88)
    subjects[0] = tuple(first)
    with pytest.raises(Q1PrestartWireError, match="evidence roots"):
        verify_prestart_gate_evidence_v1(receipt, tuple(subjects))


def test_genesis_rejects_diagnostic_actor_alias_and_bool_integer_alias() -> None:
    genesis = _genesis()
    bad_rows = (
        (b"python", _root(2)),
        (b"rust", _root(3)),
        (b"diagnostic-host", _root(4)),
    )
    with pytest.raises(Q1PrestartWireError, match="actor order"):
        replace(genesis, final_implementation_root_rows=bad_rows)
    with pytest.raises(Q1PrestartWireError, match="uint64"):
        replace(genesis, created_at_unix_seconds=True)
