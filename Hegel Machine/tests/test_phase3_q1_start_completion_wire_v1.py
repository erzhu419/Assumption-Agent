from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from hegel_machine import phase3_q1_start_completion_wire_v1 as wire
from hegel_machine.phase3_q1_formal_archive_contract_v1 import Q1_OUTPUT_SLOT_NAMES
from hegel_machine.phase3_q1_prestart_candidate_v1 import (
    FORMAL_INPUT_KEY_REGISTRY_ABSENCE_ROOT,
)
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode, content_hash


def _root(value: int) -> bytes:
    return bytes([value]) * 32


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _formal_capacity_rows() -> list[dict[str, int]]:
    fields = (
        "input_signature_id",
        "maximum_ast_depth",
        "maximum_ast_node_count",
        "maximum_raw_operator_applications",
        "maximum_behavior_classes",
        "maximum_visible_frontier_points",
        "maximum_visible_frontier_points_per_class",
        "maximum_continuation_bank_points",
        "maximum_continuation_bank_points_per_class",
        "maximum_work_queue_points",
        "maximum_saturation_rounds",
        "maximum_wall_time_seconds",
    )
    return [
        dict(
            zip(
                fields,
                (
                    signature_id,
                    3,
                    6,
                    *(value for _guard_id, value in wire.signature_guard_ceiling_rows_v1(signature_id)),
                    3_600,
                ),
                strict=True,
            )
        )
        for signature_id in (1, 2)
    ]


def _endpoint_result_payload(actor_id: bytes) -> bytes:
    actor = actor_id.decode("ascii")
    return _canonical_json(
        {
            "action_id": "materialize-full-node6-formal-archive-v1",
            "actor_id": actor,
            "actual_formal_peak_scratch_bytes": 1_700_000,
            "actual_formal_terminal_live_scratch_bytes": 0,
            "capsule_digest": "sha256:" + _root(90).hex(),
            "closure_bundle_root": "sha256:" + _root(22).hex(),
            "closure_semantics_version": "fixture-v1",
            "diagnostic_only": False,
            "dsl_version": "fixture-v1",
            "formal_archive_materialized": True,
            "formal_archive_payload_bytes": 5_000_000,
            "formal_capacity_limits_by_signature": _formal_capacity_rows(),
            "formal_output_bytes_limit": 16_777_216,
            "formal_scratch_bytes_limit": 134_217_728,
            "full_v16_structural_limits_applied": True,
            "implementation_id": f"hegel-{actor}-q1-formal-archive-materializer-v1",
            "maximum_ast_depth": 3,
            "maximum_ast_node_count": 6,
            "partitions": [
                {
                    "input_signature_id": row.input_signature_id,
                    "fixed_point_record_root": (
                        "sha256:" + row.fixed_point_record_root.hex()
                    ),
                    "signature_manifest_root": (
                        "sha256:" + row.signature_manifest_root.hex()
                    ),
                    "saturation_state_root": (
                        "sha256:" + row.saturation_state_root.hex()
                    ),
                }
                for row in _signature_rows()
            ],
            "prestart_admission_receipt_root": "sha256:" + _root(21).hex(),
            "q1_gate_count": 14,
            "q1_gate_mask": 16_383,
            "q1_gate_total": 20,
            "q1_output_slots": [None] * 8,
            "q1_state": "NOT_RUN",
            "role_evaluation_performed": False,
            "run_genesis_root": "sha256:" + _root(20).hex(),
            "schema_version": "hegel-phase3a-q1-full-node6-formal-archive-materializer/1",
            "source_commit": "a" * 40,
            "split_accessed": False,
            "status": "FULL_NODE6_FORMAL_ARCHIVE_MATERIALIZED_Q1_NOT_RUN",
            "target_truth_accessed": False,
        }
    )


def _host_result_payload() -> bytes:
    return _canonical_json(
        {
            "schema_version": "hegel-q1-formal-host-replay-result/1",
            "actor_id": "host",
            "implementation_id": "hegel-host-q1-formal-archive-replay-v1",
            "action_id": "replay-full-node6-formal-archive-v1",
            "status": "DUAL_FORMAL_ARCHIVE_EXACT_AGREEMENT",
            "capsule_digest": "sha256:" + _root(90).hex(),
            "source_commit": "a" * 40,
            "run_genesis_root": "sha256:" + _root(20).hex(),
            "prestart_admission_receipt_root": "sha256:" + _root(21).hex(),
            "q1_state": "NOT_RUN",
            "q1_gate_count": 14,
            "q1_gate_total": 20,
            "q1_gate_mask": 16_383,
            "q1_output_slots": [None] * 8,
            "formal_capacity_limits_by_signature": _formal_capacity_rows(),
            "formal_output_bytes_limit": 16_777_216,
            "formal_scratch_bytes_limit": 134_217_728,
            "semantic_binding_root": "sha256:" + _root(91).hex(),
            "projection_profile_root": "sha256:" + _root(92).hex(),
            "closure_bundle_root": "sha256:" + _root(22).hex(),
            "archive_payload_bytes_per_endpoint": 5_000_000,
            "formal_file_count_per_endpoint": 12,
            "signatures": [
                {
                    "input_signature_id": row.input_signature_id,
                    "universe_root": (
                        "sha256:" + _root(92 + row.input_signature_id).hex()
                    ),
                    "fixed_point_record_root": (
                        "sha256:" + row.fixed_point_record_root.hex()
                    ),
                    "signature_manifest_root": (
                        "sha256:" + row.signature_manifest_root.hex()
                    ),
                    "saturation_state_root": (
                        "sha256:" + row.saturation_state_root.hex()
                    ),
                    "record_count_by_stream": [1, 1, 1, 1],
                }
                for row in _signature_rows()
            ],
            "python_rust_formal_bytes_exact": True,
            "python_rust_replayed_roots_exact": True,
            "target_truth_accessed": False,
            "split_accessed": False,
            "role_evaluation_performed": False,
            "target_split_role_absence_attested": True,
            "q1_completion_receipt_issued": False,
            "state_published": False,
        }
    )


def _signature_rows() -> tuple[wire.Q1SignatureCompletionRowV1, ...]:
    return (
        wire.Q1SignatureCompletionRowV1(
            1,
            _root(1),
            _root(2),
            _root(3),
            wire.signature_guard_ceiling_rows_v1(1),
        ),
        wire.Q1SignatureCompletionRowV1(
            2,
            _root(4),
            _root(5),
            _root(6),
            wire.signature_guard_ceiling_rows_v1(2),
        ),
    )


def _result_binding_kwargs() -> dict[str, object]:
    return {
        "expected_capsule_digest": "sha256:" + _root(90).hex(),
        "expected_source_commit": "a" * 40,
        "expected_run_genesis_root": _root(20),
        "expected_prestart_admission_receipt_root": _root(21),
        "expected_closure_bundle_root": _root(22),
        "expected_signature_rows": _signature_rows(),
    }


def _endpoint(
    actor_id: bytes,
    implementation_root: bytes,
    result_root: bytes,
    *,
    signature_rows: tuple[wire.Q1SignatureCompletionRowV1, ...] | None = None,
) -> wire.Q1EndpointCompletionEvidenceV1:
    return wire.Q1EndpointCompletionEvidenceV1(
        actor_id=actor_id,
        actor_implementation_root=implementation_root,
        run_genesis_root=_root(20),
        prestart_admission_receipt_root=_root(21),
        endpoint_result_root=result_root,
        closure_bundle_root=_root(22),
        signature_rows=_signature_rows() if signature_rows is None else signature_rows,
        actual_output_bytes=5_900_000,
        actual_peak_scratch_bytes=1_700_000,
        actual_peak_rss_bytes=500_000_000,
        actual_wall_time_ns=20_000_000_000,
    )


def _complete_fixture() -> tuple[
    wire.Q1EndpointCompletionEvidenceV1,
    wire.Q1EndpointCompletionEvidenceV1,
    wire.Q1DualReplayAgreementV1,
    wire.Q1TargetBlindAccessLedgerV1,
    wire.Q1CompletionCoreV1,
    wire.Q1CompletionReceiptV1,
]:
    python_payload = _endpoint_result_payload(b"python")
    rust_payload = _endpoint_result_payload(b"rust")
    python = _endpoint(
        b"python",
        _root(30),
        wire.endpoint_result_root_v1(
            b"python", python_payload, **_result_binding_kwargs()
        ),
    )
    rust = _endpoint(
        b"rust",
        _root(32),
        wire.endpoint_result_root_v1(
            b"rust", rust_payload, **_result_binding_kwargs()
        ),
    )
    formal_files = (
        wire.Q1FileEvidenceV1("odd/fixed-point.framed", b"odd"),
        wire.Q1FileEvidenceV1("sink/fixed-point.framed", b"sink"),
    )
    agreement = wire.Q1DualReplayAgreementV1(
        run_genesis_root=_root(20),
        prestart_admission_receipt_root=_root(21),
        python_endpoint_evidence_root=python.evidence_root,
        rust_endpoint_evidence_root=rust.evidence_root,
        formal_file_closure_root=wire.formal_output_file_closure_root_v1(
            formal_files
        ),
        signature_rows=python.signature_rows,
        closure_bundle_root=_root(22),
        host_actor_implementation_root=_root(35),
        host_replay_result_root=wire.host_replay_result_root_v1(
            _host_result_payload(), **_result_binding_kwargs()
        ),
        actual_host_output_bytes=1_100_000,
        actual_host_peak_scratch_bytes=0,
        actual_host_peak_rss_bytes=25_000_000,
        actual_host_wall_time_ns=3_000_000_000,
    )
    actor_rows = tuple(
        wire.Q1IsolationEvidenceAdapterV1(
            actor_id=actor_id,
            actor_implementation_root=implementation_root,
            source_manifest_rows=(
                wire.Q1SourceManifestEvidenceRowV1(
                    f"actor-{offset}.py",
                    4,
                    _root(40 + offset),
                ),
            ),
            mount_manifest_rows=(
                wire.Q1MountManifestEvidenceRowV1(
                    "/output", f"/run/{actor_id.decode()}/output", True
                ),
                wire.Q1MountManifestEvidenceRowV1(
                    "/scratch", f"/run/{actor_id.decode()}/scratch", True
                ),
                wire.Q1MountManifestEvidenceRowV1(
                    "/source", f"/run/{actor_id.decode()}/source", False
                ),
            ),
            output_files=(
                wire.Q1FileEvidenceV1(
                    "host-result.json" if actor_id == b"host" else "endpoint-result.json",
                    _host_result_payload()
                    if actor_id == b"host"
                    else (python_payload if actor_id == b"python" else rust_payload),
                ),
            ),
            scratch_path=f"/run/{actor_id.decode()}/scratch",
        ).isolation_row()
        for offset, (actor_id, implementation_root) in enumerate(
            (
                (b"python", python.actor_implementation_root),
                (b"rust", rust.actor_implementation_root),
                (b"host", agreement.host_actor_implementation_root),
            )
        )
    )
    ledger = wire.Q1TargetBlindAccessLedgerV1(
        run_genesis_root=_root(20),
        prestart_admission_receipt_root=_root(21),
        dual_replay_agreement_root=agreement.agreement_root,
        mount_manifest_root=wire.runtime_mount_registry_root_v1(actor_rows),
        input_key_registry_absence_root=(
            wire.formal_input_key_registry_absence_root_v1()
        ),
        actor_rows=actor_rows,
    )
    subjects = (
        (
            _root(20),
            _root(21),
            (python.canonical_object(), rust.canonical_object()),
        ),
        (
            python.evidence_root,
            rust.evidence_root,
            agreement.formal_file_closure_root,
            tuple(row.canonical_object() for row in python.signature_rows),
            _root(22),
            True,
            True,
        ),
        (agreement.canonical_object(), agreement.agreement_root),
        (ledger.canonical_object(), ledger.ledger_root),
    )
    core = wire.Q1CompletionCoreV1(
        run_genesis_root=_root(20),
        prestart_admission_receipt_root=_root(21),
        gate_subject_rows_15_through_18=subjects,
        output_slots_1_through_7=wire.output_slots_1_through_7_v1(
            agreement, ledger
        ),
    )
    return python, rust, agreement, ledger, core, wire.Q1CompletionReceiptV1(core)


def test_start_registry_completes_exact_gates_15_through_20() -> None:
    assert tuple(row[0] for row in wire.Q1_START_GATE_SUBJECT_REGISTRY) == tuple(
        range(15, 21)
    )
    assert wire.Q1_COMPLETE_GATE_COUNT == 20
    assert wire.Q1_COMPLETE_GATE_MASK == 0x0FFFFF
    assert wire.START_COMPLETION_TAG_REGISTRY_ROOT != wire.Q1_START_GATE_SUBJECT_REGISTRY_ROOT


def test_signature_rows_bind_frozen_guard_ceilings_not_observed_peaks() -> None:
    odd = _signature_rows()[0]
    assert odd.count_guard_ceiling_rows == (
        (1, 32_768),
        (2, 1_024),
        (3, 4_096),
        (4, 32),
        (5, 8_192),
        (6, 4_096),
        (7, 16_384),
        (8, 7),
    )
    lower = list(odd.count_guard_ceiling_rows)
    lower[0] = (1, lower[0][1] - 1)
    with pytest.raises(wire.Q1StartCompletionWireError, match="frozen envelope"):
        replace(odd, count_guard_ceiling_rows=tuple(lower))
    higher = list(odd.count_guard_ceiling_rows)
    higher[0] = (1, higher[0][1] + 1)
    with pytest.raises(wire.Q1StartCompletionWireError, match="frozen envelope"):
        replace(odd, count_guard_ceiling_rows=tuple(higher))


def test_result_root_adapters_require_exact_successful_canonical_json() -> None:
    python_payload = _endpoint_result_payload(b"python")
    assert wire.endpoint_result_root_v1(
        b"python", python_payload, **_result_binding_kwargs()
    ) == sha256(
        wire.ENDPOINT_RESULT_JSON_ROOT_DOMAIN + python_payload
    ).digest()
    host_payload = _host_result_payload()
    assert wire.host_replay_result_root_v1(
        host_payload, **_result_binding_kwargs()
    ) == sha256(
        wire.HOST_RESULT_JSON_ROOT_DOMAIN + host_payload
    ).digest()

    changed = json.loads(python_payload)
    changed["status"] = "FAILED"
    with pytest.raises(wire.Q1StartCompletionWireError, match="success boundary"):
        wire.endpoint_result_root_v1(
            b"python", _canonical_json(changed), **_result_binding_kwargs()
        )
    with pytest.raises(wire.Q1StartCompletionWireError, match="canonical JSON"):
        wire.host_replay_result_root_v1(
            host_payload.rstrip(b"\n"), **_result_binding_kwargs()
        )

    stale = json.loads(python_payload)
    stale["capsule_digest"] = "sha256:" + _root(89).hex()
    with pytest.raises(wire.Q1StartCompletionWireError, match="current run binding"):
        wire.endpoint_result_root_v1(
            b"python", _canonical_json(stale), **_result_binding_kwargs()
        )
    stale_host = json.loads(host_payload)
    stale_host["signatures"][0]["signature_manifest_root"] = (
        "sha256:" + _root(88).hex()
    )
    with pytest.raises(wire.Q1StartCompletionWireError, match="signature roots"):
        wire.host_replay_result_root_v1(
            _canonical_json(stale_host), **_result_binding_kwargs()
        )


def test_file_closure_adapter_matches_supervisor_streaming_recipe() -> None:
    files = (
        wire.Q1FileEvidenceV1("odd/a.framed", b"a"),
        wire.Q1FileEvidenceV1("sink/b.framed", b"bc"),
    )
    digest = sha256(wire.FORMAL_FILE_CLOSURE_BYTES_DOMAIN)
    for row in files:
        path = row.relative_path.encode("utf-8")
        digest.update(len(path).to_bytes(4, "big"))
        digest.update(path)
        digest.update(len(row.payload).to_bytes(8, "big"))
        digest.update(row.payload)
    assert wire.formal_output_file_closure_root_v1(files) == digest.digest()
    with pytest.raises(wire.Q1StartCompletionWireError, match="path order"):
        wire.formal_output_file_closure_root_v1(tuple(reversed(files)))


def test_gate18_adapter_derives_all_roots_from_typed_preimages() -> None:
    source_rows = (
        wire.Q1SourceManifestEvidenceRowV1("actor.py", 5, _root(70)),
    )
    mount_rows = (
        wire.Q1MountManifestEvidenceRowV1("/output", "/run/python/output", True),
        wire.Q1MountManifestEvidenceRowV1("/scratch", "/run/python/scratch", True),
        wire.Q1MountManifestEvidenceRowV1("/source", "/run/python/source", False),
    )
    output_files = (wire.Q1FileEvidenceV1("endpoint-result.json", b"{}\n"),)
    adapter = wire.Q1IsolationEvidenceAdapterV1(
        actor_id=b"python",
        actor_implementation_root=_root(30),
        source_manifest_rows=source_rows,
        mount_manifest_rows=mount_rows,
        output_files=output_files,
        scratch_path="/run/python/scratch",
    )
    row = adapter.isolation_row()
    assert row.source_module_closure_root == wire.source_module_closure_root_v1(
        source_rows
    )
    assert row.mount_manifest_root == wire.mount_manifest_root_v1(mount_rows)
    assert row.output_closure_root == wire.output_file_closure_root_v1(output_files)
    assert row.scratch_cleanup_root == wire.scratch_cleanup_root_v1(
        "/run/python/scratch", absent=True
    )
    assert row.precheck_root == adapter.precheck_root
    assert row.postcheck_root == adapter.postcheck_root
    assert row.precheck_root != row.postcheck_root
    with pytest.raises(wire.Q1StartCompletionWireError, match="precheck derivation"):
        replace(row, precheck_root=_root(98))
    with pytest.raises(wire.Q1StartCompletionWireError, match="postcheck derivation"):
        replace(row, postcheck_root=_root(99))
    with pytest.raises(wire.Q1StartCompletionWireError, match="exact true"):
        wire.scratch_cleanup_root_v1("/run/python/scratch", absent=False)


def test_gate18_registry_roots_are_exact_and_ledger_enforces_them() -> None:
    _, _, _, ledger, _, _ = _complete_fixture()
    expected_mount_root = content_hash(
        wire.RUNTIME_MOUNT_REGISTRY_ROOT_DOMAIN,
        (
            1,
            b"hegel-q1-runtime-mount-registry/1",
            tuple(
                (row.actor_id, row.mount_manifest_root) for row in ledger.actor_rows
            ),
            False,
            False,
            False,
            False,
        ),
    )
    assert ledger.mount_manifest_root == expected_mount_root
    assert (
        wire.formal_input_key_registry_absence_root_v1()
        == FORMAL_INPUT_KEY_REGISTRY_ABSENCE_ROOT
        == ledger.input_key_registry_absence_root
    )

    with pytest.raises(wire.Q1StartCompletionWireError, match="actor order"):
        wire.runtime_mount_registry_root_v1(
            (ledger.actor_rows[1], ledger.actor_rows[0], ledger.actor_rows[2])
        )
    with pytest.raises(wire.Q1StartCompletionWireError, match="mount registry"):
        replace(ledger, mount_manifest_root=_root(60))
    with pytest.raises(wire.Q1StartCompletionWireError, match="input-key absence"):
        replace(ledger, input_key_registry_absence_root=_root(61))

    changed = list(ledger.canonical_object())
    actor_rows = list(changed[9])
    actor = list(actor_rows[0])
    actor[4] = _root(98)
    actor_rows[0] = tuple(actor)
    changed[9] = tuple(actor_rows)
    with pytest.raises(wire.Q1StartCompletionWireError, match="precheck derivation"):
        wire.decode_target_blind_access_ledger_v1(canonical_cbor_encode(tuple(changed)))


def test_dual_ledger_core_and_receipt_strict_round_trip() -> None:
    _, _, agreement, ledger, core, receipt = _complete_fixture()

    assert wire.decode_dual_replay_agreement_v1(
        agreement.canonical_bytes
    ) == agreement
    assert wire.decode_target_blind_access_ledger_v1(ledger.canonical_bytes) == ledger
    assert wire.decode_completion_core_v1(core.canonical_bytes) == core
    assert wire.decode_completion_receipt_v1(receipt.canonical_bytes) == receipt
    assert tuple(row[0] for row in receipt.gate_rows_15_through_20) == tuple(
        range(15, 21)
    )


def test_completion_receipt_projects_nonrecursive_slot8_and_complete_authority() -> None:
    _, _, _, _, core, receipt = _complete_fixture()

    assert receipt.output_slots[:7] == core.output_slots_1_through_7
    assert receipt.output_slots[7] == (
        8,
        Q1_OUTPUT_SLOT_NAMES[7],
        receipt.receipt_root,
    )
    authority = receipt.authority.canonical_object()
    assert authority[3:7] == (b"COMPLETE", 20, 0x0FFFFF, 20)
    assert authority[8][7] == (
        8,
        Q1_OUTPUT_SLOT_NAMES[7],
        wire.SLOT8_SELF_CONTENT_ROOT_POLICY,
    )
    assert authority[9:14] == (b"NOT_RUN", None, False, None, False)

    def contains_root(value: object) -> bool:
        if type(value) is bytes:
            return value == receipt.receipt_root
        if type(value) is tuple:
            return any(contains_root(child) for child in value)
        return False

    assert contains_root(receipt.canonical_object()) is False


def test_gate15_rejects_endpoint_resource_guard_exceedance() -> None:
    with pytest.raises(wire.Q1StartCompletionWireError, match="resource guard"):
        replace(
            _endpoint(b"python", _root(30), _root(31)),
            actual_peak_rss_bytes=2_147_483_649,
        )


def test_gate15_uses_successor_scratch_bound_without_mutating_prestart_wire() -> None:
    historical_python_limits = next(
        dict(rows)
        for _ordinal, actor_id, rows in wire.FORMAL_RESOURCE_ENVELOPE[3]
        if actor_id == b"python"
    )
    assert historical_python_limits[10] == 4_194_304
    assert wire.FORMAL_ENDPOINT_SCRATCH_BYTES_LIMIT == 134_217_728
    replace(
        _endpoint(b"python", _root(30), _root(31)),
        actual_peak_scratch_bytes=134_217_728,
    )
    with pytest.raises(wire.Q1StartCompletionWireError, match="resource guard"):
        replace(
            _endpoint(b"python", _root(30), _root(31)),
            actual_peak_scratch_bytes=134_217_729,
        )


def test_dual_replay_uses_successor_host_memory_bound_without_mutating_prestart_wire() -> None:
    historical_host_limits = next(
        dict(rows)
        for _ordinal, actor_id, rows in wire.FORMAL_RESOURCE_ENVELOPE[3]
        if actor_id == b"host"
    )
    assert historical_host_limits[11] == 67_108_864
    assert wire.FORMAL_HOST_MEMORY_BYTES_LIMIT == 2_147_483_648
    _, _, agreement, _, _, _ = _complete_fixture()

    replace(agreement, actual_host_peak_rss_bytes=111_919_104)
    replace(agreement, actual_host_peak_rss_bytes=2_147_483_648)
    with pytest.raises(wire.Q1StartCompletionWireError, match="resource guard"):
        replace(agreement, actual_host_peak_rss_bytes=2_147_483_649)


def test_gate15_decode_rejects_claimed_guard_hit() -> None:
    endpoint = _endpoint(b"python", _root(30), _root(31))
    changed = list(endpoint.canonical_object())
    guard_rows = list(changed[14])
    guard_rows[0] = (1, b"HIT")
    changed[14] = tuple(guard_rows)
    payload = (endpoint.run_genesis_root, endpoint.prestart_admission_receipt_root, (tuple(changed), _endpoint(b"rust", _root(32), _root(33)).canonical_object()))
    with pytest.raises(wire.Q1StartCompletionWireError, match="terminal wire"):
        wire.start_gate_evidence_root_v1(15, payload)


def test_completion_core_rejects_cross_endpoint_state_mismatch() -> None:
    python, _, agreement, ledger, core, _ = _complete_fixture()
    changed_rows = list(_signature_rows())
    changed_rows[1] = replace(changed_rows[1], saturation_state_root=_root(99))
    rust = _endpoint(
        b"rust",
        _root(32),
        _root(33),
        signature_rows=tuple(changed_rows),
    )
    subjects = list(core.gate_subject_rows_15_through_18)
    subjects[0] = (
        _root(20),
        _root(21),
        (python.canonical_object(), rust.canonical_object()),
    )
    with pytest.raises(wire.Q1StartCompletionWireError, match="equality binding"):
        wire.Q1CompletionCoreV1(
            core.run_genesis_root,
            core.prestart_admission_receipt_root,
            tuple(subjects),
            wire.output_slots_1_through_7_v1(agreement, ledger),
        )


def test_dual_replay_decoder_rejects_bool_integer_alias() -> None:
    _, _, agreement, _, _, _ = _complete_fixture()
    changed = list(agreement.canonical_object())
    changed[18] = 1
    with pytest.raises(wire.Q1StartCompletionWireError, match="fixed wire"):
        wire.decode_dual_replay_agreement_v1(canonical_cbor_encode(tuple(changed)))


def test_access_ledger_rejects_forbidden_access_event_and_network_flag() -> None:
    _, _, _, ledger, _, _ = _complete_fixture()
    changed = list(ledger.canonical_object())
    changed[10] = ((1, b"FORBIDDEN_INPUT_ACCESS"),)
    with pytest.raises(wire.Q1StartCompletionWireError, match="fixed wire"):
        wire.decode_target_blind_access_ledger_v1(canonical_cbor_encode(tuple(changed)))

    changed = list(ledger.canonical_object())
    actor_rows = list(changed[9])
    actor = list(actor_rows[0])
    actor[11] = True
    actor_rows[0] = tuple(actor)
    changed[9] = tuple(actor_rows)
    with pytest.raises(wire.Q1StartCompletionWireError, match="postcheck flags"):
        wire.decode_target_blind_access_ledger_v1(canonical_cbor_encode(tuple(changed)))


def test_completion_core_rejects_wrong_formal_output_slot() -> None:
    _, _, _, _, core, _ = _complete_fixture()
    slots = list(core.output_slots_1_through_7)
    slots[4] = (5, Q1_OUTPUT_SLOT_NAMES[4], _root(98))
    with pytest.raises(wire.Q1StartCompletionWireError, match="output slots"):
        replace(core, output_slots_1_through_7=tuple(slots))


def test_completion_receipt_rejects_gate20_and_slot8_policy_tamper() -> None:
    _, _, _, _, _, receipt = _complete_fixture()
    changed = list(receipt.canonical_object())
    gate_rows = list(changed[9])
    gate20 = list(gate_rows[-1])
    gate20[3] = _root(97)
    gate_rows[-1] = tuple(gate20)
    changed[9] = tuple(gate_rows)
    with pytest.raises(wire.Q1StartCompletionWireError, match="replay differs"):
        wire.decode_completion_receipt_v1(canonical_cbor_encode(tuple(changed)))

    changed = list(receipt.canonical_object())
    changed[14] = (1, b"wrong")
    with pytest.raises(wire.Q1StartCompletionWireError, match="fixed wire"):
        wire.decode_completion_receipt_v1(canonical_cbor_encode(tuple(changed)))
