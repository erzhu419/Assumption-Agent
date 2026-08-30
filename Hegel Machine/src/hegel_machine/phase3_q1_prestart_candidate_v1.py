"""Pure derivation of the Q1 14/20 prestart candidate.

The function in this module replays only small local predecessor records and
the exact final actor identities supplied by an outer preflight.  It builds a
candidate genesis and receipt in memory.  It has no filesystem, process,
network, ledger, publication, authorization-consumption, or Q1-start ability.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Final, NoReturn

from . import phase3_q05b_wire_qualification_contract_v1 as _q05b
from . import phase3_q1_formal_resource_amendment_v1 as _resource
from . import phase3_q1_full_node6_projection_v1 as _projection
from . import phase3_q1_prestart_wire_v1 as _wire
from .phase3_q1_universe_v1 import production_universe_v1
from .strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode, content_hash


SCHEMA_VERSION: Final = "hegel-phase3a-q1-prestart-candidate/1"
ACTION_ID: Final = "phase3-q1-prestart-admission-v1"
STATUS: Final = "Q1_PRESTART_CANDIDATE_NOT_AUTHORIZED_NOT_PUBLISHED"

Q0_ARTIFACT_SHA256: Final = bytes.fromhex(
    "1d8d530cbccf133af97da900164b89c9385db21d873bfff039ee7ce07c45f28d"
)
Q0_SOURCE_COMMIT: Final = bytes.fromhex(
    "ae201905526747980ff8e46e70f9ec96de93d3eb"
)
Q0_RECEIPT_ROOT: Final = bytes.fromhex(
    "ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad2"
)
Q05B_TERMINAL_RAW_SHA256: Final = bytes.fromhex(
    "d8dfc08daa1bac4e8e26782c67ea953c53439de6e5464a8f215f979d0b5ee952"
)
Q05B_SUPERVISOR_STDOUT_SHA256: Final = bytes.fromhex(
    "28d61af0198d14eeddb4ac769116fd191720d0e73a380a717d71712c1fc75180"
)
Q05B_ARTIFACT_SHA256: Final = bytes.fromhex(
    "7bbbe13efbc43a128c7310fa3a3a32c0545fe0d55acf03d6212742ebdef3c700"
)
Q05B_RECEIPT_ROOT: Final = bytes.fromhex(
    "06d219b403c0302862badc020edc6409f44f78fe26e8d30a23d178dae4410c52"
)
V6_TERMINAL_RAW_SHA256: Final = bytes.fromhex(
    "d0c11ef939df13e70502d2f623a44fc42fefef0beaffeb3a0d1ba77e36521624"
)
V6_TERMINAL_RECORD_DIGEST: Final = bytes.fromhex(
    "6385c52fb63b9745ffa414fc90eef3362b42f2cb8601c64ee79c508011e20fa9"
)
V6_OUTCOME_DIGEST: Final = bytes.fromhex(
    "4a29e82d50cb29bfa9b61c980568885296511820b7ced1b91571dd06a13985ab"
)
POST_SHRINK6_NORMATIVE_RAW_SHA256: Final = bytes.fromhex(
    "1df8d3ff3ede2cbead98e7901a3e82b91c460ad1d5eb0d1af78938e7b2d23b95"
)
Q1_FREEZE_RAW_SHA256: Final = bytes.fromhex(
    "e106aa9ccab264aae73df8c3fc69de957bff9ec9bb630155cf765e348c84a4f3"
)

Q0_GATE_BUNDLE_DOMAIN: Final = "HEGEL/Q1/PRESTART/Q0_GATE_BUNDLE/V1"
ACTOR_SOURCE_CLOSURE_DOMAIN: Final = "HEGEL/Q1/FORMAL_ACTOR_SOURCE_CLOSURE/V1"
OFFLINE_ISOLATION_BUNDLE_DOMAIN: Final = "HEGEL/Q1/FORMAL_OFFLINE_ISOLATION/V1"
INPUT_KEY_ABSENCE_DOMAIN: Final = "HEGEL/Q1/FORMAL_INPUT_KEY_ABSENCE/V1"

FORMAL_INPUT_KEY_REGISTRY: Final = ()
FORMAL_INPUT_KEY_REGISTRY_ABSENCE_ROOT: Final = content_hash(
    INPUT_KEY_ABSENCE_DOMAIN,
    (1, b"hegel-q1-formal-input-key-absence/1", FORMAL_INPUT_KEY_REGISTRY,
     False, False, False),
)


class Q1PrestartCandidateError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1PrestartCandidateError(code, detail)


def _raw_sha(payload: bytes, expected: bytes, label: str) -> None:
    if type(payload) is not bytes or sha256(payload).digest() != expected:
        _fail("REJECT_Q1_PRESTART_PREDECESSOR", f"{label} raw SHA-256 differs")


def _root32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        _fail("REJECT_Q1_PRESTART_ROOT", f"{label} must be exactly 32 bytes")
    return value


def _canonical_object(payload: bytes, label: str) -> dict[str, object]:
    try:
        return _resource.decode_canonical_json_object_v1(payload, label)
    except _resource.FormalResourceAmendmentError as error:
        _fail("REJECT_Q1_PRESTART_PREDECESSOR", f"{label}: {error}")


def _q0_gate_bundle_root_v1(
    gates: list[object], gate_ids: tuple[int, ...]
) -> bytes:
    rows: list[tuple[object, ...]] = []
    for gate_id in gate_ids:
        raw = gates[gate_id - 1]
        if type(raw) is not dict:
            _fail("REJECT_Q1_PRESTART_Q0", "Q0 gate row type differs")
        expected_name = raw.get("name")
        if (
            raw.get("gate_id") != gate_id
            or type(raw.get("gate_id")) is not int
            or type(expected_name) is not str
            or raw.get("passed") is not True
            or raw.get("pending_dual") is not False
        ):
            _fail("REJECT_Q1_PRESTART_Q0", f"Q0 Gate {gate_id} did not pass")
        encoded = _resource.canonical_json_bytes_v1(raw)
        rows.append((gate_id, expected_name.encode("ascii"), sha256(encoded).digest()))
    return content_hash(Q0_GATE_BUNDLE_DOMAIN, tuple(rows))


def _validate_q0_v1(payload: bytes) -> tuple[dict[str, object], list[object]]:
    _raw_sha(payload, Q0_ARTIFACT_SHA256, "Q0 artifact")
    value = _canonical_object(payload, "Q0 artifact")
    if (
        value.get("schema_version")
        != "hegel-phase3a-q0-dual-qualification-evidence/1"
        or value.get("status") != "DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS"
        or value.get("claim_scope") != "Q0_DUAL_ENGINEERING_QUALIFICATION_ONLY"
        or value.get("source_commit") != Q0_SOURCE_COMMIT.hex()
        or value.get("q0_state") != "QUALIFIED_NOT_Q1_RUN"
        or value.get("readiness_gate_total") != 14
        or type(value.get("readiness_gate_total")) is not int
        or value.get("readiness_gates_passed") != 14
        or type(value.get("readiness_gates_passed")) is not int
        or value.get("readiness_gate_mask") != 0x3FFF
        or type(value.get("readiness_gate_mask")) is not int
        or value.get("receipt_root") != "sha256:" + Q0_RECEIPT_ROOT.hex()
        or value.get("q1_status_id") != 0
        or value.get("q1_output_root") is not None
        or value.get("role_evaluation_performed") is not False
        or value.get("outside_certificate_issued") is not False
        or value.get("m3_formal_roots") is not None
    ):
        _fail("REJECT_Q1_PRESTART_Q0", "Q0 terminal boundary differs")
    gates = value.get("gates")
    if type(gates) is not list or len(gates) != 14:
        _fail("REJECT_Q1_PRESTART_Q0", "Q0 gate registry differs")
    for gate_id in range(1, 15):
        _q0_gate_bundle_root_v1(gates, (gate_id,))
    receipt_hex = value.get("receipt_cbor_hex")
    if type(receipt_hex) is not str:
        _fail("REJECT_Q1_PRESTART_Q0", "Q0 receipt bytes are absent")
    try:
        receipt_bytes = bytes.fromhex(receipt_hex)
        receipt_object = canonical_cbor_decode(receipt_bytes)
    except (ValueError, TypeError) as error:
        _fail("REJECT_Q1_PRESTART_Q0", f"Q0 receipt decode failed: {error}")
    if canonical_cbor_encode(receipt_object) != receipt_bytes or content_hash(
        "HEGEL/Q0/SATURATION_RECEIPT/V1", receipt_object
    ) != Q0_RECEIPT_ROOT:
        _fail("REJECT_Q1_PRESTART_Q0", "Q0 receipt replay root differs")
    return value, gates


def _validate_q05b_v1(
    terminal_payload: bytes,
    supervisor_stdout_payload: bytes,
) -> None:
    _raw_sha(terminal_payload, Q05B_TERMINAL_RAW_SHA256, "Q0.5b terminal")
    _raw_sha(
        supervisor_stdout_payload,
        Q05B_SUPERVISOR_STDOUT_SHA256,
        "Q0.5b supervisor stdout",
    )
    terminal = _canonical_object(terminal_payload, "Q0.5b terminal")
    stdout = _canonical_object(supervisor_stdout_payload, "Q0.5b supervisor stdout")
    body = terminal.get("body")
    if type(body) is not dict:
        _fail("REJECT_Q1_PRESTART_Q05B", "Q0.5b terminal body differs")
    outcome = body.get("outcome")
    delivery = stdout.get("actual_final_delivery_identity")
    if (
        type(outcome) is not dict
        or type(delivery) is not dict
        or terminal.get("record_digest")
        != "sha256:c5494513991a1c94e4d43116f62d4b829f0deeb643f992d6daf49168271166b1"
        or terminal.get("record_type") != "TERMINAL"
        or terminal.get("status") != "ATTEMPT2_TARGET_D_V2_QUALIFIED_Q1_NOT_RUN"
        or outcome.get("status") != "ATTEMPT2_TARGET_D_V2_QUALIFIED_Q1_NOT_RUN"
        or outcome.get("q1_state") != "NOT_RUN"
        or outcome.get("retry_or_resume_allowed") is not False
        or outcome.get("stdout_sha256") != Q05B_SUPERVISOR_STDOUT_SHA256.hex()
        or outcome.get("artifact_sha256_or_null") != Q05B_ARTIFACT_SHA256.hex()
        or stdout.get("artifact_sha256") != Q05B_ARTIFACT_SHA256.hex()
        or stdout.get("final_receipt_root") != Q05B_RECEIPT_ROOT.hex()
        or delivery.get("payload_sha256") != Q05B_ARTIFACT_SHA256.hex()
        or delivery.get("final_receipt_root") != Q05B_RECEIPT_ROOT.hex()
        or stdout.get("qualification_count") != 20
        or type(stdout.get("qualification_count")) is not int
        or stdout.get("qualification_mask") != (1 << 20) - 1
        or type(stdout.get("qualification_mask")) is not int
        or stdout.get("q1_state") != "NOT_RUN"
        or stdout.get("q1_gate_count") != 0
        or type(stdout.get("q1_gate_count")) is not int
        or stdout.get("q1_gate_mask") != 0
        or type(stdout.get("q1_gate_mask")) is not int
        or stdout.get("q1_formal_output_roots") != [None] * 8
    ):
        _fail("REJECT_Q1_PRESTART_Q05B", "Q0.5b terminal replay differs")


def _validate_v6_terminal_v1(payload: bytes) -> None:
    _raw_sha(payload, V6_TERMINAL_RAW_SHA256, "v6 terminal")
    value = _canonical_object(payload, "v6 terminal")
    body = value.get("body")
    if type(body) is not dict or type(body.get("outcome")) is not dict:
        _fail("REJECT_Q1_PRESTART_V6", "v6 terminal body differs")
    outcome = body["outcome"]
    if (
        value.get("record_digest") != "sha256:" + V6_TERMINAL_RECORD_DIGEST.hex()
        or value.get("record_type") != "TERMINAL"
        or value.get("status") != _resource.V6_TERMINAL_STATUS
        or outcome.get("outcome_digest") != "sha256:" + V6_OUTCOME_DIGEST.hex()
        or outcome.get("artifact_sha256_or_null") != _resource.V6_ARTIFACT_SHA256
        or outcome.get("capsule_digest") != _resource.V6_CAPSULE_DIGEST
        or outcome.get("status") != _resource.V6_TERMINAL_STATUS
        or outcome.get("success") is not True
        or outcome.get("retry_or_resume_allowed") is not False
        or outcome.get("q1_state") != "NOT_RUN"
        or outcome.get("q1_gate_count") != 0
        or type(outcome.get("q1_gate_count")) is not int
        or outcome.get("q1_gate_mask") != 0
        or type(outcome.get("q1_gate_mask")) is not int
        or outcome.get("q1_output_slots") != [None] * 8
    ):
        _fail("REJECT_Q1_PRESTART_V6", "v6 terminal replay differs")


def _validate_resource_envelope_v1(candidate: dict[str, object]) -> None:
    per_signature = candidate.get("per_signature_formal_count_guards")
    resources = candidate.get("formal_resource_envelope")
    if type(per_signature) is not list or len(per_signature) != 2 or type(resources) is not dict:
        _fail("REJECT_Q1_PRESTART_RESOURCE", "resource candidate shape differs")
    expected_signature = _wire.FORMAL_RESOURCE_ENVELOPE[2]
    guard_names = (
        "maximum_raw_operator_applications",
        "maximum_behavior_classes",
        "maximum_visible_frontier_points",
        "maximum_visible_frontier_points_per_class",
        "maximum_continuation_bank_points",
        "maximum_continuation_bank_points_per_class",
        "maximum_work_queue_points",
        "maximum_saturation_rounds",
    )
    for row, expected in zip(per_signature, expected_signature, strict=True):
        if type(row) is not dict or row.get("input_signature_id") != expected[0]:
            _fail("REJECT_Q1_PRESTART_RESOURCE", "signature resource row differs")
        guard = row.get("formal_guard")
        if type(guard) is not dict or tuple(guard.get(name) for name in guard_names) != tuple(
            value for _guard_id, value in expected[1]
        ):
            _fail("REJECT_Q1_PRESTART_RESOURCE", "signature guard values differ")
    endpoint = resources.get("endpoint")
    host = resources.get("trusted_host_replay")
    if type(endpoint) is not dict or type(host) is not dict:
        _fail("REJECT_Q1_PRESTART_RESOURCE", "actor resource rows differ")
    names = ("output_bytes", "scratch_bytes", "memory_bytes", "wall_time_seconds")
    expected_actors = _wire.FORMAL_RESOURCE_ENVELOPE[3]
    for actor in expected_actors[:2]:
        if tuple(endpoint.get(name) for name in names) != tuple(
            value for guard_id, value in actor[2] if guard_id in (9, 10, 11, 12)
        ):
            _fail("REJECT_Q1_PRESTART_RESOURCE", "endpoint envelope differs")
    if tuple(host.get(name) for name in names) != tuple(
        value for guard_id, value in expected_actors[2][2] if guard_id in (9, 10, 11, 12)
    ):
        _fail("REJECT_Q1_PRESTART_RESOURCE", "host envelope differs")


def derive_q1_prestart_candidate_v1(
    q0_artifact_payload: bytes,
    q05b_terminal_payload: bytes,
    q05b_supervisor_stdout_payload: bytes,
    v6_terminal_payload: bytes,
    v6_artifact_payload: bytes,
    v6_python_stdout_payload: bytes,
    v6_rust_stdout_payload: bytes,
    *,
    source_identity: _wire.Q1SourceIdentityV1,
    actor_identities: tuple[_wire.Q1FormalActorIdentityV1, ...],
    run_id: bytes,
    created_at_unix_seconds: int,
    prestart_wire_contract_raw_sha256: bytes,
    mount_manifest_root: bytes,
) -> dict[str, object]:
    """Return one deterministic, non-authoritative prestart candidate."""

    _q0, q0_gates = _validate_q0_v1(q0_artifact_payload)
    _validate_q05b_v1(q05b_terminal_payload, q05b_supervisor_stdout_payload)
    _validate_v6_terminal_v1(v6_terminal_payload)
    resource_candidate = _resource.derive_formal_resource_amendment_v1(
        v6_artifact_payload,
        v6_python_stdout_payload,
        v6_rust_stdout_payload,
    )
    _validate_resource_envelope_v1(resource_candidate)
    if type(source_identity) is not _wire.Q1SourceIdentityV1:
        _fail("REJECT_Q1_PRESTART_SOURCE", "source identity exact type differs")
    if type(actor_identities) is not tuple or len(actor_identities) != 3:
        _fail("REJECT_Q1_PRESTART_ACTORS", "three actor identities are required")
    if tuple(identity.actor_id for identity in actor_identities) != (
        b"python", b"rust", b"host",
    ) or any(type(identity) is not _wire.Q1FormalActorIdentityV1 for identity in actor_identities):
        _fail("REJECT_Q1_PRESTART_ACTORS", "actor identity registry differs")
    prestart_contract_root = _root32(
        prestart_wire_contract_raw_sha256, "prestart wire contract raw SHA-256"
    )
    mount_root = _root32(mount_manifest_root, "mount manifest root")
    implementation_rows = tuple(
        (identity.actor_id, identity.implementation_root)
        for identity in actor_identities
    )
    actor_source_closure_root = content_hash(
        ACTOR_SOURCE_CLOSURE_DOMAIN,
        tuple((identity.actor_id, identity.source_manifest_root) for identity in actor_identities),
    )
    offline_isolation_root = content_hash(
        OFFLINE_ISOLATION_BUNDLE_DOMAIN,
        tuple((identity.actor_id, identity.offline_isolation_root) for identity in actor_identities),
    )
    genesis = _wire.Q1RunGenesisV1(
        run_id=run_id,
        source_identity=source_identity,
        q1_semantic_binding_root=_projection.FROZEN_SEMANTIC_BINDING_ROOT,
        q1_projection_profile_root=_projection.FROZEN_PROJECTION_PROFILE_ROOT,
        q0_14_of_14_receipt_root=Q0_RECEIPT_ROOT,
        q05b_qualification_receipt_root=Q05B_RECEIPT_ROOT,
        full_v16_leaf_manifest_root=_q05b.full_v16_leaf_manifest_v1().manifest_root,
        final_implementation_root_rows=implementation_rows,
        created_at_unix_seconds=created_at_unix_seconds,
    )
    agreement = resource_candidate.get("dual_semantic_agreement_sha256")
    if type(agreement) is not str or not agreement.startswith("sha256:"):
        _fail("REJECT_Q1_PRESTART_V6", "v6 dual agreement root differs")
    dual_agreement_root = bytes.fromhex(agreement[7:])
    v6_capsule_root = bytes.fromhex(_resource.V6_CAPSULE_DIGEST[7:])
    v6_artifact_root = bytes.fromhex(_resource.V6_ARTIFACT_SHA256[7:])
    subjects = (
        (POST_SHRINK6_NORMATIVE_RAW_SHA256, Q1_FREEZE_RAW_SHA256, prestart_contract_root),
        (Q0_ARTIFACT_SHA256, Q0_RECEIPT_ROOT, Q0_SOURCE_COMMIT),
        (Q0_RECEIPT_ROOT, _q05b.full_v16_leaf_manifest_v1().manifest_root,
         _projection.FROZEN_SEMANTIC_BINDING_ROOT),
        (Q0_RECEIPT_ROOT, production_universe_v1(1).universe_root,
         production_universe_v1(2).universe_root,
         _q0_gate_bundle_root_v1(q0_gates, (3, 5))),
        (Q0_RECEIPT_ROOT, Q05B_RECEIPT_ROOT,
         _q0_gate_bundle_root_v1(q0_gates, (4, 6, 12)), dual_agreement_root),
        (Q0_RECEIPT_ROOT, Q05B_RECEIPT_ROOT,
         _q0_gate_bundle_root_v1(q0_gates, (7,))),
        (Q0_RECEIPT_ROOT, Q05B_RECEIPT_ROOT,
         _q0_gate_bundle_root_v1(q0_gates, (8,))),
        (Q0_RECEIPT_ROOT, Q05B_RECEIPT_ROOT,
         _q0_gate_bundle_root_v1(q0_gates, (9,))),
        (Q0_RECEIPT_ROOT, v6_artifact_root,
         _q0_gate_bundle_root_v1(q0_gates, (10,))),
        (Q05B_ARTIFACT_SHA256, Q05B_RECEIPT_ROOT, actor_source_closure_root),
        (v6_capsule_root, V6_TERMINAL_RECORD_DIGEST, v6_artifact_root,
         V6_OUTCOME_DIGEST, _wire.FORMAL_RESOURCE_ENVELOPE),
        (implementation_rows, actor_source_closure_root, offline_isolation_root,
         _wire.FORMAL_OUTPUT_LAYOUT_ROOT),
        (actor_source_closure_root, mount_root, FORMAL_INPUT_KEY_REGISTRY_ABSENCE_ROOT,
         False, False, False),
        (genesis.genesis_root, genesis.canonical_object(),
         _wire.q1_null_output_slots_v1(), False),
    )
    gate_rows = tuple(
        _wire.gate_row_v1(source_identity, genesis, gate_id, row)
        for gate_id, row in enumerate(subjects, 1)
    )
    receipt = _wire.Q1PrestartAdmissionReceiptV1(genesis, gate_rows)
    _wire.verify_prestart_gate_evidence_v1(receipt, subjects)
    evidence_rows = [
        {
            "evidence_root": "sha256:" + gate_rows[index - 1][3].hex(),
            "gate_id": index,
            "gate_name": gate_rows[index - 1][1].decode("ascii"),
            "subjects_cbor_hex": canonical_cbor_encode(subjects[index - 1]).hex(),
        }
        for index in range(1, 15)
    ]
    return {
        "action_id": ACTION_ID,
        "actor_implementation_roots": {
            identity.actor_id.decode("ascii"): "sha256:" + identity.implementation_root.hex()
            for identity in actor_identities
        },
        "candidate_only": True,
        "formal_output_layout_root": "sha256:" + _wire.FORMAL_OUTPUT_LAYOUT_ROOT.hex(),
        "gate_evidence_rows": evidence_rows,
        "intended_q1_gate_count": 14,
        "intended_q1_gate_mask": 0x003FFF,
        "intended_q1_state": "NOT_RUN",
        "network_accessed": False,
        "popen_count": 0,
        "prestart_admission_receipt_cbor_hex": receipt.canonical_bytes.hex(),
        "prestart_admission_receipt_root": "sha256:" + receipt.receipt_root.hex(),
        "published": False,
        "q1_execution_started": False,
        "q1_output_slots": [None] * 8,
        "q1_state_before_authoritative_apply": "NOT_RUN",
        "q1_gate_count_before_authoritative_apply": 0,
        "q1_gate_mask_before_authoritative_apply": 0,
        "run_genesis_cbor_hex": genesis.canonical_bytes.hex(),
        "run_genesis_root": "sha256:" + genesis.genesis_root.hex(),
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }


__all__ = [
    "ACTION_ID",
    "FORMAL_INPUT_KEY_REGISTRY_ABSENCE_ROOT",
    "Q1PrestartCandidateError",
    "SCHEMA_VERSION",
    "STATUS",
    "derive_q1_prestart_candidate_v1",
]
