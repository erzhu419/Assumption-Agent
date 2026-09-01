"""Pure state and completion replay for the Phase-3A-Q5a key ceremony.

Q5a creates three fresh, public-only purpose-5 signer manifests and an
*unpinned* trust-genesis/status candidate.  This module deliberately contains
no entropy source, private-key reader, signer, process launcher, or network
surface.  It replays the sealed Q4 completion, defines the one-shot state
objects, and permits only exact clean-prefix continuation before all three
key-generation workers have completed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Final, Mapping, NoReturn, Sequence

from . import phase3_q5a_purpose5_signer_genesis_v1 as _wire


CONTRACT_SCHEMA: Final = "hegel-phase3a-q5a-purpose5-signer-genesis-contract/1"
CONTRACT_ID: Final = "hegel-phase3a-q5a-purpose5-signer-genesis-actual-v1"
PREREGISTRATION_SCHEMA: Final = (
    "hegel-phase3a-q5a-purpose5-signer-genesis-preregistration/1"
)
PREREGISTRATION_ID: Final = "hegel-phase3a-q5a-purpose5-signer-genesis-v1"
ACTION_ID: Final = "Q5A_PURPOSE5_SIGNER_GENESIS_CANDIDATE_V1"

CAPSULE_SCHEMA: Final = "hegel-phase3a-q5a-purpose5-signer-genesis-capsule/1"
AUTHORIZATION_SCHEMA: Final = (
    "hegel-phase3a-q5a-purpose5-signer-genesis-authorization/1"
)
CONSUMPTION_SCHEMA: Final = (
    "hegel-phase3a-q5a-purpose5-signer-genesis-consumption/1"
)
PENDING_SCHEMA: Final = "hegel-phase3a-q5a-purpose5-signer-genesis-pending/1"
CHECKPOINT_SCHEMA: Final = (
    "hegel-phase3a-q5a-purpose5-signer-genesis-checkpoint/1"
)
KEYGEN_INTENT_SCHEMA: Final = (
    "hegel-phase3a-q5a-purpose5-keygen-intent/1"
)
PUBLIC_RECEIPT_SCHEMA: Final = (
    "hegel-phase3a-q5a-purpose5-public-receipt/1"
)
TERMINAL_SCHEMA: Final = "hegel-phase3a-q5a-purpose5-signer-genesis-terminal/1"

CAPSULE_READY: Final = "Q5A_PURPOSE5_SIGNER_GENESIS_CAPSULE_READY_NOT_AUTHORIZED"
AUTHORIZATION_RECORDED: Final = (
    "Q5A_PURPOSE5_SIGNER_GENESIS_AUTHORIZATION_RECORDED_NOT_CONSUMED"
)
AUTHORIZATION_CONSUMED: Final = (
    "Q5A_PURPOSE5_SIGNER_GENESIS_AUTHORIZATION_CONSUMED"
)
PENDING: Final = "Q5A_PURPOSE5_SIGNER_GENESIS_PENDING"
CLEAN_CHECKPOINT: Final = "Q5A_PURPOSE5_SIGNER_GENESIS_CLEAN_CHECKPOINT"
INVALIDATED_TERMINAL_NO_RESUME: Final = (
    "Q5A_PURPOSE5_SIGNER_GENESIS_INVALIDATED_TERMINAL_NO_RESUME"
)
PARTIAL_FINALIZATION_REQUIRED: Final = (
    "Q5A_PURPOSE5_SIGNER_GENESIS_PARTIAL_FINALIZATION_REQUIRED"
)
KEYGEN_INTENT_RECORDED: Final = "Q5A_PURPOSE5_KEYGEN_INTENT_RECORDED"
PUBLIC_RECEIPT_COMPLETE: Final = "Q5A_PURPOSE5_PUBLIC_RECEIPT_COMPLETE"
AUTHORIZED_SCOPE: Final = (
    "Q5A_EXACT_Q4_REPLAY_AND_THREE_FRESH_PURPOSE5_KEY_CANDIDATES_ONCE"
)
NOT_AUTHORIZED: Final = "Q5A_PURPOSE5_SIGNER_GENESIS_NOT_AUTHORIZED_NOT_CONSUMED"

DOMAINS: Final = {
    "source_closure": b"HEGEL/Q5A/SOURCE_CLOSURE/V1\x00",
    "capsule": b"HEGEL/Q5A/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q5A/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q5A/CONSUMPTION/V1\x00",
    "pending": b"HEGEL/Q5A/PENDING/V1\x00",
    "checkpoint": b"HEGEL/Q5A/CHECKPOINT/V1\x00",
    "intent": b"HEGEL/Q5A/KEYGEN_INTENT/V1\x00",
    "public_receipt": b"HEGEL/Q5A/PUBLIC_RECEIPT/V1\x00",
    "terminal": b"HEGEL/Q5A/TERMINAL/V1\x00",
}
Q4_DOMAINS: Final = {
    "capsule": b"HEGEL/Q4/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q4/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q4/CONSUMPTION/V1\x00",
    "terminal": b"HEGEL/Q4/TERMINAL/V1\x00",
}

Q5_SOURCE_PATHS: Final = (
    "src/hegel_machine/phase3_q5a_purpose5_signer_genesis_v1.py",
    "src/hegel_machine/phase3_q5a_completion_replay_v1.py",
    "src/hegel_machine/strict_cbor_v1.py",
    "tools/phase3_q5a_purpose5_keygen_worker_v1.py",
    "tools/phase3_q5a_purpose5_signer_genesis_candidate_v1.py",
)

REQUIRED_SIGNER_ORDINALS: Final = (1, 2, 3)
SIGNER_ROLE_NAMES: Final = (
    "FINAL_CERTIFICATE_CUSTODIAN",
    "FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER",
    "FINAL_CERTIFICATE_RUST_REPLAY_ATTESTER",
)
CLEAN_PENDING_PREFIXES: Final = ((), (1,), (1, 2))
POSTCONSUMPTION_PENDING_PREFIXES: Final = CLEAN_PENDING_PREFIXES + ((1, 2, 3),)
WORKER_RECEIPT_SCHEMA: Final = (
    "hegel-phase3-q5a-purpose5-keygen-public-receipt/1"
)
WORKER_SUCCESS_STATUS: Final = (
    "PURPOSE5_KEYGEN_COMPLETE_PRIVATE_KEY_RETAINED"
)
ED25519_SPKI_DER_PREFIX: Final = bytes.fromhex("302a300506032b6570032100")

EXECUTION_POLICY: Final = {
    "logical_campaign_limit": 1,
    "maximum_worker_popen_count": 3,
    "maximum_worker_openssl_popen_count": 6,
    "maximum_key_generation_count": 3,
    "maximum_public_key_derivation_count": 3,
    "maximum_public_receipt_count": 3,
    "maximum_actor_start_count": 0,
    "maximum_docker_invocation_count": 0,
    "maximum_measurement_count": 0,
    "maximum_target_open_count": 0,
    "maximum_target_truth_access_count": 0,
    "maximum_split_access_count": 0,
    "maximum_role_evaluation_count": 0,
    "maximum_q1_read_count": 0,
    "maximum_q2_read_count": 0,
    "maximum_q3_read_count": 0,
    "maximum_key_import_count": 0,
    "maximum_signature_count": 0,
    "maximum_certificate_envelope_publication_count": 0,
    "maximum_certificate_issuance_count": 0,
    "network_allowed": False,
    "preconsumption_pending_same_capsule_digest": True,
    "preconsumption_pending_consumes_authorization": False,
    "postconsumption_clean_prefix_pending_allowed": True,
    "postintent_unknown_terminal_no_resume": True,
    "partial_finalization_artifact_to_terminal_only": True,
    "partial_finalization_terminal_to_trajectory_suffix_only": True,
}

AUTHORITY_BOUNDARY: Final = {
    "q4_state": "Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_ADMITTED_NOT_ISSUED",
    "q5a_disposition": "GENESIS_CANDIDATE_ONLY_NOT_PINNED_NOT_SIGNED",
    "purpose5_candidate_signer_count": 3,
    "purpose5_active_signer_count": 0,
    "purpose5_trust_genesis_pinned": False,
    "purpose5_current_key_status_accepted": False,
    "formal_certificate_signed": False,
    "certificate_envelope_published": False,
    "outside_certificate_issued": False,
    "external_authority": False,
    "synthesis_allowed": False,
    "mdl_allowed": False,
    "external_scientific_claim_allowed": False,
    "adoption_allowed": False,
    "promotion_allowed": False,
    "active_transition_allowed": False,
    "production_allowed": False,
    "current_pointer_update_allowed": False,
}

_ZERO_RESTRICTED_ACTIONS: Final = {
    "actor_start_count": 0,
    "docker_invocation_count": 0,
    "measurement_count": 0,
    "target_open_count": 0,
    "target_truth_access_count": 0,
    "split_access_count": 0,
    "role_evaluation_count": 0,
    "q1_read_count": 0,
    "q2_read_count": 0,
    "q3_read_count": 0,
    "key_import_count": 0,
    "signature_count": 0,
    "certificate_envelope_publication_count": 0,
    "certificate_issuance_count": 0,
    "synthesis_action_count": 0,
    "mdl_action_count": 0,
    "network_accessed": False,
    "external_scientific_claim_made": False,
    "private_key_exported": False,
    "private_key_repository_write_performed": False,
    "trust_genesis_pinned": False,
    "current_key_status_accepted": False,
    "adoption_performed": False,
    "promotion_performed": False,
    "active_transition_performed": False,
    "production_accessed": False,
    "current_pointer_updated": False,
}

Q4_ZERO_ACTION_AUDIT: Final = {
    "internal_popen_count": 0,
    "actor_start_count": 0,
    "docker_invocation_count": 0,
    "measurement_count": 0,
    "target_open_count": 0,
    "target_truth_access_count": 0,
    "split_access_count": 0,
    "role_evaluation_count": 0,
    "q1_read_count": 0,
    "q2_read_count": 0,
    "archive_read_count": 0,
    "key_generation_count": 0,
    "key_import_count": 0,
    "key_installation_count": 0,
    "signature_count": 0,
    "certificate_issuance_count": 0,
    "synthesis_action_count": 0,
    "mdl_action_count": 0,
    "network_accessed": False,
    "external_scientific_claim_made": False,
    "adoption_performed": False,
    "promotion_performed": False,
    "active_transition_performed": False,
    "production_accessed": False,
    "current_pointer_updated": False,
}

MAX_JSON_BYTES: Final = 32 * 1024 * 1024
MAX_Q4_ARTIFACT_BYTES: Final = 1024 * 1024
MAX_TRAJECTORY_BYTES: Final = 1024 * 1024
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEX32 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class Q5aCompletionReplayError(ValueError):
    """A Q5a state object, Q4 predecessor, or public result differs."""


def _fail(detail: str) -> NoReturn:
    raise Q5aCompletionReplayError(detail)


@dataclass(frozen=True, slots=True)
class LoadedJson:
    path: Path
    raw: bytes
    raw_sha256: str
    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Q4SealedReplayV1:
    preflight_receipt: dict[str, Any]
    capability_receipt: Any


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON token {value!r}")


def _finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("non-finite JSON number")
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            _fail("JSON object keys must be strings")
        for child in value.values():
            _finite(child)
    elif type(value) in (list, tuple):
        for child in value:
            _finite(child)


def canonical_json_bytes(value: object) -> bytes:
    _finite(value)
    try:
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
    except (TypeError, ValueError, UnicodeError) as error:
        _fail(f"canonical JSON rejected: {error}")


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or len(payload) > MAX_JSON_BYTES:
        _fail(f"{label} exceeds its byte bound")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label} is not strict ASCII JSON: {error}")
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        _fail(f"{label} is not one canonical JSON object")
    return value


def pretty_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or len(payload) > MAX_JSON_BYTES:
        _fail(f"{label} exceeds its byte bound")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label} is not strict JSON: {error}")
    if type(value) is not dict:
        _fail(f"{label} must be one JSON object")
    return value


def raw_sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        _fail("raw SHA-256 input must be bytes")
    return "sha256:" + sha256(payload).hexdigest()


def _digest(domain: bytes, value: object) -> str:
    return "sha256:" + sha256(domain + canonical_json_bytes(value)).hexdigest()


def _tagged(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be one tagged SHA-256")
    return value


def _source_commit(value: object, label: str) -> str:
    if type(value) is not str or _SOURCE_COMMIT.fullmatch(value) is None:
        _fail(f"{label} must be one Git SHA-1 commit")
    return value


def load_pretty_json(path: Path, label: str) -> LoadedJson:
    try:
        raw = path.read_bytes()
    except OSError as error:
        _fail(f"{label} unavailable: {error}")
    return LoadedJson(path, raw, raw_sha256(raw), pretty_json_bytes(raw, label))


def load_preregistration(path: Path) -> LoadedJson:
    loaded = load_pretty_json(path, "Q5a preregistration")
    value = loaded.value
    if (
        value.get("schema_version") != PREREGISTRATION_SCHEMA
        or value.get("preregistration_id") != PREREGISTRATION_ID
        or value.get("status") != "Q5A_SPEC_FROZEN_NOT_AUTHORIZED"
        or type(value.get("predecessor_q4")) is not dict
    ):
        _fail("Q5a preregistration identity differs")
    return loaded


def load_contract(path: Path) -> LoadedJson:
    loaded = load_pretty_json(path, "Q5a contract")
    value = loaded.value
    if (
        value.get("schema_version") != CONTRACT_SCHEMA
        or value.get("contract_id") != CONTRACT_ID
        or value.get("action_id") != ACTION_ID
        or type(value.get("fixed_paths")) is not dict
        or type(value.get("q4_inputs")) is not dict
        or value.get("execution_policy") != EXECUTION_POLICY
        or value.get("authority_boundary") != AUTHORITY_BOUNDARY
    ):
        _fail("Q5a contract identity differs")
    prereg = load_preregistration(Path(value.get("preregistration_path", "")))
    if prereg.raw_sha256 != value.get("preregistration_raw_sha256"):
        _fail("Q5a preregistration raw identity differs")
    return loaded


def _completed_prefix(value: object, *, allow_complete: bool = True) -> tuple[int, ...]:
    if type(value) not in (list, tuple):
        _fail("Q5a completed prefix must be an array")
    prefix = tuple(value)
    allowed = CLEAN_PENDING_PREFIXES + (((1, 2, 3),) if allow_complete else ())
    if prefix not in allowed:
        _fail("Q5a completed prefix differs")
    return prefix


def q5a_action_audit(
    completed_prefix: Sequence[int],
    *,
    worker_popen_count: int | None = None,
    key_generation_count: int | None = None,
) -> dict[str, Any]:
    prefix = _completed_prefix(completed_prefix)
    completed = len(prefix)
    popens = completed if worker_popen_count is None else worker_popen_count
    generations = completed if key_generation_count is None else key_generation_count
    if (
        type(popens) is not int
        or type(generations) is not int
        or not (completed <= popens <= 3)
        or not (completed <= generations <= 3)
        or abs(popens - generations) > 0
        or popens > completed + 1
    ):
        _fail("Q5a worker/key-generation audit differs")
    result: dict[str, Any] = {
        "worker_popen_count": popens,
        "worker_openssl_popen_count": 2 * generations,
        "key_generation_count": generations,
        "public_key_derivation_count": generations,
        "public_receipt_count": completed,
        "candidate_signer_count": completed,
        "active_signer_count": 0,
    }
    result.update(deepcopy(_ZERO_RESTRICTED_ACTIONS))
    return result


ZERO_ACTION_AUDIT: Final = q5a_action_audit(())
SUCCESS_ACTION_AUDIT: Final = q5a_action_audit((1, 2, 3))


def _semantic_digest(
    value: Mapping[str, Any], kind: str, field: str
) -> str:
    unsigned = deepcopy(dict(value))
    supplied = unsigned.pop(field, None)
    expected = _digest(Q4_DOMAINS[kind], unsigned)
    if supplied != expected:
        _fail(f"Q4 {kind} semantic digest differs")
    return expected


def _expected_predecessor(preregistration: Mapping[str, Any]) -> Mapping[str, Any]:
    value = preregistration.get("predecessor_q4")
    if type(value) is not dict:
        _fail("Q5a predecessor Q4 registry differs")
    return value


def _expected_input(
    predecessor: Mapping[str, Any], kind: str
) -> Mapping[str, Any]:
    value = predecessor.get(kind)
    if type(value) is not dict:
        _fail(f"Q5a predecessor Q4 {kind} registry differs")
    _tagged(value.get("raw_sha256"), f"Q4 {kind} raw digest")
    return value


def _decode_q4_capability(payload: bytes) -> Any:
    decoder = getattr(_wire, "decode_sealed_q4_capability_receipt_v1", None)
    if decoder is None:
        decoder = getattr(_wire, "decode_q4_capability_receipt_v1", None)
    if not callable(decoder):
        _fail("Q5a core has no producer-free Q4 capability decoder")
    try:
        return decoder(payload)
    except (TypeError, ValueError) as error:
        _fail(f"Q4 capability receipt replay rejected: {error}")


def _q4_receipt_bindings(receipt: Any) -> tuple[str, str, str, str]:
    receipt_root = getattr(receipt, "receipt_root", None)
    body_hash = getattr(receipt, "body_hash", None)
    q4_commit = getattr(
        getattr(receipt, "q4_repository_commit_id", None), "raw_digest", None
    )
    q4_closure = getattr(receipt, "q4_source_closure_root", None)
    if (
        type(receipt_root) is not bytes
        or len(receipt_root) != 32
        or type(body_hash) is not bytes
        or len(body_hash) != 32
        or type(q4_commit) is not bytes
        or len(q4_commit) != 20
        or type(q4_closure) is not bytes
        or len(q4_closure) != 32
    ):
        _fail("Q4 capability receipt bindings differ")
    return (
        "sha256:" + receipt_root.hex(),
        "sha256:" + body_hash.hex(),
        q4_commit.hex(),
        "sha256:" + q4_closure.hex(),
    )


def _replay_trajectory(
    payload: bytes,
    *,
    capsule_digest: str,
    authorization_digest: str,
    capability_receipt_root: str,
) -> list[dict[str, Any]]:
    if type(payload) is not bytes or not payload or len(payload) > MAX_TRAJECTORY_BYTES:
        _fail("Q4 trajectory exceeds its byte bound")
    raw_lines = payload.splitlines(keepends=True)
    if len(raw_lines) != 3 or b"".join(raw_lines) != payload:
        _fail("Q4 trajectory event count differs")
    rows = [strict_json_bytes(line, f"Q4 trajectory event {index}") for index, line in enumerate(raw_lines, 1)]
    expected = (
        ("CAPSULE_PREPARED_NOT_AUTHORIZED", 0, 0),
        ("AUTHORIZATION_RECORDED_NOT_CONSUMED", 0, 0),
        ("Q4_CAPABILITY_ADMISSION_COMPLETE_NOT_ISSUED", 1, 1),
    )
    for row, (event, consumed, campaigns) in zip(rows, expected, strict=True):
        if (
            row.get("schema_version") != "hegel-phase3a-q4-trajectory/1"
            or row.get("event") != event
            or row.get("capsule_digest") != capsule_digest
            or row.get("authorization_consumption_count") != consumed
            or row.get("logical_campaign_count") != campaigns
            or row.get("q4_action_audit") != Q4_ZERO_ACTION_AUDIT
            or type(row.get("time_ns")) is not int
            or row["time_ns"] < 0
        ):
            _fail("Q4 trajectory event replay differs")
    if rows[1].get("authorization_digest") != authorization_digest:
        _fail("Q4 trajectory authorization binding differs")
    if rows[2].get("capability_receipt_root") != capability_receipt_root:
        _fail("Q4 trajectory capability binding differs")
    return deepcopy(rows)


def replay_q4_predecessor_v1(
    preregistration: Mapping[str, Any],
    *,
    capsule_payload: bytes,
    authorization_payload: bytes,
    consumption_payload: bytes,
    terminal_payload: bytes,
    artifact_payload: bytes,
    trajectory_payload: bytes,
) -> Q4SealedReplayV1:
    """Replay Q4's five authority objects plus its final three-event audit."""

    predecessor = _expected_predecessor(preregistration)
    payloads = {
        "capsule": capsule_payload,
        "authorization": authorization_payload,
        "consumption": consumption_payload,
        "terminal": terminal_payload,
        "artifact": artifact_payload,
    }
    authority_rows: list[dict[str, Any]] = []
    for kind, payload in payloads.items():
        expected = _expected_input(predecessor, kind)
        actual_raw = raw_sha256(payload)
        if actual_raw != expected.get("raw_sha256"):
            _fail(f"Q4 {kind} raw SHA-256 differs")
        authority_rows.append(
            {"kind": kind, "byte_count": len(payload), "raw_sha256": actual_raw}
        )
    trajectory_expected = _expected_input(predecessor, "trajectory")
    trajectory_raw = raw_sha256(trajectory_payload)
    if trajectory_raw != trajectory_expected.get("raw_sha256"):
        _fail("Q4 trajectory raw SHA-256 differs")

    capsule = strict_json_bytes(capsule_payload, "Q4 capsule")
    authorization = strict_json_bytes(authorization_payload, "Q4 authorization")
    consumption = strict_json_bytes(consumption_payload, "Q4 consumption")
    terminal = strict_json_bytes(terminal_payload, "Q4 terminal")
    capsule_digest = _semantic_digest(capsule, "capsule", "capsule_digest")
    authorization_digest = _semantic_digest(
        authorization, "authorization", "authorization_digest"
    )
    consumption_digest = _semantic_digest(consumption, "consumption", "record_digest")
    terminal_digest = _semantic_digest(terminal, "terminal", "record_digest")
    semantic = {
        "capsule": ("capsule_digest", capsule_digest),
        "authorization": ("authorization_digest", authorization_digest),
        "consumption": ("record_digest", consumption_digest),
        "terminal": ("record_digest", terminal_digest),
    }
    for kind, (field, actual) in semantic.items():
        if _expected_input(predecessor, kind).get(field) != actual:
            _fail(f"Q4 {kind} configured semantic digest differs")

    outcome = terminal.get("outcome")
    if (
        capsule.get("status") != "Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPSULE_READY_NOT_AUTHORIZED"
        or authorization.get("status") != "Q4_FORMAL_QUOTIENT_CERTIFICATE_AUTHORIZATION_RECORDED_NOT_CONSUMED"
        or consumption.get("status") != "Q4_FORMAL_QUOTIENT_CERTIFICATE_AUTHORIZATION_CONSUMED"
        or terminal.get("status") != "Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_ADMITTED_NOT_ISSUED"
        or type(outcome) is not dict
        or outcome.get("status") != terminal.get("status")
        or authorization.get("capsule_digest") != capsule_digest
        or consumption.get("capsule_digest") != capsule_digest
        or terminal.get("capsule_digest") != capsule_digest
        or consumption.get("authorization_digest") != authorization_digest
        or terminal.get("authorization_digest") != authorization_digest
        or terminal.get("consumption_record_digest") != consumption_digest
        or terminal.get("authorization_consumption_count") != 1
        or terminal.get("logical_campaign_count") != 1
        or capsule.get("source_commit") != predecessor.get("source_commit")
        or capsule.get("source_closure_root") != predecessor.get("source_closure_root")
        or outcome.get("artifact_raw_sha256") != raw_sha256(artifact_payload)
        or outcome.get("capability_disposition") != "ADMITTED_NOT_ISSUED"
        or outcome.get("q4_action_audit") != Q4_ZERO_ACTION_AUDIT
        or outcome.get("purpose5_key_installed") is not False
        or outcome.get("formal_certificate_signed") is not False
        or outcome.get("outside_certificate_issued") is not False
        or outcome.get("external_authority") is not False
        or outcome.get("synthesis_released") is not False
        or outcome.get("campaign_complete") is not True
        or outcome.get("terminal_no_resume") is not False
    ):
        _fail("Q4 predecessor completion boundary differs")

    receipt = _decode_q4_capability(artifact_payload)
    receipt_root, body_hash, q4_commit, q4_closure = _q4_receipt_bindings(receipt)
    artifact_expected = _expected_input(predecessor, "artifact")
    if (
        receipt_root != artifact_expected.get("capability_receipt_root")
        or body_hash != artifact_expected.get("quotient_body_hash")
        or outcome.get("capability_receipt_root") != receipt_root
        or q4_commit != predecessor.get("source_commit")
        or q4_closure != predecessor.get("source_closure_root")
    ):
        _fail("Q4 capability receipt binding differs")

    events = _replay_trajectory(
        trajectory_payload,
        capsule_digest=capsule_digest,
        authorization_digest=authorization_digest,
        capability_receipt_root=receipt_root,
    )
    if trajectory_expected.get("event_count") not in (None, 3):
        _fail("Q4 configured trajectory event count differs")

    preflight = {
        "schema_version": "hegel-phase3a-q5a-q4-predecessor-preflight/1",
        "selected_q4_authority_inputs": authority_rows,
        "selected_q4_authority_input_count": 5,
        "selected_q4_authority_input_total_bytes": sum(
            len(payload) for payload in payloads.values()
        ),
        "q4_trajectory_audit": {
            "raw_sha256": trajectory_raw,
            "byte_count": len(trajectory_payload),
            "event_count": 3,
            "events": [row["event"] for row in events],
            "all_action_audits_zero": True,
        },
        "q4_capsule_digest": capsule_digest,
        "q4_authorization_digest": authorization_digest,
        "q4_consumption_record_digest": consumption_digest,
        "q4_terminal_record_digest": terminal_digest,
        "q4_capability_artifact_raw_sha256": raw_sha256(artifact_payload),
        "q4_capability_receipt_root": receipt_root,
        "q4_quotient_body_hash": body_hash,
        "q4_source_commit": q4_commit,
        "q4_source_closure_root": q4_closure,
        "producer_free_read_only_replay_completed": True,
        "q4_trajectory_replay_completed": True,
        "q5a_action_audit": deepcopy(ZERO_ACTION_AUDIT),
    }
    return Q4SealedReplayV1(preflight, receipt)


def validate_source_closure_rows(rows: object) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != len(Q5_SOURCE_PATHS):
        _fail("Q5a source closure row count differs")
    checked: list[dict[str, Any]] = []
    for row, expected_path in zip(rows, Q5_SOURCE_PATHS, strict=True):
        if (
            type(row) is not dict
            or set(row) != {"relative_path", "byte_count", "raw_sha256"}
            or row.get("relative_path") != expected_path
            or type(row.get("byte_count")) is not int
            or row["byte_count"] < 1
        ):
            _fail("Q5a source closure row differs")
        _tagged(row.get("raw_sha256"), "Q5a source row digest")
        checked.append(deepcopy(row))
    return checked


def source_closure_root(rows: object) -> str:
    return _digest(DOMAINS["source_closure"], validate_source_closure_rows(rows))


def validate_q4_predecessor_receipt(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q5a preflight receipt differs")
    required = {
        "schema_version", "selected_q4_authority_inputs",
        "selected_q4_authority_input_count",
        "selected_q4_authority_input_total_bytes", "q4_trajectory_audit",
        "q4_capsule_digest", "q4_authorization_digest",
        "q4_consumption_record_digest", "q4_terminal_record_digest",
        "q4_capability_artifact_raw_sha256", "q4_capability_receipt_root",
        "q4_quotient_body_hash", "q4_source_commit",
        "q4_source_closure_root", "producer_free_read_only_replay_completed",
        "q4_trajectory_replay_completed", "q5a_action_audit",
    }
    trajectory = value.get("q4_trajectory_audit")
    rows = value.get("selected_q4_authority_inputs")
    if (
        set(value) != required
        or value.get("schema_version") != "hegel-phase3a-q5a-q4-predecessor-preflight/1"
        or value.get("selected_q4_authority_input_count") != 5
        or type(rows) is not list
        or [row.get("kind") for row in rows]
        != ["capsule", "authorization", "consumption", "terminal", "artifact"]
        or type(trajectory) is not dict
        or trajectory.get("event_count") != 3
        or trajectory.get("events")
        != [
            "CAPSULE_PREPARED_NOT_AUTHORIZED",
            "AUTHORIZATION_RECORDED_NOT_CONSUMED",
            "Q4_CAPABILITY_ADMISSION_COMPLETE_NOT_ISSUED",
        ]
        or trajectory.get("all_action_audits_zero") is not True
        or value.get("producer_free_read_only_replay_completed") is not True
        or value.get("q4_trajectory_replay_completed") is not True
        or value.get("q5a_action_audit") != ZERO_ACTION_AUDIT
    ):
        _fail("Q5a preflight receipt differs")
    for field in (
        "q4_capsule_digest", "q4_authorization_digest",
        "q4_consumption_record_digest", "q4_terminal_record_digest",
        "q4_capability_artifact_raw_sha256", "q4_capability_receipt_root",
        "q4_quotient_body_hash", "q4_source_closure_root",
    ):
        _tagged(value.get(field), field)
    _tagged(trajectory.get("raw_sha256"), "Q4 trajectory raw digest")
    _source_commit(value.get("q4_source_commit"), "Q4 source commit")
    return deepcopy(value)


def _validate_worker_startup_smoke_receipt(
    value: object, ordinal: int
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q5a worker startup-smoke receipt must be one mapping")
    expected_keys = {
        "expected_gid", "expected_uid", "key_generation_count",
        "openssl_path", "openssl_popen_count", "public_key_derivation_count",
        "purpose_id", "schema", "signature_count", "signer_ordinal",
        "signer_role", "status",
    }
    if (
        set(value) != expected_keys
        or ordinal not in REQUIRED_SIGNER_ORDINALS
        or value.get("schema")
        != "hegel-phase3-q5a-purpose5-keygen-startup-smoke-receipt/1"
        or value.get("status") != "Q5A_PURPOSE5_KEYGEN_STARTUP_SMOKE_PASS"
        or value.get("purpose_id") != 5
        or value.get("expected_uid") != 1000
        or value.get("expected_gid") != 1000
        or value.get("openssl_path") != "/usr/bin/openssl"
        or value.get("key_generation_count") != 0
        or value.get("openssl_popen_count") != 0
        or value.get("public_key_derivation_count") != 0
        or value.get("signature_count") != 0
        or value.get("signer_ordinal") != ordinal
        or value.get("signer_role") != SIGNER_ROLE_NAMES[ordinal - 1]
    ):
        _fail("Q5a worker startup-smoke receipt replay differs")
    return deepcopy(value)


def build_preflight_receipt(
    q4_predecessor_receipt: Mapping[str, Any],
    worker_startup_smoke_receipts: Sequence[Mapping[str, Any]],
    *,
    fresh_root_admission_completed: bool,
) -> dict[str, Any]:
    q4_receipt = validate_q4_predecessor_receipt(q4_predecessor_receipt)
    if (
        type(worker_startup_smoke_receipts) not in (list, tuple)
        or len(worker_startup_smoke_receipts) != 3
    ):
        _fail("Q5a requires exactly three worker startup-smoke receipts")
    smokes = [
        _validate_worker_startup_smoke_receipt(value, ordinal)
        for ordinal, value in enumerate(worker_startup_smoke_receipts, 1)
    ]
    if fresh_root_admission_completed is not True:
        _fail("Q5a fresh-root admission is incomplete")
    return {
        "schema_version": "hegel-phase3a-q5a-preflight/1",
        "q4_predecessor_receipt": q4_receipt,
        "worker_startup_smoke_receipts": smokes,
        "worker_startup_smoke_popen_count": 3,
        "preconsumption_key_generation_count": 0,
        "preconsumption_worker_openssl_popen_count": 0,
        "preconsumption_public_key_derivation_count": 0,
        "preconsumption_signature_count": 0,
        "fresh_root_admission_completed": True,
        "q5a_action_audit": deepcopy(ZERO_ACTION_AUDIT),
    }


def validate_preflight_receipt(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q5a capsule preflight receipt differs")
    required = {
        "schema_version", "q4_predecessor_receipt",
        "worker_startup_smoke_receipts", "worker_startup_smoke_popen_count",
        "preconsumption_key_generation_count",
        "preconsumption_worker_openssl_popen_count",
        "preconsumption_public_key_derivation_count",
        "preconsumption_signature_count", "fresh_root_admission_completed",
        "q5a_action_audit",
    }
    if set(value) != required:
        _fail("Q5a capsule preflight receipt keys differ")
    expected = build_preflight_receipt(
        value.get("q4_predecessor_receipt"),
        value.get("worker_startup_smoke_receipts"),
        fresh_root_admission_completed=value.get("fresh_root_admission_completed"),
    )
    if value != expected:
        _fail("Q5a capsule preflight receipt replay differs")
    return deepcopy(expected)


def _validate_preflight_predecessor_binding(
    preflight: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> None:
    receipt = preflight["q4_predecessor_receipt"]
    predecessor = _expected_predecessor(preregistration)
    expected_rows = [
        (kind, _expected_input(predecessor, kind).get("raw_sha256"))
        for kind in ("capsule", "authorization", "consumption", "terminal", "artifact")
    ]
    actual_rows = [
        (row.get("kind"), row.get("raw_sha256"))
        for row in receipt["selected_q4_authority_inputs"]
    ]
    expected = {
        "q4_capsule_digest": _expected_input(predecessor, "capsule").get(
            "capsule_digest"
        ),
        "q4_authorization_digest": _expected_input(
            predecessor, "authorization"
        ).get("authorization_digest"),
        "q4_consumption_record_digest": _expected_input(
            predecessor, "consumption"
        ).get("record_digest"),
        "q4_terminal_record_digest": _expected_input(
            predecessor, "terminal"
        ).get("record_digest"),
        "q4_capability_artifact_raw_sha256": _expected_input(
            predecessor, "artifact"
        ).get("raw_sha256"),
        "q4_capability_receipt_root": _expected_input(
            predecessor, "artifact"
        ).get("capability_receipt_root"),
        "q4_quotient_body_hash": _expected_input(
            predecessor, "artifact"
        ).get("quotient_body_hash"),
        "q4_source_commit": predecessor.get("source_commit"),
        "q4_source_closure_root": predecessor.get("source_closure_root"),
    }
    if (
        actual_rows != expected_rows
        or any(receipt.get(field) != value for field, value in expected.items())
        or receipt["q4_trajectory_audit"].get("raw_sha256")
        != _expected_input(predecessor, "trajectory").get("raw_sha256")
        or receipt["q4_trajectory_audit"].get("event_count")
        != _expected_input(predecessor, "trajectory").get("event_count")
        or predecessor.get("authority_input_total_bytes") not in (
            None,
            receipt["selected_q4_authority_input_total_bytes"],
        )
        or predecessor.get("audit_input_total_bytes") not in (
            None,
            receipt["q4_trajectory_audit"].get("byte_count"),
        )
    ):
        _fail("Q5a capsule predecessor binding differs from preregistration")


def build_capsule(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    rows = validate_source_closure_rows(inputs.get("source_closure_rows"))
    closure = source_closure_root(rows)
    if inputs.get("source_closure_root") != closure:
        _fail("Q5a source closure root differs")
    commit = _source_commit(inputs.get("source_commit"), "Q5a source commit")
    run_id = inputs.get("run_id")
    created = inputs.get("created_at_unix_seconds")
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        _fail("Q5a run id differs")
    if type(created) is not int or created < 0:
        _fail("Q5a creation time differs")
    preflight = validate_preflight_receipt(inputs.get("preflight_receipt"))
    _validate_preflight_predecessor_binding(preflight, preregistration)
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA,
        "status": CAPSULE_READY,
        "contract_raw_sha256": contract.raw_sha256,
        "preregistration_raw_sha256": contract.value["preregistration_raw_sha256"],
        "action_id": ACTION_ID,
        "source_commit": commit,
        "source_closure_rows": rows,
        "source_closure_root": closure,
        "run_id": run_id,
        "created_at_unix_seconds": created,
        "preflight_receipt": preflight,
        "execution_policy": deepcopy(EXECUTION_POLICY),
        "authority_boundary": deepcopy(AUTHORITY_BOUNDARY),
        "capsule_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("capsule_digest")
    body["capsule_digest"] = _digest(DOMAINS["capsule"], unsigned)
    return body


def validate_capsule(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q5a capsule differs")
    inputs = {
        key: value.get(key)
        for key in (
            "source_commit", "source_closure_rows", "source_closure_root",
            "run_id", "created_at_unix_seconds", "preflight_receipt",
        )
    }
    expected = build_capsule(contract, preregistration, inputs)
    if value != expected:
        _fail("Q5a capsule replay differs")
    return deepcopy(expected)


def build_authorization(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_capsule(capsule, contract, preregistration)
    body: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": AUTHORIZATION_RECORDED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked["capsule_digest"],
        "authorized_scope": AUTHORIZED_SCOPE,
        "logical_campaign_limit": 1,
        "worker_popen_limit": 3,
        "worker_openssl_popen_limit": 6,
        "key_generation_limit": 3,
        "public_key_derivation_limit": 3,
        "public_receipt_limit": 3,
        "trust_genesis_pin_allowed": False,
        "current_key_status_acceptance_allowed": False,
        "certificate_signing_allowed": False,
        "certificate_envelope_publication_allowed": False,
        "certificate_issuance_allowed": False,
        "synthesis_allowed": False,
        "network_allowed": False,
        "authorization_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("authorization_digest")
    body["authorization_digest"] = _digest(DOMAINS["authorization"], unsigned)
    return body


def validate_authorization(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_authorization(contract, preregistration, capsule)
    if value != expected:
        _fail("Q5a authorization replay differs")
    return deepcopy(expected)


def build_consumption(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    body: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA,
        "status": AUTHORIZATION_CONSUMED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "completed_prefix": [],
        "q5a_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "trust_genesis_pinned": False,
        "current_key_status_accepted": False,
        "formal_certificate_signed": False,
        "certificate_envelope_published": False,
        "outside_certificate_issued": False,
        "synthesis_released": False,
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["consumption"], unsigned)
    return body


def validate_consumption(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_consumption(
        contract, preregistration, capsule, authorization
    )
    if value != expected:
        _fail("Q5a consumption replay differs")
    return deepcopy(expected)


def _receipt_digest_row(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "signer_ordinal": receipt["signer_ordinal"],
        "public_receipt_digest": receipt["public_receipt_digest"],
        "key_id_hex": receipt["key_id_hex"],
        "public_key_hex": receipt["public_key_hex"],
    }


def _validate_worker_public_receipt(value: object, ordinal: int) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q5a worker public receipt must be one mapping")
    expected_keys = {
        "key_epoch", "key_generation_count", "key_id_hex",
        "openssl_popen_count", "private_key_exported",
        "public_key_derivation_count", "public_key_hex", "purpose_id",
        "schema", "signature_count", "signer_ordinal", "signer_role",
        "status",
    }
    public_hex = value.get("public_key_hex")
    if type(public_hex) is not str or _HEX32.fullmatch(public_hex) is None:
        _fail("Q5a worker public key differs")
    public_key = bytes.fromhex(public_hex)
    if (
        set(value) != expected_keys
        or ordinal not in REQUIRED_SIGNER_ORDINALS
        or value.get("schema") != WORKER_RECEIPT_SCHEMA
        or value.get("status") != WORKER_SUCCESS_STATUS
        or value.get("purpose_id") != 5
        or value.get("key_epoch") != 1
        or value.get("key_generation_count") != 1
        or value.get("openssl_popen_count") != 2
        or value.get("public_key_derivation_count") != 1
        or value.get("private_key_exported") is not False
        or value.get("signature_count") != 0
        or value.get("signer_ordinal") != ordinal
        or value.get("signer_role") != SIGNER_ROLE_NAMES[ordinal - 1]
        or value.get("key_id_hex") != sha256(public_key).digest()[:16].hex()
    ):
        _fail("Q5a worker public receipt replay differs")
    return deepcopy(value)


def _validate_public_receipt_shape(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q5a public receipt differs")
    expected_keys = {
        "schema_version", "status", "contract_raw_sha256", "capsule_digest",
        "authorization_digest", "consumption_record_digest", "signer_ordinal",
        "completed_prefix_before", "keygen_intent_digest", "signer_role",
        "key_id_hex", "public_key_hex", "worker_public_receipt",
        "worker_public_receipt_raw_sha256", "public_key_der_raw_sha256",
        "private_key_persisted", "private_key_exported", "candidate_only",
        "q5a_action_audit", "public_receipt_digest",
    }
    if set(value) != expected_keys:
        _fail("Q5a public receipt keys differ")
    unsigned = deepcopy(value)
    supplied = unsigned.pop("public_receipt_digest")
    ordinal = value.get("signer_ordinal")
    worker = _validate_worker_public_receipt(
        value.get("worker_public_receipt"), ordinal if type(ordinal) is int else 0
    )
    public_key = bytes.fromhex(worker["public_key_hex"])
    if (
        value.get("schema_version") != PUBLIC_RECEIPT_SCHEMA
        or value.get("status") != PUBLIC_RECEIPT_COMPLETE
        or type(value.get("signer_ordinal")) is not int
        or value["signer_ordinal"] not in REQUIRED_SIGNER_ORDINALS
        or _HEX32.fullmatch(value.get("public_key_hex", "")) is None
        or value.get("signer_role") != worker["signer_role"]
        or value.get("key_id_hex") != worker["key_id_hex"]
        or value.get("public_key_hex") != worker["public_key_hex"]
        or value.get("worker_public_receipt_raw_sha256")
        != raw_sha256(canonical_json_bytes(worker))
        or value.get("public_key_der_raw_sha256")
        != raw_sha256(ED25519_SPKI_DER_PREFIX + public_key)
        or value.get("private_key_persisted") is not True
        or value.get("private_key_exported") is not False
        or value.get("candidate_only") is not True
        or supplied != _digest(DOMAINS["public_receipt"], unsigned)
    ):
        _fail("Q5a public receipt replay differs")
    _tagged(value.get("contract_raw_sha256"), "Q5a receipt contract digest")
    _tagged(value.get("capsule_digest"), "Q5a receipt capsule digest")
    _tagged(value.get("authorization_digest"), "Q5a receipt authorization digest")
    _tagged(value.get("consumption_record_digest"), "Q5a receipt consumption digest")
    _tagged(value.get("keygen_intent_digest"), "Q5a keygen intent digest")
    _tagged(
        value.get("worker_public_receipt_raw_sha256"),
        "Q5a worker receipt raw digest",
    )
    _tagged(value.get("public_key_der_raw_sha256"), "Q5a public DER raw digest")
    return deepcopy(value)


def validate_public_receipt_sequence(
    values: Sequence[Mapping[str, Any]],
    contract: LoadedJson,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if type(values) not in (list, tuple) or len(values) > 3:
        _fail("Q5a public receipt sequence differs")
    checked: list[dict[str, Any]] = []
    public_keys: set[str] = set()
    for index, raw in enumerate(values, 1):
        receipt = _validate_public_receipt_shape(raw)
        prefix = tuple(range(1, index))
        if (
            receipt["contract_raw_sha256"] != contract.raw_sha256
            or receipt["capsule_digest"] != capsule["capsule_digest"]
            or receipt["authorization_digest"] != authorization["authorization_digest"]
            or receipt["consumption_record_digest"] != consumption["record_digest"]
            or receipt["signer_ordinal"] != index
            or tuple(receipt["completed_prefix_before"]) != prefix
            or receipt["q5a_action_audit"] != q5a_action_audit(tuple(range(1, index + 1)))
            or receipt["public_key_hex"] in public_keys
        ):
            _fail("Q5a public receipt sequence replay differs")
        public_keys.add(receipt["public_key_hex"])
        checked.append(receipt)
    return checked


def build_keygen_intent(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
    *,
    created_at_unix_seconds: int,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        consumption, contract, preregistration, checked_capsule,
        checked_authorization,
    )
    receipts = validate_public_receipt_sequence(
        completed_receipts, contract, checked_capsule,
        checked_authorization, checked_consumption,
    )
    prefix = tuple(range(1, len(receipts) + 1))
    if prefix not in CLEAN_PENDING_PREFIXES:
        _fail("Q5a cannot create an intent after all signers completed")
    if type(created_at_unix_seconds) is not int or created_at_unix_seconds < 0:
        _fail("Q5a keygen intent time differs")
    ordinal = len(receipts) + 1
    body: dict[str, Any] = {
        "schema_version": KEYGEN_INTENT_SCHEMA,
        "status": KEYGEN_INTENT_RECORDED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "consumption_record_digest": checked_consumption["record_digest"],
        "signer_ordinal": ordinal,
        "completed_prefix": list(prefix),
        "completed_public_receipts": [_receipt_digest_row(row) for row in receipts],
        "worker_invocation_ordinal": ordinal,
        "created_at_unix_seconds": created_at_unix_seconds,
        "q5a_action_audit_before_worker": q5a_action_audit(prefix),
        "keygen_intent_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("keygen_intent_digest")
    body["keygen_intent_digest"] = _digest(DOMAINS["intent"], unsigned)
    return body


def validate_keygen_intent(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q5a keygen intent differs")
    expected = build_keygen_intent(
        contract, preregistration, capsule, authorization, consumption,
        completed_receipts,
        created_at_unix_seconds=value.get("created_at_unix_seconds"),
    )
    if value != expected:
        _fail("Q5a keygen intent replay differs")
    return deepcopy(expected)


def build_public_receipt(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
    intent: Mapping[str, Any],
    *,
    worker_receipt: Mapping[str, Any],
    worker_public_receipt_raw_sha256: str,
    public_key_der_raw_sha256: str,
) -> dict[str, Any]:
    checked_intent = validate_keygen_intent(
        intent, contract, preregistration, capsule, authorization, consumption,
        completed_receipts,
    )
    ordinal = checked_intent["signer_ordinal"]
    worker = _validate_worker_public_receipt(worker_receipt, ordinal)
    public_key = bytes.fromhex(worker["public_key_hex"])
    expected_worker_raw = raw_sha256(canonical_json_bytes(worker))
    expected_der_raw = raw_sha256(ED25519_SPKI_DER_PREFIX + public_key)
    if (
        worker_public_receipt_raw_sha256 != expected_worker_raw
        or public_key_der_raw_sha256 != expected_der_raw
    ):
        _fail("Q5a worker receipt/public DER raw identity differs")
    prior = validate_public_receipt_sequence(
        completed_receipts, contract, capsule, authorization, consumption
    )
    if public_key.hex() in {row["public_key_hex"] for row in prior}:
        _fail("Q5a public keys must be pairwise distinct")
    prefix = tuple(checked_intent["completed_prefix"])
    body: dict[str, Any] = {
        "schema_version": PUBLIC_RECEIPT_SCHEMA,
        "status": PUBLIC_RECEIPT_COMPLETE,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "signer_ordinal": ordinal,
        "completed_prefix_before": list(prefix),
        "keygen_intent_digest": checked_intent["keygen_intent_digest"],
        "signer_role": worker["signer_role"],
        "key_id_hex": worker["key_id_hex"],
        "public_key_hex": public_key.hex(),
        "worker_public_receipt": worker,
        "worker_public_receipt_raw_sha256": expected_worker_raw,
        "public_key_der_raw_sha256": expected_der_raw,
        "private_key_persisted": True,
        "private_key_exported": False,
        "candidate_only": True,
        "q5a_action_audit": q5a_action_audit(tuple(range(1, ordinal + 1))),
        "public_receipt_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("public_receipt_digest")
    body["public_receipt_digest"] = _digest(DOMAINS["public_receipt"], unsigned)
    return body


def build_pending(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    phase: str,
    condition: str,
    detail: str,
    authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    completed_receipts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    if any(type(item) is not str or not item for item in (phase, condition, detail)):
        _fail("Q5a pending fields differ")
    if consumption is None:
        if completed_receipts or authorization is not None and type(authorization) is not dict:
            _fail("Q5a preconsumption pending state differs")
        authorization_digest = None
        if authorization is not None:
            authorization_digest = validate_authorization(
                authorization, contract, preregistration, checked_capsule
            )["authorization_digest"]
        consumption_digest = None
        prefix: tuple[int, ...] = ()
        receipts: list[dict[str, Any]] = []
        kind = "PRECONSUMPTION"
        consumed = campaigns = 0
    else:
        if authorization is None:
            _fail("Q5a postconsumption pending requires authorization")
        checked_authorization = validate_authorization(
            authorization, contract, preregistration, checked_capsule
        )
        checked_consumption = validate_consumption(
            consumption, contract, preregistration, checked_capsule,
            checked_authorization,
        )
        receipts = validate_public_receipt_sequence(
            completed_receipts, contract, checked_capsule,
            checked_authorization, checked_consumption,
        )
        prefix = tuple(range(1, len(receipts) + 1))
        if prefix not in POSTCONSUMPTION_PENDING_PREFIXES:
            _fail("Q5a PENDING is allowed only at a clean public-result prefix")
        authorization_digest = checked_authorization["authorization_digest"]
        consumption_digest = checked_consumption["record_digest"]
        kind = "POSTCONSUMPTION_CLEAN_CHECKPOINT"
        consumed = campaigns = 1
    body: dict[str, Any] = {
        "schema_version": PENDING_SCHEMA,
        "status": PENDING,
        "pending_kind": kind,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": authorization_digest,
        "consumption_record_digest": consumption_digest,
        "phase": phase,
        "condition": condition,
        "detail": detail,
        "resume_capsule_digest": checked_capsule["capsule_digest"],
        "completed_prefix": list(prefix),
        "completed_public_receipts": [_receipt_digest_row(row) for row in receipts],
        "authorization_consumption_count": consumed,
        "logical_campaign_count": campaigns,
        "q5a_action_audit": q5a_action_audit(prefix),
        "pending_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("pending_digest")
    body["pending_digest"] = _digest(DOMAINS["pending"], unsigned)
    return body


def validate_pending(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    completed_receipts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q5a pending record differs")
    expected = build_pending(
        contract,
        preregistration,
        capsule,
        phase=value.get("phase"),
        condition=value.get("condition"),
        detail=value.get("detail"),
        authorization=authorization,
        consumption=consumption,
        completed_receipts=completed_receipts,
    )
    if value != expected:
        _fail("Q5a pending replay differs")
    return deepcopy(expected)


def build_checkpoint(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        consumption,
        contract,
        preregistration,
        checked_capsule,
        checked_authorization,
    )
    receipts = validate_public_receipt_sequence(
        completed_receipts,
        contract,
        checked_capsule,
        checked_authorization,
        checked_consumption,
    )
    prefix = tuple(range(1, len(receipts) + 1))
    body: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": CLEAN_CHECKPOINT,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "consumption_record_digest": checked_consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "completed_prefix": list(prefix),
        "completed_public_receipts": [
            _receipt_digest_row(row) for row in receipts
        ],
        "q5a_action_audit": q5a_action_audit(prefix),
        "trust_genesis_pinned": False,
        "current_key_status_accepted": False,
        "formal_certificate_signed": False,
        "certificate_envelope_published": False,
        "outside_certificate_issued": False,
        "checkpoint_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("checkpoint_digest")
    body["checkpoint_digest"] = _digest(DOMAINS["checkpoint"], unsigned)
    return body


def validate_checkpoint(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q5a checkpoint differs")
    expected = build_checkpoint(
        contract,
        preregistration,
        capsule,
        authorization,
        consumption,
        completed_receipts,
    )
    if value != expected:
        _fail("Q5a checkpoint replay differs")
    return deepcopy(expected)


def build_candidate_from_receipts(
    capsule: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
) -> Any:
    if len(completed_receipts) != 3:
        _fail("Q5a candidate requires three public receipts")
    capsule_preflight = validate_preflight_receipt(capsule.get("preflight_receipt"))
    preflight = capsule_preflight["q4_predecessor_receipt"]
    public_keys = tuple(bytes.fromhex(row["public_key_hex"]) for row in completed_receipts)
    try:
        return _wire.build_q5a_candidate_v1(
            q4_capsule_digest=bytes.fromhex(preflight["q4_capsule_digest"][7:]),
            q4_authorization_digest=bytes.fromhex(preflight["q4_authorization_digest"][7:]),
            q4_consumption_record_digest=bytes.fromhex(preflight["q4_consumption_record_digest"][7:]),
            q4_terminal_record_digest=bytes.fromhex(preflight["q4_terminal_record_digest"][7:]),
            q4_capability_receipt_root=bytes.fromhex(preflight["q4_capability_receipt_root"][7:]),
            q4_body_hash=bytes.fromhex(preflight["q4_quotient_body_hash"][7:]),
            q4_source_commit=preflight["q4_source_commit"],
            q4_source_closure_root=bytes.fromhex(preflight["q4_source_closure_root"][7:]),
            q5_source_commit=capsule["source_commit"],
            q5_source_closure_root=bytes.fromhex(capsule["source_closure_root"][7:]),
            public_keys=public_keys,
        )
    except (TypeError, ValueError) as error:
        _fail(f"Q5a candidate construction rejected: {error}")


def validate_artifact_binding(
    payload: bytes,
    capsule: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
) -> Any:
    if type(payload) is not bytes:
        _fail("Q5a candidate artifact must be bytes")
    try:
        candidate = _wire.decode_q5a_candidate_v1(payload)
    except (TypeError, ValueError) as error:
        _fail(f"Q5a candidate replay rejected: {error}")
    expected = build_candidate_from_receipts(capsule, completed_receipts)
    if (
        getattr(candidate, "canonical_bytes", None) != payload
        or candidate != expected
        or getattr(candidate, "candidate_root", None)
        != getattr(expected, "candidate_root", None)
    ):
        _fail("Q5a candidate artifact binding differs")
    return candidate


def build_complete_terminal(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
    artifact_payload: bytes,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        consumption, contract, preregistration, checked_capsule,
        checked_authorization,
    )
    receipts = validate_public_receipt_sequence(
        completed_receipts, contract, checked_capsule,
        checked_authorization, checked_consumption,
    )
    if len(receipts) != 3:
        _fail("Q5a complete terminal requires three public receipts")
    candidate = validate_artifact_binding(artifact_payload, checked_capsule, receipts)
    candidate_root = getattr(candidate, "candidate_root", None)
    if type(candidate_root) is not bytes or len(candidate_root) != 32:
        _fail("Q5a candidate root differs")
    outcome = {
        "status": _wire.Q5A_SUCCESS_STATUS,
        "disposition": "UNPINNED_NOT_SIGNED",
        "capsule_digest": checked_capsule["capsule_digest"],
        "completed_prefix": [1, 2, 3],
        "completed_public_receipts": [_receipt_digest_row(row) for row in receipts],
        "artifact_raw_sha256": raw_sha256(artifact_payload),
        "candidate_root": "sha256:" + candidate_root.hex(),
        "q4_capability_receipt_root": checked_capsule["preflight_receipt"]["q4_predecessor_receipt"]["q4_capability_receipt_root"],
        "q4_quotient_body_hash": checked_capsule["preflight_receipt"]["q4_predecessor_receipt"]["q4_quotient_body_hash"],
        "q5a_action_audit": deepcopy(SUCCESS_ACTION_AUDIT),
        "purpose5_candidate_signer_count": 3,
        "purpose5_active_signer_count": 0,
        "purpose5_trust_genesis_pinned": False,
        "purpose5_current_key_status_accepted": False,
        "formal_certificate_signed": False,
        "certificate_envelope_published": False,
        "outside_certificate_issued": False,
        "external_authority": False,
        "synthesis_released": False,
        "terminal_no_resume": False,
        "campaign_complete": True,
    }
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": _wire.Q5A_SUCCESS_STATUS,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "consumption_record_digest": checked_consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "outcome": outcome,
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["terminal"], unsigned)
    return body


def build_invalidated_terminal(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
    *,
    failure_phase: str,
    failure_code: str,
    failure_detail: str,
    unknown_intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        consumption, contract, preregistration, checked_capsule,
        checked_authorization,
    )
    receipts = validate_public_receipt_sequence(
        completed_receipts, contract, checked_capsule,
        checked_authorization, checked_consumption,
    )
    if any(type(item) is not str or not item for item in (
        failure_phase, failure_code, failure_detail
    )):
        _fail("Q5a invalidated failure fields differ")
    prefix = tuple(range(1, len(receipts) + 1))
    intent_digest = None
    popens = generations = len(prefix)
    if unknown_intent is not None:
        checked_intent = validate_keygen_intent(
            unknown_intent, contract, preregistration, checked_capsule,
            checked_authorization, checked_consumption, receipts,
        )
        intent_digest = checked_intent["keygen_intent_digest"]
        popens = generations = len(prefix) + 1
    audit = q5a_action_audit(
        prefix, worker_popen_count=popens, key_generation_count=generations
    )
    outcome = {
        "status": INVALIDATED_TERMINAL_NO_RESUME,
        "capsule_digest": checked_capsule["capsule_digest"],
        "completed_prefix": list(prefix),
        "completed_public_receipts": [_receipt_digest_row(row) for row in receipts],
        "unknown_intent_digest": intent_digest,
        "artifact_raw_sha256": None,
        "candidate_root": None,
        "q5a_action_audit": audit,
        "purpose5_candidate_signer_count": len(prefix),
        "purpose5_active_signer_count": 0,
        "purpose5_trust_genesis_pinned": False,
        "purpose5_current_key_status_accepted": False,
        "formal_certificate_signed": False,
        "certificate_envelope_published": False,
        "outside_certificate_issued": False,
        "external_authority": False,
        "synthesis_released": False,
        "failure_phase": failure_phase,
        "failure_code": failure_code,
        "failure_detail": failure_detail,
        "terminal_no_resume": True,
        "campaign_complete": False,
    }
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": INVALIDATED_TERMINAL_NO_RESUME,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "consumption_record_digest": checked_consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "outcome": outcome,
        "record_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("record_digest")
    body["record_digest"] = _digest(DOMAINS["terminal"], unsigned)
    return body


def validate_terminal(
    value: object,
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    completed_receipts: Sequence[Mapping[str, Any]],
    *,
    artifact_payload: bytes | None = None,
    unknown_intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("outcome")) is not dict:
        _fail("Q5a terminal must contain an outcome")
    if value.get("status") == _wire.Q5A_SUCCESS_STATUS:
        if artifact_payload is None or unknown_intent is not None:
            _fail("Q5a complete terminal requires the sealed candidate")
        expected = build_complete_terminal(
            contract, preregistration, capsule, authorization, consumption,
            completed_receipts, artifact_payload,
        )
    elif value.get("status") == INVALIDATED_TERMINAL_NO_RESUME:
        if artifact_payload is not None:
            _fail("Q5a invalidated terminal forbids a candidate artifact")
        outcome = value["outcome"]
        unknown = outcome.get("unknown_intent_digest")
        if (unknown is None) != (unknown_intent is None):
            _fail("Q5a invalidated terminal intent replay differs")
        expected = build_invalidated_terminal(
            contract, preregistration, capsule, authorization, consumption,
            completed_receipts,
            failure_phase=outcome.get("failure_phase"),
            failure_code=outcome.get("failure_code"),
            failure_detail=outcome.get("failure_detail"),
            unknown_intent=unknown_intent,
        )
    else:
        _fail("Q5a terminal status differs")
    if value != expected:
        _fail("Q5a terminal replay differs")
    return deepcopy(expected)


__all__ = [
    "ACTION_ID", "AUTHORIZATION_CONSUMED", "AUTHORIZATION_RECORDED",
    "AUTHORIZATION_SCHEMA", "AUTHORIZED_SCOPE", "AUTHORITY_BOUNDARY",
    "CAPSULE_READY", "CAPSULE_SCHEMA", "CHECKPOINT_SCHEMA",
    "CLEAN_CHECKPOINT", "CLEAN_PENDING_PREFIXES",
    "CONSUMPTION_SCHEMA", "CONTRACT_ID", "CONTRACT_SCHEMA", "DOMAINS",
    "EXECUTION_POLICY", "INVALIDATED_TERMINAL_NO_RESUME",
    "KEYGEN_INTENT_RECORDED", "KEYGEN_INTENT_SCHEMA", "LoadedJson",
    "NOT_AUTHORIZED", "PARTIAL_FINALIZATION_REQUIRED", "PENDING",
    "PENDING_SCHEMA", "POSTCONSUMPTION_PENDING_PREFIXES",
    "PREREGISTRATION_ID", "PREREGISTRATION_SCHEMA",
    "PUBLIC_RECEIPT_COMPLETE", "PUBLIC_RECEIPT_SCHEMA", "Q4SealedReplayV1",
    "Q4_ZERO_ACTION_AUDIT", "Q5_SOURCE_PATHS", "Q5aCompletionReplayError",
    "REQUIRED_SIGNER_ORDINALS", "SUCCESS_ACTION_AUDIT", "TERMINAL_SCHEMA",
    "ZERO_ACTION_AUDIT", "build_authorization", "build_candidate_from_receipts",
    "build_capsule", "build_checkpoint", "build_complete_terminal", "build_consumption",
    "build_invalidated_terminal", "build_keygen_intent", "build_pending",
    "build_public_receipt", "canonical_json_bytes", "load_contract",
    "load_preregistration", "pretty_json_bytes", "q5a_action_audit",
    "raw_sha256", "replay_q4_predecessor_v1", "source_closure_root",
    "strict_json_bytes", "validate_artifact_binding", "validate_authorization",
    "validate_capsule", "validate_checkpoint", "validate_consumption",
    "validate_keygen_intent", "validate_pending", "validate_preflight_receipt",
    "validate_public_receipt_sequence",
    "validate_source_closure_rows", "validate_terminal",
]
