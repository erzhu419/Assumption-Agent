"""Pure state and completion replay for Phase-3A-Q4a capability admission.

The runtime consumes no target, archive, key, signature, or certificate-
issuance capability.  It reads exactly the five sealed Q3 predecessor objects
and can publish only an unsigned ``ADMITTED_NOT_ISSUED`` capability receipt.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Final, Mapping, NoReturn

from . import phase3_q4_formal_quotient_certificate_capability_v1 as _wire


CONTRACT_SCHEMA: Final = (
    "hegel-phase3a-q4-formal-quotient-certificate-capability-contract/1"
)
CONTRACT_ID: Final = (
    "hegel-phase3a-q4-formal-quotient-certificate-capability-actual-v1"
)
ACTION_ID: Final = "Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_ADMISSION_V1"
CAPSULE_INPUTS_SCHEMA: Final = (
    "hegel-phase3a-q4-formal-quotient-certificate-capability-capsule-inputs/1"
)
CAPSULE_SCHEMA: Final = (
    "hegel-phase3a-q4-formal-quotient-certificate-capability-capsule/1"
)
AUTHORIZATION_SCHEMA: Final = (
    "hegel-phase3a-q4-formal-quotient-certificate-capability-authorization/1"
)
CONSUMPTION_SCHEMA: Final = (
    "hegel-phase3a-q4-formal-quotient-certificate-capability-consumption/1"
)
PENDING_SCHEMA: Final = (
    "hegel-phase3a-q4-formal-quotient-certificate-capability-pending/1"
)
TERMINAL_SCHEMA: Final = (
    "hegel-phase3a-q4-formal-quotient-certificate-capability-terminal/1"
)

CAPSULE_READY: Final = "Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPSULE_READY_NOT_AUTHORIZED"
AUTHORIZATION_RECORDED: Final = (
    "Q4_FORMAL_QUOTIENT_CERTIFICATE_AUTHORIZATION_RECORDED_NOT_CONSUMED"
)
AUTHORIZATION_CONSUMED: Final = (
    "Q4_FORMAL_QUOTIENT_CERTIFICATE_AUTHORIZATION_CONSUMED"
)
PENDING_NOT_CONSUMED: Final = (
    "Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_PENDING_NOT_CONSUMED"
)
INVALIDATED_TERMINAL_NO_RESUME: Final = (
    "Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_INVALIDATED_TERMINAL_NO_RESUME"
)
PARTIAL_FINALIZATION_REQUIRED: Final = (
    "Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_PARTIAL_FINALIZATION_REQUIRED"
)
AUTHORIZED_SCOPE: Final = (
    "Q4_EXACT_SEALED_Q3_FORMAL_CAPABILITY_ADMISSION_REPLAY_ONCE"
)
NOT_AUTHORIZED: Final = (
    "Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_NOT_AUTHORIZED_NOT_CONSUMED"
)

DOMAINS: Final = {
    "source_closure": b"HEGEL/Q4/SOURCE_CLOSURE/V1\x00",
    "capsule": b"HEGEL/Q4/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q4/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q4/CONSUMPTION/V1\x00",
    "pending": b"HEGEL/Q4/PENDING/V1\x00",
    "terminal": b"HEGEL/Q4/TERMINAL/V1\x00",
}
Q3_DOMAINS: Final = {
    "capsule": b"HEGEL/Q3/CAPSULE/V1\x00",
    "authorization": b"HEGEL/Q3/AUTHORIZATION/V1\x00",
    "consumption": b"HEGEL/Q3/CONSUMPTION/V1\x00",
    "terminal": b"HEGEL/Q3/TERMINAL/V1\x00",
}

Q4_SOURCE_PATHS: Final = (
    "src/hegel_machine/phase3_q4_formal_quotient_certificate_capability_v1.py",
    "src/hegel_machine/phase3_q4_completion_replay_v1.py",
    "src/hegel_machine/strict_cbor_v1.py",
    "tools/phase3_q4_formal_quotient_certificate_capability_v1.py",
)

EXECUTION_POLICY: Final = {
    "logical_campaign_limit": 1,
    "maximum_internal_popen_count": 0,
    "maximum_actor_start_count": 0,
    "maximum_docker_invocation_count": 0,
    "maximum_measurement_count": 0,
    "maximum_target_open_count": 0,
    "maximum_target_truth_access_count": 0,
    "maximum_split_access_count": 0,
    "maximum_role_evaluation_count": 0,
    "maximum_q1_read_count": 0,
    "maximum_q2_read_count": 0,
    "maximum_archive_read_count": 0,
    "maximum_key_operation_count": 0,
    "maximum_signature_count": 0,
    "maximum_certificate_issuance_count": 0,
    "network_allowed": False,
    "preconsumption_pending_same_capsule_digest": True,
    "preconsumption_pending_consumes_authorization": False,
    "postconsumption_pending_allowed": False,
    "partial_finalization_artifact_to_terminal_only": True,
}

AUTHORITY_BOUNDARY: Final = {
    "q3_state": "Q3_BOUNDED_QUOTIENT_CERTIFICATE_CANDIDATE_COMPLETE",
    "q4_disposition": "ADMISSION_ONLY_NOT_ISSUANCE",
    "purpose5_key_installed": False,
    "formal_certificate_signed": False,
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

ZERO_ACTION_AUDIT: Final = {
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
MAX_Q3_ARTIFACT_BYTES: Final = 1024 * 1024
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class Q4CompletionReplayError(ValueError):
    """One Q4 state object or sealed Q3 predecessor differs."""


def _fail(detail: str) -> NoReturn:
    raise Q4CompletionReplayError(detail)


@dataclass(frozen=True, slots=True)
class LoadedJson:
    path: Path
    raw: bytes
    raw_sha256: str
    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Q3SealedReplayV1:
    preflight_receipt: dict[str, Any]
    capability_receipt: Any

    @property
    def artifact_payload(self) -> bytes:
        payload = getattr(self.capability_receipt, "canonical_bytes", None)
        if type(payload) is not bytes:
            _fail("Q4 core capability receipt has no canonical bytes")
        return payload


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


def load_pretty_json(path: Path, label: str) -> LoadedJson:
    try:
        raw = path.read_bytes()
    except OSError as error:
        _fail(f"{label} unavailable: {error}")
    return LoadedJson(path, raw, raw_sha256(raw), pretty_json_bytes(raw, label))


def load_preregistration(path: Path) -> LoadedJson:
    loaded = load_pretty_json(path, "Q4 preregistration")
    value = loaded.value
    if (
        value.get("schema_version")
        != "hegel-phase3a-q4-formal-quotient-certificate-capability-preregistration/1"
        or value.get("status") != "Q4_SPEC_FROZEN_NOT_AUTHORIZED"
        or type(value.get("predecessor_q3")) is not dict
    ):
        _fail("Q4 preregistration identity differs")
    return loaded


def load_contract(path: Path) -> LoadedJson:
    loaded = load_pretty_json(path, "Q4 contract")
    value = loaded.value
    if (
        value.get("schema_version") != CONTRACT_SCHEMA
        or value.get("contract_id") != CONTRACT_ID
        or value.get("action_id") != ACTION_ID
        or type(value.get("fixed_paths")) is not dict
        or type(value.get("q3_inputs")) is not dict
        or value.get("execution_policy") != EXECUTION_POLICY
        or value.get("authority_boundary") != AUTHORITY_BOUNDARY
    ):
        _fail("Q4 contract identity differs")
    prereg_path = Path(value.get("preregistration_path", ""))
    prereg = load_preregistration(prereg_path)
    if prereg.raw_sha256 != value.get("preregistration_raw_sha256"):
        _fail("Q4 preregistration raw identity differs")
    return loaded


def _semantic_digest(value: Mapping[str, Any], kind: str, field: str) -> str:
    unsigned = deepcopy(dict(value))
    supplied = unsigned.pop(field, None)
    expected = _digest(Q3_DOMAINS[kind], unsigned)
    if supplied != expected:
        _fail(f"Q3 {kind} semantic digest differs")
    return expected


def _expected_predecessor(preregistration: Mapping[str, Any]) -> Mapping[str, Any]:
    predecessor = preregistration.get("predecessor_q3")
    if type(predecessor) is not dict:
        _fail("Q4 predecessor Q3 registry differs")
    return predecessor


def _expected_input(
    predecessor: Mapping[str, Any], kind: str
) -> Mapping[str, Any]:
    value = predecessor.get(kind)
    if type(value) is not dict:
        _fail(f"Q4 predecessor Q3 {kind} registry differs")
    if _tagged(value.get("raw_sha256"), f"Q3 {kind} raw digest") is None:
        raise AssertionError
    return value


def replay_q3_predecessor_v1(
    preregistration: Mapping[str, Any],
    *,
    q4_source_commit: str,
    q4_source_closure_root: str,
    capsule_payload: bytes,
    authorization_payload: bytes,
    consumption_payload: bytes,
    terminal_payload: bytes,
    artifact_payload: bytes,
) -> Q3SealedReplayV1:
    """Replay exactly five sealed Q3 inputs and construct one Q4a receipt."""

    predecessor = _expected_predecessor(preregistration)
    if (
        type(q4_source_commit) is not str
        or _SOURCE_COMMIT.fullmatch(q4_source_commit) is None
    ):
        _fail("Q4 source commit differs")
    q4_closure = _tagged(q4_source_closure_root, "Q4 source closure")
    payloads = {
        "capsule": capsule_payload,
        "authorization": authorization_payload,
        "consumption": consumption_payload,
        "terminal": terminal_payload,
        "artifact": artifact_payload,
    }
    selected_rows: list[dict[str, Any]] = []
    for kind, payload in payloads.items():
        expected = _expected_input(predecessor, kind)
        actual_raw = raw_sha256(payload)
        if actual_raw != expected["raw_sha256"]:
            _fail(f"Q3 {kind} raw SHA-256 differs")
        selected_rows.append(
            {"kind": kind, "byte_count": len(payload), "raw_sha256": actual_raw}
        )

    capsule = strict_json_bytes(capsule_payload, "Q3 capsule")
    authorization = strict_json_bytes(authorization_payload, "Q3 authorization")
    consumption = strict_json_bytes(consumption_payload, "Q3 consumption")
    terminal = strict_json_bytes(terminal_payload, "Q3 terminal")

    capsule_digest = _semantic_digest(capsule, "capsule", "capsule_digest")
    authorization_digest = _semantic_digest(
        authorization, "authorization", "authorization_digest"
    )
    consumption_digest = _semantic_digest(consumption, "consumption", "record_digest")
    terminal_digest = _semantic_digest(terminal, "terminal", "record_digest")

    semantic_expectations = (
        ("capsule", "capsule_digest", capsule_digest),
        ("authorization", "authorization_digest", authorization_digest),
        ("consumption", "record_digest", consumption_digest),
        ("terminal", "record_digest", terminal_digest),
    )
    for kind, field, actual in semantic_expectations:
        expected = _expected_input(predecessor, kind).get(field)
        if expected != actual:
            _fail(f"Q3 {kind} configured semantic digest differs")

    outcome = terminal.get("outcome")
    if (
        capsule.get("status")
        != "Q3_BOUNDED_QUOTIENT_CERTIFICATE_CAPSULE_READY_NOT_AUTHORIZED"
        or authorization.get("status")
        != "Q3_BOUNDED_QUOTIENT_CERTIFICATE_AUTHORIZATION_RECORDED_NOT_CONSUMED"
        or consumption.get("status")
        != "Q3_BOUNDED_QUOTIENT_CERTIFICATE_AUTHORIZATION_CONSUMED"
        or terminal.get("status")
        != "Q3_BOUNDED_QUOTIENT_CERTIFICATE_CANDIDATE_COMPLETE"
        or type(outcome) is not dict
        or outcome.get("status")
        != "Q3_BOUNDED_QUOTIENT_CERTIFICATE_CANDIDATE_COMPLETE"
        or authorization.get("capsule_digest") != capsule_digest
        or consumption.get("capsule_digest") != capsule_digest
        or terminal.get("capsule_digest") != capsule_digest
        or consumption.get("authorization_digest") != authorization_digest
        or terminal.get("authorization_digest") != authorization_digest
        or terminal.get("consumption_record_digest") != consumption_digest
        or terminal.get("authorization_consumption_count") != 1
        or terminal.get("logical_campaign_count") != 1
        or capsule.get("source_commit") != predecessor.get("source_commit")
        or capsule.get("source_closure_root")
        != predecessor.get("source_closure_root")
        or outcome.get("outside_certificate_issued") is not False
        or outcome.get("formal_certificate_signed") is not False
        or outcome.get("external_authority") is not False
        or outcome.get("synthesis_released") is not False
    ):
        _fail("Q3 predecessor completion boundary differs")

    try:
        candidate = _wire.decode_sealed_q3_candidate_v1(artifact_payload)
    except (TypeError, ValueError) as error:
        _fail(f"Q3 candidate replay rejected: {error}")
    candidate_root = "sha256:" + candidate.candidate_root.hex()
    expected_candidate = _expected_input(predecessor, "artifact")
    if (
        candidate_root != expected_candidate.get("candidate_root")
        or outcome.get("candidate_root") != candidate_root
        or outcome.get("artifact_raw_sha256") != raw_sha256(artifact_payload)
    ):
        _fail("Q3 candidate root binding differs")

    capability = _wire.build_q4_capability_receipt_v1(
        q3_capsule_digest=bytes.fromhex(capsule_digest[7:]),
        q3_authorization_digest=bytes.fromhex(authorization_digest[7:]),
        q3_consumption_record_digest=bytes.fromhex(consumption_digest[7:]),
        q3_terminal_record_digest=bytes.fromhex(terminal_digest[7:]),
        q3_candidate_root=candidate.candidate_root,
        q3_source_closure_root=bytes.fromhex(
            _tagged(predecessor.get("source_closure_root"), "Q3 source closure")[7:]
        ),
        q3_source_commit=predecessor.get("source_commit"),
        q4_source_closure_root=bytes.fromhex(q4_closure[7:]),
        q4_source_commit=q4_source_commit,
        q3_candidate=candidate,
    )
    capability_payload = capability.canonical_bytes
    capability_root = "sha256:" + capability.receipt_root.hex()
    preflight = {
        "schema_version": (
            "hegel-phase3a-q4-formal-quotient-certificate-capability-preflight/1"
        ),
        "selected_q3_inputs": selected_rows,
        "selected_q3_input_count": 5,
        "selected_q3_input_total_bytes": sum(len(payload) for payload in payloads.values()),
        "q3_capsule_digest": capsule_digest,
        "q3_authorization_digest": authorization_digest,
        "q3_consumption_record_digest": consumption_digest,
        "q3_terminal_record_digest": terminal_digest,
        "q3_candidate_root": candidate_root,
        "expected_capability_artifact_raw_sha256": raw_sha256(capability_payload),
        "expected_capability_receipt_root": capability_root,
        "producer_free_read_only_replay_completed": True,
        "formal_codec_round_trip_completed": True,
        "capability_disposition": "ADMITTED_NOT_ISSUED",
        "q4_action_audit": deepcopy(ZERO_ACTION_AUDIT),
    }
    return Q3SealedReplayV1(preflight, capability)


def validate_source_closure_rows(rows: object) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != len(Q4_SOURCE_PATHS):
        _fail("Q4 source closure row count differs")
    checked: list[dict[str, Any]] = []
    for row, expected_path in zip(rows, Q4_SOURCE_PATHS, strict=True):
        if (
            type(row) is not dict
            or set(row) != {"relative_path", "byte_count", "raw_sha256"}
            or row.get("relative_path") != expected_path
            or type(row.get("byte_count")) is not int
            or row["byte_count"] < 1
        ):
            _fail("Q4 source closure row differs")
        _tagged(row.get("raw_sha256"), "Q4 source row digest")
        checked.append(deepcopy(row))
    return checked


def source_closure_root(rows: object) -> str:
    return _digest(DOMAINS["source_closure"], validate_source_closure_rows(rows))


def validate_preflight_receipt(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("Q4 preflight receipt differs")
    expected_keys = {
        "schema_version", "selected_q3_inputs", "selected_q3_input_count",
        "selected_q3_input_total_bytes", "q3_capsule_digest",
        "q3_authorization_digest", "q3_consumption_record_digest",
        "q3_terminal_record_digest", "q3_candidate_root",
        "expected_capability_artifact_raw_sha256",
        "expected_capability_receipt_root",
        "producer_free_read_only_replay_completed",
        "formal_codec_round_trip_completed", "capability_disposition",
        "q4_action_audit",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version")
        != "hegel-phase3a-q4-formal-quotient-certificate-capability-preflight/1"
        or value.get("selected_q3_input_count") != 5
        or type(value.get("selected_q3_inputs")) is not list
        or [row.get("kind") for row in value["selected_q3_inputs"]]
        != ["capsule", "authorization", "consumption", "terminal", "artifact"]
        or value.get("producer_free_read_only_replay_completed") is not True
        or value.get("formal_codec_round_trip_completed") is not True
        or value.get("capability_disposition") != "ADMITTED_NOT_ISSUED"
        or value.get("q4_action_audit") != ZERO_ACTION_AUDIT
    ):
        _fail("Q4 preflight receipt differs")
    for field in (
        "q3_capsule_digest", "q3_authorization_digest",
        "q3_consumption_record_digest", "q3_terminal_record_digest",
        "q3_candidate_root", "expected_capability_artifact_raw_sha256",
        "expected_capability_receipt_root",
    ):
        _tagged(value.get(field), field)
    return deepcopy(value)


def build_capsule(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    rows = validate_source_closure_rows(inputs.get("source_closure_rows"))
    root = source_closure_root(rows)
    if inputs.get("source_closure_root") != root:
        _fail("Q4 source closure root differs")
    source_commit = inputs.get("source_commit")
    run_id = inputs.get("run_id")
    created = inputs.get("created_at_unix_seconds")
    if type(source_commit) is not str or _SOURCE_COMMIT.fullmatch(source_commit) is None:
        _fail("Q4 source commit differs")
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        _fail("Q4 run id differs")
    if type(created) is not int or created < 0:
        _fail("Q4 creation time differs")
    preflight = validate_preflight_receipt(inputs.get("preflight_receipt"))
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA,
        "status": CAPSULE_READY,
        "contract_raw_sha256": contract.raw_sha256,
        "preregistration_raw_sha256": contract.value[
            "preregistration_raw_sha256"
        ],
        "action_id": ACTION_ID,
        "source_commit": source_commit,
        "source_closure_rows": rows,
        "source_closure_root": root,
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
        _fail("Q4 capsule differs")
    inputs = {
        key: value.get(key)
        for key in (
            "source_commit", "source_closure_rows", "source_closure_root",
            "run_id", "created_at_unix_seconds", "preflight_receipt",
        )
    }
    expected = build_capsule(contract, preregistration, inputs)
    if value != expected:
        _fail("Q4 capsule replay differs")
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
        "q4_action_audit_limit": deepcopy(ZERO_ACTION_AUDIT),
        "certificate_issuance_allowed": False,
        "certificate_signing_allowed": False,
        "key_operation_allowed": False,
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
        _fail("Q4 authorization replay differs")
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
        "q4_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "outside_certificate_issued": False,
        "formal_certificate_signed": False,
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
        _fail("Q4 consumption replay differs")
    return deepcopy(expected)


def build_pending(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    phase: str,
    condition: str,
    detail: str,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked = validate_capsule(capsule, contract, preregistration)
    authorization_digest = None
    if authorization is not None:
        authorization_digest = validate_authorization(
            authorization, contract, preregistration, checked
        )["authorization_digest"]
    if any(type(item) is not str or not item for item in (phase, condition, detail)):
        _fail("Q4 pending fields differ")
    body: dict[str, Any] = {
        "schema_version": PENDING_SCHEMA,
        "status": PENDING_NOT_CONSUMED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked["capsule_digest"],
        "authorization_digest": authorization_digest,
        "phase": phase,
        "condition": condition,
        "detail": detail,
        "resume_capsule_digest": checked["capsule_digest"],
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "q4_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "pending_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("pending_digest")
    body["pending_digest"] = _digest(DOMAINS["pending"], unsigned)
    return body


def validate_artifact_binding(payload: bytes, capsule: Mapping[str, Any]) -> Any:
    try:
        receipt = _wire.decode_q4_capability_receipt_v1(payload)
    except (TypeError, ValueError) as error:
        _fail(f"Q4 capability receipt replay rejected: {error}")
    preflight = validate_preflight_receipt(capsule.get("preflight_receipt"))
    if (
        raw_sha256(payload) != preflight["expected_capability_artifact_raw_sha256"]
        or "sha256:" + receipt.receipt_root.hex()
        != preflight["expected_capability_receipt_root"]
    ):
        _fail("Q4 capability receipt/capsule binding differs")
    return receipt


def build_complete_terminal(
    contract: LoadedJson,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
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
    receipt = validate_artifact_binding(artifact_payload, checked_capsule)
    outcome = {
        "status": _wire.Q4_SUCCESS_STATUS,
        "capability_disposition": "ADMITTED_NOT_ISSUED",
        "capsule_digest": checked_capsule["capsule_digest"],
        "artifact_raw_sha256": raw_sha256(artifact_payload),
        "capability_receipt_root": "sha256:" + receipt.receipt_root.hex(),
        "producer_free_q3_replay_completed": True,
        "formal_codec_round_trip_completed": True,
        "q4_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "purpose5_key_installed": False,
        "formal_certificate_signed": False,
        "outside_certificate_issued": False,
        "external_authority": False,
        "synthesis_released": False,
        "terminal_no_resume": False,
        "campaign_complete": True,
    }
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": _wire.Q4_SUCCESS_STATUS,
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
    *,
    failure_phase: str,
    failure_code: str,
    failure_detail: str,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        consumption, contract, preregistration, checked_capsule,
        checked_authorization,
    )
    if any(type(item) is not str or not item for item in (
        failure_phase, failure_code, failure_detail
    )):
        _fail("Q4 invalidated failure fields differ")
    outcome = {
        "status": INVALIDATED_TERMINAL_NO_RESUME,
        "capsule_digest": checked_capsule["capsule_digest"],
        "artifact_raw_sha256": None,
        "capability_receipt_root": None,
        "q4_action_audit": deepcopy(ZERO_ACTION_AUDIT),
        "purpose5_key_installed": False,
        "formal_certificate_signed": False,
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
    *,
    artifact_payload: bytes | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("outcome")) is not dict:
        _fail("Q4 terminal must contain an outcome")
    if value.get("status") == _wire.Q4_SUCCESS_STATUS:
        if artifact_payload is None:
            _fail("Q4 complete terminal requires the sealed capability receipt")
        expected = build_complete_terminal(
            contract, preregistration, capsule, authorization, consumption,
            artifact_payload,
        )
    elif value.get("status") == INVALIDATED_TERMINAL_NO_RESUME:
        if artifact_payload is not None:
            _fail("Q4 invalidated terminal forbids a valid capability receipt")
        outcome = value["outcome"]
        expected = build_invalidated_terminal(
            contract, preregistration, capsule, authorization, consumption,
            failure_phase=outcome.get("failure_phase"),
            failure_code=outcome.get("failure_code"),
            failure_detail=outcome.get("failure_detail"),
        )
    else:
        _fail("Q4 terminal status differs")
    if value != expected:
        _fail("Q4 terminal replay differs")
    return deepcopy(expected)


__all__ = [
    "ACTION_ID", "AUTHORIZATION_CONSUMED", "AUTHORIZATION_RECORDED",
    "AUTHORIZATION_SCHEMA", "AUTHORIZED_SCOPE", "AUTHORITY_BOUNDARY",
    "CAPSULE_INPUTS_SCHEMA", "CAPSULE_READY", "CAPSULE_SCHEMA",
    "CONSUMPTION_SCHEMA", "CONTRACT_ID", "CONTRACT_SCHEMA", "DOMAINS",
    "EXECUTION_POLICY", "INVALIDATED_TERMINAL_NO_RESUME", "LoadedJson",
    "NOT_AUTHORIZED", "PARTIAL_FINALIZATION_REQUIRED", "PENDING_NOT_CONSUMED",
    "PENDING_SCHEMA", "Q3SealedReplayV1", "Q4CompletionReplayError",
    "Q4_SOURCE_PATHS", "TERMINAL_SCHEMA", "ZERO_ACTION_AUDIT",
    "build_authorization", "build_capsule", "build_complete_terminal",
    "build_consumption", "build_invalidated_terminal", "build_pending",
    "canonical_json_bytes", "load_contract", "load_preregistration",
    "pretty_json_bytes", "raw_sha256", "replay_q3_predecessor_v1",
    "source_closure_root", "strict_json_bytes", "validate_artifact_binding",
    "validate_authorization", "validate_capsule", "validate_consumption",
    "validate_preflight_receipt", "validate_source_closure_rows",
    "validate_terminal",
]
