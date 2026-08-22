"""R01 governance current-cut *mechanics* for Phase-2B Step 5.

This module freezes one canonical current-cut request ``/1`` and emits the
``CURRENT_CUT_RESPONSE /1`` and ``CURRENT_CUT_STATEMENT /1`` shapes frozen by
``phase2b_formal_wire_covert_step5_r01_evidence_contract_v1``.  It is a narrow
connected-socket primitive, not a listener, daemon, systemd unit, admission
controller, or authority issuer.

The governance side obtains an optimistic append-only snapshot exclusively
through public producer APIs::

    current_head_bytes() -> latest_projection_bytes() -> current_head_bytes()

Both head reads must be byte equal and the exact producer projection must bind
the same ledger, head, complete-event-set root, authority graph, attempt, key,
count, and ACTIVE/zero-revocation state.  ``latest_projection_bytes()`` performs
the producer's complete-chain replay, but the three calls are not one atomic
operation.  Byte equality rules out an append *during* these three reads for
this append-only bounded ledger; an append can still occur immediately after
the last read or after the response is sent.  Therefore this property is named
``stable_during_three_replays`` and never ``currentness_verified``.

The client accepts the raw authority graph, raw out-of-band pins, and the raw
complete producer-free logical-time chain.  It invokes and then requires the
private-lineage logical-time replay result; caller-supplied decoded mappings or
replay-result objects are not accepted.  A genesis-only time chain is rejected
and the verified event count must equal the ACTIVE owner event count (exactly
one).  This only cross-binds count, authority, attempt, and the presented time
head/token.  It cannot bind the time event plan to owner decision bytes because
the current public owner ledger API does not export the raw owner chain.

Both endpoints require an already-connected AF_UNIX SOCK_SEQPACKET socket and
read Linux SO_PEERCRED before and after the exchange.  The server requires the
static runner UID/GID; the client requires the static governance UID/GID.  The
authority profile contains no socket-endpoint or runtime-instance pin.  The
derived endpoint/runtime identifiers are consequently local, content-addressed
observations only and are not live-runtime-closure evidence.

Success proves candidate wire/socket/snapshot mechanics only.  It does *not*
prove response-time or admission-time currentness, an atomic owner/time fence,
external anti-fork anchoring, CAS durability, image execution, live distinct
UID/control domains, service-key custody, owner authority, R01, readiness, or
execution authorization.  Real admission requires a fresh admission-time
requery/fence plus the producer-free owner replay and main-CAS evidence graph;
it remains deliberately NOT_IMPLEMENTED here.

Aligned frozen sources (informational source review pins): authority-v2
``03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86``;
owner-status producer
``477178052e3db8e7773e3832334c3a2faf2b2d04f2e93255ad7e3d59556206a2``;
logical-time replay core
``a618291256723e2ce5ac8d77d4b77a3deec4bbf787ab3422b0b61b9c36358def``;
R01 evidence contract
``574e07e7ce0f19075c29fa59cc94e4e988a34c301e9e09005a8aed805d69e96d``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import re
import socket
import stat
import struct
from types import FunctionType
from typing import Final, Mapping, NoReturn

from .phase2b_formal_wire_covert_step5_authority_v2 import (
    Phase2BStep5AuthorityErrorV2 as _AuthorityError,
    load_step5_authority_profile_v2 as _load_authority_profile,
    load_step5_signer_registry_v2 as _load_signer_registry,
    load_step5_stage_b_trust_lock_v2 as _load_stage_b_lock,
    require_locked_step5_authority_graph_v2 as _require_authority_graph,
)
from .phase2b_formal_wire_covert_step5_logical_time_replay_core_v1 import (
    ExpectedLogicalEventCoreV1,
    Phase2BStep5LogicalTimeReplayCoreErrorV1 as _LogicalTimeReplayError,
    require_verified_logical_time_replay_core_v1 as _require_time_replay,
    verify_logical_time_replay_core_v1 as _verify_time_replay,
)
from .phase2b_formal_wire_covert_step5_owner_status_v1 import (
    OwnerStatusHeadV1 as _OwnerHead,
    OwnerStatusLedgerV1,
    OwnerStatusProjectionV1 as _OwnerProjection,
    OwnerStatusServiceAppendV1,
    Phase2BStep5OwnerStatusErrorV1 as _OwnerStatusError,
)
from .phase2b_formal_wire_covert_step5_r01_evidence_contract_v1 import (
    CANDIDATE_CLAIM_LEVEL_V1,
    CANDIDATE_STATUS_V1,
    CURRENT_CUT_RESPONSE_FIELDS_V1,
    CURRENT_CUT_RESPONSE_SCHEMA_V1,
    CURRENT_CUT_STATEMENT_FIELDS_V1,
    CURRENT_CUT_STATEMENT_SCHEMA_V1,
    GOVERNANCE_PEER_OBSERVATION_FIELDS_V1,
    OWNER_STATUS_BINDING_FIELDS_V1,
    PRE_CUT_LOGICAL_TIME_BINDING_FIELDS_V1,
    canonical_r01_evidence_contract_json_bytes_v1 as _evidence_canonical,
    phase2b_step5_r01_evidence_content_id_v1 as _evidence_content_id,
)
from .rfc8785_v1 import (
    RFC8785Error as _RFC8785Error,
    canonicalize_rfc8785_v1 as _canonicalize,
    decode_and_require_canonical_rfc8785_v1 as _decode_canonical,
)


AUTHORITY_V2_SOURCE_SHA256_V1: Final = (
    "03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86"
)
OWNER_STATUS_SOURCE_SHA256_V1: Final = (
    "477178052e3db8e7773e3832334c3a2faf2b2d04f2e93255ad7e3d59556206a2"
)
LOGICAL_TIME_REPLAY_CORE_SOURCE_SHA256_V1: Final = (
    "a618291256723e2ce5ac8d77d4b77a3deec4bbf787ab3422b0b61b9c36358def"
)
R01_EVIDENCE_CONTRACT_SOURCE_SHA256_V1: Final = (
    "574e07e7ce0f19075c29fa59cc94e4e988a34c301e9e09005a8aed805d69e96d"
)

CURRENT_CUT_REQUEST_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-current-cut-request/1"
)
CURRENT_CUT_MECHANICS_RESULT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-current-cut-mechanics-result/1"
)
CURRENT_CUT_REQUEST_FIELDS_V1: Final = frozenset({
    "schema_version",
    "status",
    "claim_level",
    "stage_b_trust_lock_id",
    "authority_profile_id",
    "signer_registry_id",
    "attempt_context_id",
    "requester_service_name",
    "requester_service_binding_id",
    "governance_service_name",
    "governance_service_binding_id",
    "cut_request_nonce",
    "request_id",
})

REQUESTER_SERVICE_NAME_V1: Final = "runner"
GOVERNANCE_SERVICE_NAME_V1: Final = "governance_status"
CURRENT_CUT_REQUEST_PREFIX_V1: Final = "phase2b_step5_current_cut_request_v1_"
SOCKET_ENDPOINT_PREFIX_V1: Final = (
    "phase2b_step5_governance_socket_endpoint_observation_v1_"
)
RUNTIME_OBSERVATION_PREFIX_V1: Final = (
    "phase2b_step5_governance_runtime_local_observation_v1_"
)
CURRENT_CUT_REQUEST_CLAIM_LEVEL_V1: Final = (
    "SOCK_SEQPACKET_CHALLENGE_REQUEST_CANDIDATE_NOT_CURRENT_NOT_AUTHORITY_"
    "NOT_R01_NOT_READINESS_NOT_EXECUTION"
)
CURRENT_CUT_MECHANICS_CLAIM_BOUNDARY_V1: Final = (
    "RAW_AUTHORITY_V2_BOUND_PRESENTED_OWNER_PRODUCER_PROJECTION_AND_"
    "SO_PEERCRED_SEQPACKET_AND_PRESENTED_PRECUT_TIME_REPLAY_MECHANICS_ONLY_"
    "NOT_ATOMIC_NOT_EXTERNAL_CURRENTNESS_NOT_ANTI_FORK_NOT_RUNTIME_CLOSURE_"
    "NOT_DURABLE_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION"
)

FAIL_INPUT_V1: Final = "FAIL_STEP5_R01_CURRENT_CUT_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_R01_CURRENT_CUT_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_R01_CURRENT_CUT_V1_SCHEMA"
FAIL_CONTENT_ID_V1: Final = "FAIL_STEP5_R01_CURRENT_CUT_V1_CONTENT_ID"
FAIL_AUTHORITY_V1: Final = "FAIL_STEP5_R01_CURRENT_CUT_V1_AUTHORITY"
FAIL_BINDING_V1: Final = "FAIL_STEP5_R01_CURRENT_CUT_V1_BINDING"
FAIL_SOCKET_V1: Final = "FAIL_STEP5_R01_CURRENT_CUT_V1_SOCKET"
FAIL_PEER_CREDENTIAL_V1: Final = (
    "FAIL_STEP5_R01_CURRENT_CUT_V1_PEER_CREDENTIAL"
)
FAIL_OWNER_SNAPSHOT_V1: Final = "FAIL_STEP5_R01_CURRENT_CUT_V1_OWNER_SNAPSHOT"
FAIL_LOGICAL_TIME_V1: Final = "FAIL_STEP5_R01_CURRENT_CUT_V1_LOGICAL_TIME"
FAIL_CLAIM_BOUNDARY_V1: Final = (
    "FAIL_STEP5_R01_CURRENT_CUT_V1_CLAIM_BOUNDARY"
)
FAIL_RESULT_V1: Final = "FAIL_STEP5_R01_CURRENT_CUT_V1_RESULT"
FAIL_NOT_IMPLEMENTED_V1: Final = (
    "FAIL_STEP5_R01_CURRENT_CUT_V1_NOT_IMPLEMENTED"
)

_REQUEST_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/R01/CURRENT_CUT_REQUEST/V1\0"
)
_SOCKET_ENDPOINT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/R01/GOVERNANCE_SOCKET_ENDPOINT_LOCAL/V1\0"
)
_RUNTIME_OBSERVATION_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/R01/GOVERNANCE_RUNTIME_LOCAL/V1\0"
)
_MAX_REQUEST_BYTES: Final = 64 * 1024
_MAX_RESPONSE_BYTES: Final = 8 * 1024 * 1024
_MAX_SAFE_INTEGER: Final = (1 << 53) - 1
_HEX64 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_./:-]{0,255}_[0-9a-f]{64}")
_OWNER_HEAD_FROM_BYTES: Final = _OwnerHead.from_canonical_bytes
_OWNER_PROJECTION_FROM_BYTES: Final = _OwnerProjection.from_canonical_bytes
_OWNER_LEDGER_CURRENT_HEAD_BYTES: Final = OwnerStatusLedgerV1.current_head_bytes
_OWNER_LEDGER_LATEST_PROJECTION_BYTES: Final = (
    OwnerStatusLedgerV1.latest_projection_bytes
)
_OWNER_APPEND_POST_INIT: Final = OwnerStatusServiceAppendV1.__post_init__


class Phase2BStep5R01CurrentCutErrorV1(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5R01CurrentCutErrorV1(code, detail)


def _canonical(value: object, _canon=_canonicalize) -> bytes:
    try:
        return _canon(value)
    except (TypeError, ValueError, _RFC8785Error) as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_CANONICAL_V1, "RFC-8785 canonicalization failed"
        ) from exc


def _restore_wire_integers(value: object, _isfinite=math.isfinite) -> object:
    if type(value) is float and _isfinite(value) and value.is_integer():
        if abs(value) > _MAX_SAFE_INTEGER:
            _fail(FAIL_SCHEMA_V1, "decoded integer exceeds I-JSON safe range")
        return int(value)
    if type(value) is list:
        return [_restore_wire_integers(item) for item in value]
    if type(value) is dict:
        return {key: _restore_wire_integers(item) for key, item in value.items()}
    return value


def _decode(payload: bytes, maximum: int, name: str, _decoder=_decode_canonical) -> dict[str, object]:
    if (
        type(payload) is not bytes
        or not 1 <= len(payload) <= maximum
        or payload.endswith(b"\n")
    ):
        _fail(FAIL_INPUT_V1, f"{name} must be bounded exact bytes without LF")
    try:
        value = _restore_wire_integers(_decoder(payload))
    except (TypeError, ValueError, _RFC8785Error) as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_CANONICAL_V1, f"{name} is not exact canonical RFC-8785"
        ) from exc
    if type(value) is not dict:
        _fail(FAIL_SCHEMA_V1, f"{name} must be an exact object")
    return value


def _object(value: object, fields: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != fields:
        _fail(FAIL_SCHEMA_V1, f"{name} keyset differs")
    return value


def _text(value: object, name: str, maximum: int = 4096) -> str:
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= maximum:
        _fail(FAIL_SCHEMA_V1, f"{name} must be bounded nonempty text")
    return value


def _identifier(value: object, name: str, prefix: str | None = None) -> str:
    text = _text(value, name, 320)
    if _IDENTIFIER.fullmatch(text) is None or (
        prefix is not None and not text.startswith(prefix)
    ):
        _fail(FAIL_SCHEMA_V1, f"{name} identifier differs")
    return text


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_SAFE_INTEGER:
        _fail(FAIL_SCHEMA_V1, f"{name} safe integer differs")
    return value


def _sha_hex(value: object, name: str) -> str:
    text = _text(value, name, 64)
    if _HEX64.fullmatch(text) is None:
        _fail(FAIL_SCHEMA_V1, f"{name} must be lowercase SHA-256 hex")
    return text


def _sha_address(value: object, name: str) -> str:
    text = _text(value, name, 71)
    if not text.startswith("sha256:") or _HEX64.fullmatch(text[7:]) is None:
        _fail(FAIL_SCHEMA_V1, f"{name} must be sha256:<lowercase hex>")
    return text


def _payload_sha_hex(payload: bytes, _sha256=hashlib.sha256) -> str:
    if type(payload) is not bytes:
        _fail(FAIL_INPUT_V1, "hash input must be exact bytes")
    return _sha256(payload).hexdigest()


def _payload_sha_address(payload: bytes) -> str:
    return "sha256:" + _payload_sha_hex(payload)


def _content_id(
    prefix: str,
    domain: bytes,
    subject: object,
    _sha256=hashlib.sha256,
) -> str:
    return prefix + _sha256(domain + _canonical(subject)).hexdigest()


def _without(mapping: Mapping[str, object], *names: str) -> dict[str, object]:
    return {key: value for key, value in mapping.items() if key not in names}


def _exact_mapping_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = tuple(left.keys())
        right_keys = tuple(right.keys())
        return (
            all(type(key) is str for key in left_keys + right_keys)
            and frozenset(left_keys) == frozenset(right_keys)
            and all(_exact_mapping_equal(left[key], right[key]) for key in left_keys)
        )
    if type(left) is list and type(right) in (list, tuple):
        return len(left) == len(right) and all(
            _exact_mapping_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is tuple and type(right) in (list, tuple):
        return len(left) == len(right) and all(
            _exact_mapping_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _load_authority_context(
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_payload_sha256_digest: bytes,
    expected_attempt_context_id: str,
    _load_lock=_load_stage_b_lock,
    _load_profile=_load_authority_profile,
    _load_registry=_load_signer_registry,
    _require_graph=_require_authority_graph,
    _authority_error=_AuthorityError,
) -> tuple[str, str, str, str, Mapping[str, object], Mapping[str, object]]:
    if any(
        type(value) is not bytes
        for value in (
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest,
        )
    ):
        _fail(FAIL_INPUT_V1, "raw authority graph and OOB pins must be exact bytes")
    attempt = _identifier(expected_attempt_context_id, "attempt context ID")
    try:
        lock = _load_lock(
            stage_b_trust_lock_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_payload_sha256_digest
            ),
        )
        profile = _load_profile(authority_profile_payload, lock)
        registry = _load_registry(signer_registry_payload, lock, profile)
        lock, profile, registry = _require_graph(lock, profile, registry)
    except _authority_error as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_AUTHORITY_V1, "raw authority-v2 graph rejected"
        ) from exc
    services = profile.body["service_static_bindings_in_exact_order"]
    if type(services) is not tuple or len(services) != 5:
        _fail(FAIL_AUTHORITY_V1, "authority service binding cardinality differs")
    runner_rows = tuple(
        row for row in services if row["service_name"] == REQUESTER_SERVICE_NAME_V1
    )
    governance_rows = tuple(
        row for row in services if row["service_name"] == GOVERNANCE_SERVICE_NAME_V1
    )
    if len(runner_rows) != 1 or len(governance_rows) != 1:
        _fail(FAIL_AUTHORITY_V1, "runner/governance binding cardinality differs")
    runner = runner_rows[0]
    governance = governance_rows[0]
    if not (
        runner["role"] == "SOURCE_RUNTIME_CUSTODIAN"
        and governance["role"] == "GOVERNANCE_ANCHOR"
        and runner["uid"] == 41001
        and runner["gid"] == 42001
        and governance["uid"] == 41005
        and governance["gid"] == 42005
        and runner["binding_status"] == "STATIC_REQUESTED_BINDING_NOT_OBSERVED_LIVE"
        and governance["binding_status"]
        == "STATIC_REQUESTED_BINDING_NOT_OBSERVED_LIVE"
    ):
        _fail(FAIL_AUTHORITY_V1, "runner/governance static identity policy differs")
    return (
        lock.artifact_id,
        profile.artifact_id,
        registry.artifact_id,
        attempt,
        runner,
        governance,
    )


def _request_subject(
    context: tuple[str, str, str, str, Mapping[str, object], Mapping[str, object]],
    nonce_hex: str,
) -> dict[str, object]:
    lock_id, profile_id, registry_id, attempt, runner, governance = context
    return {
        "schema_version": CURRENT_CUT_REQUEST_SCHEMA_V1,
        "status": CANDIDATE_STATUS_V1,
        "claim_level": CURRENT_CUT_REQUEST_CLAIM_LEVEL_V1,
        "stage_b_trust_lock_id": lock_id,
        "authority_profile_id": profile_id,
        "signer_registry_id": registry_id,
        "attempt_context_id": attempt,
        "requester_service_name": REQUESTER_SERVICE_NAME_V1,
        "requester_service_binding_id": runner["binding_id"],
        "governance_service_name": GOVERNANCE_SERVICE_NAME_V1,
        "governance_service_binding_id": governance["binding_id"],
        "cut_request_nonce": nonce_hex,
    }


def _build_request_from_context(
    context: tuple[str, str, str, str, Mapping[str, object], Mapping[str, object]],
    cut_request_nonce_bytes: bytes,
) -> bytes:
    if (
        type(cut_request_nonce_bytes) is not bytes
        or len(cut_request_nonce_bytes) != 32
        or cut_request_nonce_bytes == b"\x00" * 32
    ):
        _fail(FAIL_INPUT_V1, "cut request nonce must be one nonzero exact 32-byte value")
    subject = _request_subject(context, cut_request_nonce_bytes.hex())
    body = {
        **subject,
        "request_id": _content_id(
            CURRENT_CUT_REQUEST_PREFIX_V1, _REQUEST_ID_DOMAIN, subject
        ),
    }
    payload = _canonical(body)
    if len(payload) > _MAX_REQUEST_BYTES:
        _fail(FAIL_INPUT_V1, "current-cut request exceeds packet bound")
    return payload


def _load_request(
    payload: bytes,
    context: tuple[str, str, str, str, Mapping[str, object], Mapping[str, object]],
) -> dict[str, object]:
    body = _object(
        _decode(payload, _MAX_REQUEST_BYTES, "current-cut request"),
        CURRENT_CUT_REQUEST_FIELDS_V1,
        "current-cut request",
    )
    expected_header = (
        CURRENT_CUT_REQUEST_SCHEMA_V1,
        CANDIDATE_STATUS_V1,
        CURRENT_CUT_REQUEST_CLAIM_LEVEL_V1,
    )
    if (
        body["schema_version"], body["status"], body["claim_level"]
    ) != expected_header:
        _fail(FAIL_CLAIM_BOUNDARY_V1, "current-cut request header differs")
    nonce = _sha_hex(body["cut_request_nonce"], "cut request nonce")
    if nonce == "0" * 64:
        _fail(FAIL_SCHEMA_V1, "cut request nonce must be nonzero")
    expected = _request_subject(context, nonce)
    if not _exact_mapping_equal(_without(body, "request_id"), expected):
        _fail(FAIL_BINDING_V1, "current-cut request raw authority/service binding differs")
    expected_id = _content_id(
        CURRENT_CUT_REQUEST_PREFIX_V1, _REQUEST_ID_DOMAIN, expected
    )
    if body["request_id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "current-cut request ID differs")
    return body


def _require_seqpacket(connected_socket: object) -> socket.socket:
    if type(connected_socket) is not socket.socket or connected_socket.fileno() < 0:
        _fail(FAIL_SOCKET_V1, "connected socket must use exact open socket type")
    try:
        so_type = connected_socket.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
        connected_socket.getpeername()
    except OSError as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_SOCKET_V1, "socket is not connected"
        ) from exc
    if connected_socket.family != socket.AF_UNIX or so_type != socket.SOCK_SEQPACKET:
        _fail(FAIL_SOCKET_V1, "socket must be connected AF_UNIX SOCK_SEQPACKET")
    return connected_socket


def _peer_credentials(connected_socket: socket.socket) -> tuple[int, int, int]:
    if not hasattr(socket, "SO_PEERCRED"):
        _fail(FAIL_SOCKET_V1, "Linux SO_PEERCRED is unavailable")
    try:
        raw = connected_socket.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
        )
        pid, uid, gid = struct.unpack("3i", raw)
    except (OSError, struct.error) as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_PEER_CREDENTIAL_V1, "cannot read exact SO_PEERCRED tuple"
        ) from exc
    if pid < 1 or uid < 1 or gid < 1:
        _fail(FAIL_PEER_CREDENTIAL_V1, "SO_PEERCRED values must all be positive")
    return pid, uid, gid


def _require_peer_binding(
    credentials: tuple[int, int, int], binding: Mapping[str, object], name: str
) -> None:
    if (
        credentials[1] != binding["uid"]
        or credentials[2] != binding["gid"]
    ):
        _fail(
            FAIL_PEER_CREDENTIAL_V1,
            f"kernel {name} UID/GID differs from raw-pinned static binding",
        )


def _recv_packet(
    connected_socket: socket.socket, maximum: int, name: str
) -> bytes:
    try:
        payload, ancillary, flags, address = connected_socket.recvmsg(maximum + 1)
    except OSError as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_SOCKET_V1, f"{name} receive failed"
        ) from exc
    rejected_flags = socket.MSG_TRUNC
    if hasattr(socket, "MSG_CTRUNC"):
        rejected_flags |= socket.MSG_CTRUNC
    if (
        not payload
        or len(payload) > maximum
        or ancillary != []
        or flags & rejected_flags
        or address not in (None, "", b"")
    ):
        _fail(
            FAIL_SOCKET_V1,
            f"{name} packet is empty/oversize/truncated/ancillary/addressed",
        )
    return bytes(payload)


def _send_packet(connected_socket: socket.socket, payload: bytes, name: str) -> None:
    if type(payload) is not bytes or not payload:
        _fail(FAIL_INPUT_V1, f"{name} payload must be nonempty exact bytes")
    try:
        sent = connected_socket.send(payload)
    except OSError as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_SOCKET_V1, f"{name} send failed"
        ) from exc
    if sent != len(payload):
        _fail(FAIL_SOCKET_V1, f"{name} short send rejected")


def _socket_identity(connected_socket: socket.socket) -> tuple[int, int, int, int]:
    try:
        held = os.fstat(connected_socket.fileno())
    except OSError as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_SOCKET_V1, "cannot fstat held socket"
        ) from exc
    if not stat.S_ISSOCK(held.st_mode):
        _fail(FAIL_SOCKET_V1, "held descriptor is not a socket inode")
    return held.st_dev, held.st_ino, held.st_mode, connected_socket.fileno()


def _load_owner_pair(
    head_payload: bytes,
    projection_payload: bytes,
    context: tuple[str, str, str, str, Mapping[str, object], Mapping[str, object]],
    _load_head=_OWNER_HEAD_FROM_BYTES,
    _load_projection=_OWNER_PROJECTION_FROM_BYTES,
) -> tuple[_OwnerHead, _OwnerProjection]:
    try:
        head = _load_head(head_payload)
        projection = _load_projection(projection_payload)
    except _OwnerStatusError as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_OWNER_SNAPSHOT_V1, "owner head/projection producer DTO rejected"
        ) from exc
    _lock_id, profile_id, registry_id, attempt, _runner, governance = context
    if not (
        head.authority_profile_id == projection.authority_profile_id == profile_id
        and head.signer_registry_id == projection.signer_registry_id == registry_id
        and head.attempt_context_id == projection.attempt_context_id == attempt
        and head.governance_key_id == projection.governance_key_id
        == governance["role_key_id"]
        and head.governance_key_epoch == projection.governance_key_epoch
        == governance["key_epoch"] == 1
        and head.ledger_id == projection.ledger_id
        and head.head_id == projection.head_id
        and head.complete_event_set_root == projection.complete_event_set_root
        and head.event_count == projection.event_count == 1
        and head.next_sequence == 1
        and projection.latest_sequence == 0
        and head.latest_effective_status
        == projection.latest_effective_status
        == "ACTIVE"
        and head.revocation_count == projection.revocation_count == 0
        and head.terminal is False
        and projection.governance_signatures_verified is True
        and projection.logical_time_token_signatures_verified is True
        and projection.latest_authority_status_verified is False
        and projection.owner_signature_authenticated is False
        and projection.store_dependencies_verified is False
        and projection.authority_evidence is False
        and projection.r01_satisfied is False
        and projection.readiness_node_transition_issued is False
        and projection.execution_authorized is False
    ):
        _fail(
            FAIL_OWNER_SNAPSHOT_V1,
            "owner head/projection/authority/attempt/key/ACTIVE linkage differs",
        )
    return head, projection


def _owner_snapshot(
    owner_status_source: object,
    context: tuple[str, str, str, str, Mapping[str, object], Mapping[str, object]],
    _ledger_current_head=_OWNER_LEDGER_CURRENT_HEAD_BYTES,
    _ledger_latest_projection=_OWNER_LEDGER_LATEST_PROJECTION_BYTES,
    _append_post_init=_OWNER_APPEND_POST_INIT,
) -> tuple[bytes, bytes, _OwnerHead, _OwnerProjection, bool]:
    if type(owner_status_source) is OwnerStatusLedgerV1:
        ledger = owner_status_source
        try:
            head_before = _ledger_current_head(ledger)
            projection_payload = _ledger_latest_projection(ledger)
            head_after = _ledger_current_head(ledger)
        except _OwnerStatusError as exc:
            raise Phase2BStep5R01CurrentCutErrorV1(
                FAIL_OWNER_SNAPSHOT_V1,
                "public owner ledger three-replay snapshot rejected",
            ) from exc
        if head_before != head_after:
            _fail(
                FAIL_OWNER_SNAPSHOT_V1,
                "owner head changed during head/projection/head replays",
            )
        if not (
            ledger.authority_profile_id == context[1]
            and ledger.signer_registry_id == context[2]
            and ledger.attempt_context_id == context[3]
            and ledger.governance_key_id == context[5]["role_key_id"]
            and ledger.governance_key_epoch == context[5]["key_epoch"]
        ):
            _fail(FAIL_OWNER_SNAPSHOT_V1, "owner ledger constructor pins differ")
        head, projection = _load_owner_pair(head_after, projection_payload, context)
        if ledger.ledger_id != head.ledger_id:
            _fail(FAIL_OWNER_SNAPSHOT_V1, "owner ledger ID differs from replayed head")
        return head_after, projection_payload, head, projection, True
    if type(owner_status_source) is OwnerStatusServiceAppendV1:
        append = owner_status_source
        try:
            _append_post_init(append)
            head_payload = append.head_payload
            projection_payload = append.projection_payload
        except _OwnerStatusError as exc:
            raise Phase2BStep5R01CurrentCutErrorV1(
                FAIL_OWNER_SNAPSHOT_V1,
                "exact producer append/projection result rejected",
            ) from exc
        head, projection = _load_owner_pair(head_payload, projection_payload, context)
        if not (
            append.head_id == head.head_id
            and append.complete_event_set_root == head.complete_event_set_root
            and append.mechanics_only is True
            and append.authority_evidence is False
            and append.r01_satisfied is False
        ):
            _fail(FAIL_OWNER_SNAPSHOT_V1, "producer append result linkage differs")
        return bytes(head_payload), bytes(projection_payload), head, projection, False
    _fail(
        FAIL_INPUT_V1,
        "owner source must be exact OwnerStatusLedgerV1 or exact producer append result",
    )


def _build_response(
    request: Mapping[str, object],
    context: tuple[str, str, str, str, Mapping[str, object], Mapping[str, object]],
    head_payload: bytes,
    projection_payload: bytes,
    head: _OwnerHead,
    projection: _OwnerProjection,
    _canonical_evidence=_evidence_canonical,
    _content_id_evidence=_evidence_content_id,
) -> bytes:
    subject: dict[str, object] = {
        "schema_version": CURRENT_CUT_RESPONSE_SCHEMA_V1,
        "status": CANDIDATE_STATUS_V1,
        "stage_b_trust_lock_id": context[0],
        "authority_profile_id": context[1],
        "signer_registry_id": context[2],
        "attempt_context_id": context[3],
        "governance_service_binding_id": context[5]["binding_id"],
        "cut_request_nonce": request["cut_request_nonce"],
        "owner_status_ledger_id": head.ledger_id,
        "current_head_id": head.head_id,
        "current_head_payload_sha256": _payload_sha_address(head_payload),
        "complete_event_set_root": head.complete_event_set_root,
        "projection_id": projection.projection_id,
        "projection_payload_sha256": _payload_sha_address(projection_payload),
        "event_count": projection.event_count,
        "latest_sequence": projection.latest_sequence,
        "latest_effective_status": projection.latest_effective_status,
        "revocation_count": projection.revocation_count,
        "terminal": head.terminal,
    }
    body = {
        **subject,
        "response_id": _content_id_evidence("current_cut_response", subject),
    }
    if frozenset(body) != CURRENT_CUT_RESPONSE_FIELDS_V1:
        _fail(FAIL_SCHEMA_V1, "emitted evidence current-cut response keyset differs")
    try:
        payload = _canonical_evidence(body)
    except Exception as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_CANONICAL_V1, "evidence response canonicalization rejected"
        ) from exc
    if type(payload) is not bytes or len(payload) > _MAX_RESPONSE_BYTES:
        _fail(FAIL_CANONICAL_V1, "evidence response byte bound differs")
    return payload


def _load_response(
    payload: bytes,
    request: Mapping[str, object],
    context: tuple[str, str, str, str, Mapping[str, object], Mapping[str, object]],
    _content_id_evidence=_evidence_content_id,
) -> dict[str, object]:
    body = _object(
        _decode(payload, _MAX_RESPONSE_BYTES, "current-cut response"),
        CURRENT_CUT_RESPONSE_FIELDS_V1,
        "current-cut response",
    )
    if not (
        body["schema_version"] == CURRENT_CUT_RESPONSE_SCHEMA_V1
        and body["status"] == CANDIDATE_STATUS_V1
        and (
            body["stage_b_trust_lock_id"],
            body["authority_profile_id"],
            body["signer_registry_id"],
            body["attempt_context_id"],
        )
        == context[:4]
        and body["governance_service_binding_id"] == context[5]["binding_id"]
        == request["governance_service_binding_id"]
        and body["cut_request_nonce"] == request["cut_request_nonce"]
        and body["event_count"] == 1
        and body["latest_sequence"] == 0
        and body["latest_effective_status"] == "ACTIVE"
        and body["revocation_count"] == 0
        and body["terminal"] is False
    ):
        _fail(FAIL_BINDING_V1, "response request/authority/ACTIVE linkage differs")
    _identifier(body["owner_status_ledger_id"], "owner ledger ID", "phase2b_step5_owner_status_ledger_v1_")
    _identifier(body["current_head_id"], "owner head ID", "phase2b_step5_owner_status_head_v1_")
    _identifier(body["projection_id"], "owner projection ID", "phase2b_step5_owner_status_projection_v1_")
    for field in (
        "current_head_payload_sha256",
        "complete_event_set_root",
        "projection_payload_sha256",
    ):
        _sha_address(body[field], f"response {field}")
    expected_id = _content_id_evidence(
        "current_cut_response", _without(body, "response_id")
    )
    if body["response_id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "current-cut response ID differs")
    return body


def _owner_binding(response: Mapping[str, object]) -> dict[str, object]:
    return {
        "ledger_id": response["owner_status_ledger_id"],
        "current_head_id": response["current_head_id"],
        "current_head_payload_sha256": response["current_head_payload_sha256"],
        "complete_event_set_root": response["complete_event_set_root"],
        "projection_id": response["projection_id"],
        "projection_payload_sha256": response["projection_payload_sha256"],
        "event_count": response["event_count"],
        "latest_sequence": response["latest_sequence"],
        "latest_effective_status": response["latest_effective_status"],
        "revocation_count": response["revocation_count"],
        "terminal": response["terminal"],
    }


def _verify_pre_cut_time(
    context: tuple[str, str, str, str, Mapping[str, object], Mapping[str, object]],
    response: Mapping[str, object],
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    logical_time_request_payloads_in_exact_order: tuple[bytes, ...],
    logical_time_signed_token_payloads_in_exact_order: tuple[bytes, ...],
    logical_time_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
    logical_time_head_payloads_in_exact_order: tuple[bytes, ...],
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_payload_sha256_digest: bytes,
    expected_logical_time_event_plan: tuple[ExpectedLogicalEventCoreV1, ...],
    _verify=_verify_time_replay,
    _require=_require_time_replay,
    _time_error=_LogicalTimeReplayError,
) -> tuple[dict[str, object], object]:
    try:
        result = _verify(
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            logical_time_request_payloads_in_exact_order,
            logical_time_signed_token_payloads_in_exact_order,
            logical_time_checkpoint_payloads_in_exact_order,
            logical_time_head_payloads_in_exact_order,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_payload_sha256_digest
            ),
            expected_attempt_context_id=context[3],
            expected_event_plan=expected_logical_time_event_plan,
        )
        result = _require(result)
    except _time_error as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_LOGICAL_TIME_V1, "raw producer-free pre-cut time replay rejected"
        ) from exc
    if not (
        result.stage_b_trust_lock_id == context[0]
        and result.authority_profile_id == context[1]
        and result.signer_registry_id == context[2]
        and result.attempt_context_id == context[3]
        and result.verified_event_count == response["event_count"] == 1
        and result.zero_event_genesis_only is False
        and len(result.token_ids) == 1
        and result.token_ids[-1]
        and result.supplied_chain_final_head_verified is True
        and result.external_currentness_verified is False
        and result.external_anti_fork_verified is False
        and result.authority_evidence is False
        and result.r01_satisfied is False
        and result.readiness_evidence is False
        and result.execution_authorized is False
    ):
        _fail(
            FAIL_LOGICAL_TIME_V1,
            "pre-cut time authority/attempt/non-genesis/event-count boundary differs",
        )
    binding = {
        "ledger_id": result.ledger_id,
        "final_head_id": result.final_head_id,
        "final_head_payload_sha256": result.final_head_payload_sha256,
        "last_token_id": result.token_ids[-1],
        "next_sequence": result.verified_event_count,
    }
    if frozenset(binding) != PRE_CUT_LOGICAL_TIME_BINDING_FIELDS_V1:
        _fail(FAIL_SCHEMA_V1, "pre-cut time evidence binding keyset differs")
    return binding, result


def _local_observation_ids(
    socket_identity: tuple[int, int, int, int],
    peer: tuple[int, int, int],
    governance_binding_id: str,
    request_payload: bytes,
    response_payload: bytes,
) -> tuple[str, str]:
    endpoint_subject = {
        "schema_version": "hegel-machine-phase2b-step5-governance-socket-endpoint-local-observation/1",
        "address_family": "AF_UNIX",
        "socket_type": "SOCK_SEQPACKET",
        "held_st_dev_decimal": str(socket_identity[0]),
        "held_st_ino_decimal": str(socket_identity[1]),
        "peer_pid": peer[0],
        "peer_uid": peer[1],
        "peer_gid": peer[2],
        "governance_service_binding_id": governance_binding_id,
    }
    endpoint_id = _content_id(
        SOCKET_ENDPOINT_PREFIX_V1, _SOCKET_ENDPOINT_DOMAIN, endpoint_subject
    )
    runtime_subject = {
        "schema_version": "hegel-machine-phase2b-step5-governance-runtime-local-observation/1",
        "socket_endpoint_id": endpoint_id,
        "governance_service_binding_id": governance_binding_id,
        "peer_pid": peer[0],
        "peer_uid": peer[1],
        "peer_gid": peer[2],
        "request_payload_sha256": _payload_sha_hex(request_payload),
        "response_payload_sha256": _payload_sha_hex(response_payload),
    }
    runtime_id = _content_id(
        RUNTIME_OBSERVATION_PREFIX_V1,
        _RUNTIME_OBSERVATION_DOMAIN,
        runtime_subject,
    )
    return endpoint_id, runtime_id


def _build_statement(
    context: tuple[str, str, str, str, Mapping[str, object], Mapping[str, object]],
    request: Mapping[str, object],
    request_payload: bytes,
    response: Mapping[str, object],
    response_payload: bytes,
    peer: tuple[int, int, int],
    socket_identity: tuple[int, int, int, int],
    pre_cut_time_binding: Mapping[str, object],
    _canonical_evidence=_evidence_canonical,
    _content_id_evidence=_evidence_content_id,
) -> bytes:
    endpoint_id, runtime_id = _local_observation_ids(
        socket_identity,
        peer,
        context[5]["binding_id"],  # type: ignore[arg-type]
        request_payload,
        response_payload,
    )
    peer_observation = {
        "service_name": GOVERNANCE_SERVICE_NAME_V1,
        "service_binding_id": context[5]["binding_id"],
        "socket_endpoint_id": endpoint_id,
        "request_nonce": request["cut_request_nonce"],
        "request_payload_sha256": _payload_sha_hex(request_payload),
        "response_payload_sha256": _payload_sha_hex(response_payload),
        "peer_pid": peer[0],
        "peer_uid": peer[1],
        "peer_gid": peer[2],
        "runtime_instance_id": runtime_id,
    }
    owner = _owner_binding(response)
    if (
        frozenset(peer_observation) != GOVERNANCE_PEER_OBSERVATION_FIELDS_V1
        or frozenset(owner) != OWNER_STATUS_BINDING_FIELDS_V1
        or frozenset(pre_cut_time_binding) != PRE_CUT_LOGICAL_TIME_BINDING_FIELDS_V1
    ):
        _fail(FAIL_SCHEMA_V1, "statement nested evidence keyset differs")
    subject: dict[str, object] = {
        "schema_version": CURRENT_CUT_STATEMENT_SCHEMA_V1,
        "status": CANDIDATE_STATUS_V1,
        "claim_level": CANDIDATE_CLAIM_LEVEL_V1,
        "stage_b_trust_lock_id": context[0],
        "authority_profile_id": context[1],
        "signer_registry_id": context[2],
        "attempt_context_id": context[3],
        "current_cut_response_id": response["response_id"],
        "current_cut_response_payload_sha256": _payload_sha_hex(response_payload),
        "governance_peer_observation": peer_observation,
        "owner_status_binding": owner,
        "pre_cut_logical_time_binding": dict(pre_cut_time_binding),
    }
    body = {
        **subject,
        "statement_id": _content_id_evidence("current_cut_statement", subject),
    }
    if frozenset(body) != CURRENT_CUT_STATEMENT_FIELDS_V1:
        _fail(FAIL_SCHEMA_V1, "emitted evidence current-cut statement keyset differs")
    try:
        payload = _canonical_evidence(body)
    except Exception as exc:
        raise Phase2BStep5R01CurrentCutErrorV1(
            FAIL_CANONICAL_V1, "evidence statement canonicalization rejected"
        ) from exc
    if type(payload) is not bytes or len(payload) > _MAX_RESPONSE_BYTES:
        _fail(FAIL_CANONICAL_V1, "evidence statement byte bound differs")
    return payload


def _load_statement(
    payload: bytes,
    context: tuple[str, str, str, str, Mapping[str, object], Mapping[str, object]],
    request: Mapping[str, object],
    request_payload: bytes,
    response: Mapping[str, object],
    response_payload: bytes,
    peer: tuple[int, int, int],
    pre_cut_time_binding: Mapping[str, object],
    _content_id_evidence=_evidence_content_id,
) -> dict[str, object]:
    body = _object(
        _decode(payload, _MAX_RESPONSE_BYTES, "current-cut statement"),
        CURRENT_CUT_STATEMENT_FIELDS_V1,
        "current-cut statement",
    )
    if not (
        body["schema_version"] == CURRENT_CUT_STATEMENT_SCHEMA_V1
        and body["status"] == CANDIDATE_STATUS_V1
        and body["claim_level"] == CANDIDATE_CLAIM_LEVEL_V1
        and (
            body["stage_b_trust_lock_id"],
            body["authority_profile_id"],
            body["signer_registry_id"],
            body["attempt_context_id"],
        )
        == context[:4]
        and body["current_cut_response_id"] == response["response_id"]
        and body["current_cut_response_payload_sha256"]
        == _payload_sha_hex(response_payload)
        and _exact_mapping_equal(body["owner_status_binding"], _owner_binding(response))
        and _exact_mapping_equal(body["pre_cut_logical_time_binding"], pre_cut_time_binding)
    ):
        _fail(FAIL_BINDING_V1, "current-cut statement lineage differs")
    peer_body = _object(
        body["governance_peer_observation"],
        GOVERNANCE_PEER_OBSERVATION_FIELDS_V1,
        "governance peer observation",
    )
    if not (
        peer_body["service_name"] == GOVERNANCE_SERVICE_NAME_V1
        and peer_body["service_binding_id"] == context[5]["binding_id"]
        and peer_body["request_nonce"] == request["cut_request_nonce"]
        and peer_body["request_payload_sha256"] == _payload_sha_hex(request_payload)
        and peer_body["response_payload_sha256"] == _payload_sha_hex(response_payload)
        and (peer_body["peer_pid"], peer_body["peer_uid"], peer_body["peer_gid"])
        == peer
    ):
        _fail(FAIL_BINDING_V1, "governance peer observation lineage differs")
    _identifier(peer_body["socket_endpoint_id"], "local socket endpoint observation")
    _identifier(peer_body["runtime_instance_id"], "local runtime observation")
    expected_id = _content_id_evidence(
        "current_cut_statement", _without(body, "statement_id")
    )
    if body["statement_id"] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "current-cut statement ID differs")
    return body


@dataclass(frozen=True, slots=True, init=False)
class VerifiedR01CurrentCutMechanicsV1:
    """Privately issued candidate mechanics result; never currentness/R01."""

    schema_version: str
    claim_boundary: str
    stage_b_trust_lock_id: str
    authority_profile_id: str
    signer_registry_id: str
    attempt_context_id: str
    requester_service_binding_id: str
    governance_service_binding_id: str
    request_id: str
    response_id: str
    statement_id: str
    request_payload_sha256: str
    response_payload_sha256: str
    statement_payload_sha256: str
    owner_status_ledger_id: str
    owner_current_head_id: str
    owner_projection_id: str
    complete_event_set_root: str
    pre_cut_logical_time_ledger_id: str
    pre_cut_logical_time_final_head_id: str
    pre_cut_logical_time_last_token_id: str
    peer_pid: int
    peer_uid: int
    peer_gid: int
    socket_endpoint_observation_id: str
    local_runtime_observation_id: str
    request_payload: bytes
    response_payload: bytes
    statement_payload: bytes
    raw_authority_v2_graph_verified: bool
    kernel_so_peercred_verified: bool
    seqpacket_request_response_verified: bool
    owner_projection_candidate_presented: bool
    stable_during_three_replays: bool
    exact_producer_append_projection_presented: bool
    active_zero_revocation_candidate: bool
    pre_cut_logical_time_supplied_chain_verified: bool
    owner_time_event_count_cross_checked: bool
    owner_time_raw_decision_binding_verified: bool
    response_time_currentness_verified: bool
    admission_time_currentness_verified: bool
    atomic_owner_time_fence_verified: bool
    external_currentness_verified: bool
    external_anti_fork_anchor_verified: bool
    live_runtime_closure_verified: bool
    live_distinct_service_uids_gids_verified: bool
    live_distinct_control_domains_verified: bool
    service_key_custody_verified: bool
    cas_durability_verified: bool
    owner_signature_authenticated: bool
    authority_evidence: bool
    r01_satisfied: bool
    readiness_evidence: bool
    execution_authorized: bool

    def __new__(cls, *args: object, **kwargs: object) -> "VerifiedR01CurrentCutMechanicsV1":
        raise TypeError("verified current-cut mechanics results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified current-cut mechanics results are privately issued")


def _build_result_lineage():
    result_type = VerifiedR01CurrentCutMechanicsV1
    fields = tuple(result_type.__dataclass_fields__)
    issued: dict[int, tuple[VerifiedR01CurrentCutMechanicsV1, tuple[object, ...]]] = {}
    error_type = Phase2BStep5R01CurrentCutErrorV1
    fail_code = FAIL_RESULT_V1
    result_schema = CURRENT_CUT_MECHANICS_RESULT_SCHEMA_V1
    claim_boundary = CURRENT_CUT_MECHANICS_CLAIM_BOUNDARY_V1
    sha256 = hashlib.sha256

    def reject(detail: str) -> NoReturn:
        raise error_type(fail_code, detail)

    def payload_sha_hex(value: object) -> str | None:
        return sha256(value).hexdigest() if type(value) is bytes else None

    def require(value: object) -> VerifiedR01CurrentCutMechanicsV1:
        if type(value) is not result_type:
            reject("current-cut result exact type differs")
        record = issued.get(id(value))
        if record is None or record[0] is not value:
            reject("current-cut result lacks private issuance lineage")
        expected = record[1]
        try:
            observed = tuple(object.__getattribute__(value, field) for field in fields)
        except (AttributeError, TypeError):
            observed = ()
        if not (
            len(observed) == len(expected)
            and all(type(a) is type(b) and a == b for a, b in zip(observed, expected, strict=True))
        ):
            reject("current-cut result was mutated")
        true_fields = (
            "raw_authority_v2_graph_verified",
            "kernel_so_peercred_verified",
            "seqpacket_request_response_verified",
            "owner_projection_candidate_presented",
            "active_zero_revocation_candidate",
            "pre_cut_logical_time_supplied_chain_verified",
            "owner_time_event_count_cross_checked",
        )
        false_fields = (
            "owner_time_raw_decision_binding_verified",
            "response_time_currentness_verified",
            "admission_time_currentness_verified",
            "atomic_owner_time_fence_verified",
            "external_currentness_verified",
            "external_anti_fork_anchor_verified",
            "live_runtime_closure_verified",
            "live_distinct_service_uids_gids_verified",
            "live_distinct_control_domains_verified",
            "service_key_custody_verified",
            "cas_durability_verified",
            "owner_signature_authenticated",
            "authority_evidence",
            "r01_satisfied",
            "readiness_evidence",
            "execution_authorized",
        )
        if not (
            value.schema_version == result_schema
            and value.claim_boundary == claim_boundary
            and all(object.__getattribute__(value, name) is True for name in true_fields)
            and all(object.__getattribute__(value, name) is False for name in false_fields)
            # The evidence response shape does not carry the server's owner
            # source mode.  The client must not turn a caller assertion into
            # evidence for either the three-replay or append-result path.
            and value.stable_during_three_replays is False
            and value.exact_producer_append_projection_presented is False
            and value.peer_uid == 41005
            and value.peer_gid == 42005
            and payload_sha_hex(value.request_payload) == value.request_payload_sha256
            and payload_sha_hex(value.response_payload) == value.response_payload_sha256
            and payload_sha_hex(value.statement_payload) == value.statement_payload_sha256
        ):
            reject("current-cut result semantics differ")
        return value

    def issue(**values: object) -> VerifiedR01CurrentCutMechanicsV1:
        if frozenset(values) != frozenset(fields):
            reject("private current-cut issuance keyset differs")
        result = object.__new__(result_type)
        for name in fields:
            object.__setattr__(result, name, values[name])
        snapshot = tuple(values[name] for name in fields)
        issued[id(result)] = (result, snapshot)
        return require(result)

    return issue, require


_issue_result, require_verified_r01_current_cut_mechanics_v1 = _build_result_lineage()
del _build_result_lineage


def _build_current_cut_request_implementation_v1(
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_payload_sha256_digest: bytes,
    expected_attempt_context_id: str,
    cut_request_nonce_bytes: bytes,
) -> bytes:
    context = _load_authority_context(
        stage_b_trust_lock_payload,
        authority_profile_payload,
        signer_registry_payload,
        out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
        out_of_band_expected_payload_sha256_digest=(
            out_of_band_expected_payload_sha256_digest
        ),
        expected_attempt_context_id=expected_attempt_context_id,
    )
    return _build_request_from_context(context, cut_request_nonce_bytes)


def _process_one_current_cut_request_implementation_v1(
    connected_socket: socket.socket,
    owner_status_source: object,
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_payload_sha256_digest: bytes,
    expected_attempt_context_id: str,
) -> bytes:
    held_socket = _require_seqpacket(connected_socket)
    context = _load_authority_context(
        stage_b_trust_lock_payload,
        authority_profile_payload,
        signer_registry_payload,
        out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
        out_of_band_expected_payload_sha256_digest=(
            out_of_band_expected_payload_sha256_digest
        ),
        expected_attempt_context_id=expected_attempt_context_id,
    )
    socket_before = _socket_identity(held_socket)
    peer_before = _peer_credentials(held_socket)
    _require_peer_binding(peer_before, context[4], "requester runner")
    request_payload = _recv_packet(held_socket, _MAX_REQUEST_BYTES, "current-cut request")
    peer_after_receive = _peer_credentials(held_socket)
    socket_after_receive = _socket_identity(held_socket)
    if peer_after_receive != peer_before or socket_after_receive != socket_before:
        _fail(FAIL_SOCKET_V1, "requester peer credentials or held socket changed")
    _require_peer_binding(peer_after_receive, context[4], "requester runner")
    request = _load_request(request_payload, context)
    head_payload, projection_payload, head, projection, _stable = _owner_snapshot(
        owner_status_source, context
    )
    response_payload = _build_response(
        request, context, head_payload, projection_payload, head, projection
    )
    _send_packet(held_socket, response_payload, "current-cut response")
    peer_after_send = _peer_credentials(held_socket)
    socket_after_send = _socket_identity(held_socket)
    if peer_after_send != peer_before or socket_after_send != socket_before:
        _fail(FAIL_SOCKET_V1, "requester peer credentials or held socket changed after send")
    _require_peer_binding(peer_after_send, context[4], "requester runner")
    return response_payload


def _request_current_cut_over_seqpacket_implementation_v1(
    connected_socket: socket.socket,
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    logical_time_request_payloads_in_exact_order: tuple[bytes, ...],
    logical_time_signed_token_payloads_in_exact_order: tuple[bytes, ...],
    logical_time_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
    logical_time_head_payloads_in_exact_order: tuple[bytes, ...],
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_payload_sha256_digest: bytes,
    expected_attempt_context_id: str,
    expected_logical_time_event_plan: tuple[ExpectedLogicalEventCoreV1, ...],
    cut_request_nonce_bytes: bytes,
    _issue=_issue_result,
) -> VerifiedR01CurrentCutMechanicsV1:
    held_socket = _require_seqpacket(connected_socket)
    context = _load_authority_context(
        stage_b_trust_lock_payload,
        authority_profile_payload,
        signer_registry_payload,
        out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
        out_of_band_expected_payload_sha256_digest=(
            out_of_band_expected_payload_sha256_digest
        ),
        expected_attempt_context_id=expected_attempt_context_id,
    )
    request_payload = _build_request_from_context(context, cut_request_nonce_bytes)
    request = _load_request(request_payload, context)
    socket_before = _socket_identity(held_socket)
    peer_before = _peer_credentials(held_socket)
    _require_peer_binding(peer_before, context[5], "governance service")
    _send_packet(held_socket, request_payload, "current-cut request")
    response_payload = _recv_packet(
        held_socket, _MAX_RESPONSE_BYTES, "current-cut response"
    )
    peer_after = _peer_credentials(held_socket)
    socket_after = _socket_identity(held_socket)
    if peer_after != peer_before or socket_after != socket_before:
        _fail(FAIL_SOCKET_V1, "governance peer credentials or held socket changed")
    _require_peer_binding(peer_after, context[5], "governance service")
    response = _load_response(response_payload, request, context)
    pre_cut_binding, _time_result = _verify_pre_cut_time(
        context,
        response,
        stage_b_trust_lock_payload,
        authority_profile_payload,
        signer_registry_payload,
        logical_time_request_payloads_in_exact_order,
        logical_time_signed_token_payloads_in_exact_order,
        logical_time_checkpoint_payloads_in_exact_order,
        logical_time_head_payloads_in_exact_order,
        out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
        out_of_band_expected_payload_sha256_digest=(
            out_of_band_expected_payload_sha256_digest
        ),
        expected_logical_time_event_plan=expected_logical_time_event_plan,
    )
    statement_payload = _build_statement(
        context,
        request,
        request_payload,
        response,
        response_payload,
        peer_after,
        socket_after,
        pre_cut_binding,
    )
    statement = _load_statement(
        statement_payload,
        context,
        request,
        request_payload,
        response,
        response_payload,
        peer_after,
        pre_cut_binding,
    )
    peer_statement = statement["governance_peer_observation"]
    if type(peer_statement) is not dict:
        _fail(FAIL_RESULT_V1, "issued statement peer observation type differs")
    return _issue(
        schema_version=CURRENT_CUT_MECHANICS_RESULT_SCHEMA_V1,
        claim_boundary=CURRENT_CUT_MECHANICS_CLAIM_BOUNDARY_V1,
        stage_b_trust_lock_id=context[0],
        authority_profile_id=context[1],
        signer_registry_id=context[2],
        attempt_context_id=context[3],
        requester_service_binding_id=context[4]["binding_id"],
        governance_service_binding_id=context[5]["binding_id"],
        request_id=request["request_id"],
        response_id=response["response_id"],
        statement_id=statement["statement_id"],
        request_payload_sha256=_payload_sha_hex(request_payload),
        response_payload_sha256=_payload_sha_hex(response_payload),
        statement_payload_sha256=_payload_sha_hex(statement_payload),
        owner_status_ledger_id=response["owner_status_ledger_id"],
        owner_current_head_id=response["current_head_id"],
        owner_projection_id=response["projection_id"],
        complete_event_set_root=response["complete_event_set_root"],
        pre_cut_logical_time_ledger_id=pre_cut_binding["ledger_id"],
        pre_cut_logical_time_final_head_id=pre_cut_binding["final_head_id"],
        pre_cut_logical_time_last_token_id=pre_cut_binding["last_token_id"],
        peer_pid=peer_after[0],
        peer_uid=peer_after[1],
        peer_gid=peer_after[2],
        socket_endpoint_observation_id=peer_statement["socket_endpoint_id"],
        local_runtime_observation_id=peer_statement["runtime_instance_id"],
        request_payload=request_payload,
        response_payload=response_payload,
        statement_payload=statement_payload,
        raw_authority_v2_graph_verified=True,
        kernel_so_peercred_verified=True,
        seqpacket_request_response_verified=True,
        owner_projection_candidate_presented=True,
        stable_during_three_replays=False,
        exact_producer_append_projection_presented=False,
        active_zero_revocation_candidate=True,
        pre_cut_logical_time_supplied_chain_verified=True,
        owner_time_event_count_cross_checked=True,
        owner_time_raw_decision_binding_verified=False,
        response_time_currentness_verified=False,
        admission_time_currentness_verified=False,
        atomic_owner_time_fence_verified=False,
        external_currentness_verified=False,
        external_anti_fork_anchor_verified=False,
        live_runtime_closure_verified=False,
        live_distinct_service_uids_gids_verified=False,
        live_distinct_control_domains_verified=False,
        service_key_custody_verified=False,
        cas_durability_verified=False,
        owner_signature_authenticated=False,
        authority_evidence=False,
        r01_satisfied=False,
        readiness_evidence=False,
        execution_authorized=False,
    )


def verify_real_current_cut_admission_v1(*args: object, **kwargs: object) -> NoReturn:
    """Fail closed: mechanics cannot issue a real current-cut admission."""

    _fail(
        FAIL_NOT_IMPLEMENTED_V1,
        "real current-cut admission requires a fresh admission-time requery/fence, "
        "external anti-fork evidence, main-CAS replay, and later R01 gates",
    )


def _build_public_entries_v1(
    build_implementation: object,
    process_implementation: object,
    request_implementation: object,
):
    """Hide captured issuers/helpers behind exact raw/OOB public signatures."""

    if not all(
        type(value) is FunctionType
        for value in (
            build_implementation,
            process_implementation,
            request_implementation,
        )
    ):
        _fail(FAIL_RESULT_V1, "captured public implementation type differs")

    def build_current_cut_request_v1(
        stage_b_trust_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        *,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_payload_sha256_digest: bytes,
        expected_attempt_context_id: str,
        cut_request_nonce_bytes: bytes,
    ) -> bytes:
        return build_implementation(
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_payload_sha256_digest
            ),
            expected_attempt_context_id=expected_attempt_context_id,
            cut_request_nonce_bytes=cut_request_nonce_bytes,
        )

    def process_one_current_cut_request_v1(
        connected_socket: socket.socket,
        owner_status_source: object,
        stage_b_trust_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        *,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_payload_sha256_digest: bytes,
        expected_attempt_context_id: str,
    ) -> bytes:
        return process_implementation(
            connected_socket,
            owner_status_source,
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_payload_sha256_digest
            ),
            expected_attempt_context_id=expected_attempt_context_id,
        )

    def request_current_cut_over_seqpacket_v1(
        connected_socket: socket.socket,
        stage_b_trust_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        logical_time_request_payloads_in_exact_order: tuple[bytes, ...],
        logical_time_signed_token_payloads_in_exact_order: tuple[bytes, ...],
        logical_time_checkpoint_payloads_in_exact_order: tuple[bytes, ...],
        logical_time_head_payloads_in_exact_order: tuple[bytes, ...],
        *,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_payload_sha256_digest: bytes,
        expected_attempt_context_id: str,
        expected_logical_time_event_plan: tuple[ExpectedLogicalEventCoreV1, ...],
        cut_request_nonce_bytes: bytes,
    ) -> VerifiedR01CurrentCutMechanicsV1:
        return request_implementation(
            connected_socket,
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            logical_time_request_payloads_in_exact_order,
            logical_time_signed_token_payloads_in_exact_order,
            logical_time_checkpoint_payloads_in_exact_order,
            logical_time_head_payloads_in_exact_order,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_payload_sha256_digest
            ),
            expected_attempt_context_id=expected_attempt_context_id,
            expected_logical_time_event_plan=expected_logical_time_event_plan,
            cut_request_nonce_bytes=cut_request_nonce_bytes,
        )

    return (
        build_current_cut_request_v1,
        process_one_current_cut_request_v1,
        request_current_cut_over_seqpacket_v1,
    )


def _definition_captured_function_graph_v1() -> dict[str, object]:
    """Clone the module call graph and function-valued defaults at definition cut."""

    module_namespace = globals()
    namespace = dict(module_namespace)
    originals: dict[str, FunctionType] = {
        name: value
        for name, value in module_namespace.items()
        if type(value) is FunctionType
        and value.__module__ == __name__
        and name != "_definition_captured_function_graph_v1"
    }
    provisional: dict[str, FunctionType] = {}
    by_identity: dict[int, FunctionType] = {}
    for name, function in originals.items():
        clone = FunctionType(
            function.__code__,
            namespace,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        provisional[name] = clone
        by_identity[id(function)] = clone
    namespace.update(provisional)

    def captured(value: object) -> object:
        return by_identity.get(id(value), value)

    clones: dict[str, FunctionType] = {}
    for name, function in originals.items():
        defaults = function.__defaults__
        clone = FunctionType(
            function.__code__,
            namespace,
            function.__name__,
            None if defaults is None else tuple(captured(value) for value in defaults),
            function.__closure__,
        )
        clone.__kwdefaults__ = (
            None
            if function.__kwdefaults__ is None
            else {key: captured(value) for key, value in function.__kwdefaults__.items()}
        )
        clone.__annotations__ = dict(function.__annotations__)
        clone.__doc__ = function.__doc__
        clone.__module__ = function.__module__
        clone.__qualname__ = function.__qualname__
        clones[name] = clone
    final_by_identity = {id(originals[name]): clone for name, clone in clones.items()}

    def final(value: object) -> object:
        return final_by_identity.get(id(value), value)

    for name, function in originals.items():
        clone = clones[name]
        defaults = function.__defaults__
        clone.__defaults__ = (
            None if defaults is None else tuple(final(value) for value in defaults)
        )
        clone.__kwdefaults__ = (
            None
            if function.__kwdefaults__ is None
            else {key: final(value) for key, value in function.__kwdefaults__.items()}
        )
    namespace.update(clones)
    return clones


_CAPTURED = _definition_captured_function_graph_v1()
(
    build_current_cut_request_v1,
    process_one_current_cut_request_v1,
    request_current_cut_over_seqpacket_v1,
) = _build_public_entries_v1(
    _CAPTURED["_build_current_cut_request_implementation_v1"],
    _CAPTURED["_process_one_current_cut_request_implementation_v1"],
    _CAPTURED["_request_current_cut_over_seqpacket_implementation_v1"],
)
require_verified_r01_current_cut_mechanics_v1 = _CAPTURED[
    "require_verified_r01_current_cut_mechanics_v1"
]
verify_real_current_cut_admission_v1 = _CAPTURED[
    "verify_real_current_cut_admission_v1"
]
del _CAPTURED
del _definition_captured_function_graph_v1
del _build_public_entries_v1
del _build_current_cut_request_implementation_v1
del _process_one_current_cut_request_implementation_v1
del _request_current_cut_over_seqpacket_implementation_v1
del _issue_result


__all__ = [
    "AUTHORITY_V2_SOURCE_SHA256_V1",
    "OWNER_STATUS_SOURCE_SHA256_V1",
    "LOGICAL_TIME_REPLAY_CORE_SOURCE_SHA256_V1",
    "R01_EVIDENCE_CONTRACT_SOURCE_SHA256_V1",
    "CURRENT_CUT_REQUEST_SCHEMA_V1",
    "CURRENT_CUT_REQUEST_FIELDS_V1",
    "CURRENT_CUT_MECHANICS_RESULT_SCHEMA_V1",
    "CURRENT_CUT_REQUEST_CLAIM_LEVEL_V1",
    "CURRENT_CUT_MECHANICS_CLAIM_BOUNDARY_V1",
    "FAIL_INPUT_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_SCHEMA_V1",
    "FAIL_CONTENT_ID_V1",
    "FAIL_AUTHORITY_V1",
    "FAIL_BINDING_V1",
    "FAIL_SOCKET_V1",
    "FAIL_PEER_CREDENTIAL_V1",
    "FAIL_OWNER_SNAPSHOT_V1",
    "FAIL_LOGICAL_TIME_V1",
    "FAIL_CLAIM_BOUNDARY_V1",
    "FAIL_RESULT_V1",
    "FAIL_NOT_IMPLEMENTED_V1",
    "ExpectedLogicalEventCoreV1",
    "OwnerStatusLedgerV1",
    "OwnerStatusServiceAppendV1",
    "Phase2BStep5R01CurrentCutErrorV1",
    "VerifiedR01CurrentCutMechanicsV1",
    "build_current_cut_request_v1",
    "process_one_current_cut_request_v1",
    "request_current_cut_over_seqpacket_v1",
    "require_verified_r01_current_cut_mechanics_v1",
    "verify_real_current_cut_admission_v1",
]
