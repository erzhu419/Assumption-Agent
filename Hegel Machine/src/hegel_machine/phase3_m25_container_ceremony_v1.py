"""Fail-closed M2.5 container-ceremony foundation.

The module is the authority boundary between deterministic *candidate* bytes
and an owner-accepted, offline-container ceremony.  It deliberately does not
weaken :func:`phase3_m25_wire_v1.formal_content_root`: that public helper keeps
failing closed.  Candidate identities may be labelled formal only by
``promote_gate_evidence_v1`` after all ten Gate 15--24 predicates have been
replayed from real signatures and independently produced public receipts.

Two deliberately explicit extension points remain:

* both FD-3 calculators must emit the v2 response frozen in this file.  It
  contains the seed commitment and the six public split roots/counts, never
  row assignments or the seed;
* a committed public-basis bundle must supply exact static formal preimages.
  This module can already construct the document/profile and 480/85 typed-row
  portions from Commit-A bytes, but refuses arbitrary root-shaped fillers for
  the still-missing identifier/operator preimages.

No function in this file accepts a seed or private key through argv,
environment, stdin, JSON, or a public return value.  The marker helpers write
only public custody metadata outside the repository.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from types import MappingProxyType
from typing import Final, Mapping, NoReturn, Sequence

from .phase3_m25_bridge_full_dag_replay_v1 import (
    make_openssl_ed25519_verifier_v1,
)
from .phase3_container_actor_runtime_v1 import validate_qualification_report
from .phase3_m25_errata_qualification_v1 import (
    validate_dual_errata_qualification_report,
)
from .phase3_m25_external_v1 import (
    MarkerSnapshot,
    assert_public_payload_contains_no_secret_fields,
    validate_marker_snapshot,
    validate_secret_state_directory,
)
from .phase3_m25_rows_v1 import (
    generate_odd_role_rows_v1,
    generate_sink_role_rows_v1,
)
from .phase3_m25_wire_v1 import (
    AUDITED_PARENT_COMMIT_SHA1,
    M3_RUN_OUTPUT_ROOTS,
    OBJECT_TAGS,
    build_formal_object,
    candidate_content_root,
    decode_formal_object,
    encode_formal_object,
    external_signature_preimage_v1,
    git_sha1_commit_id,
    id_digest_v1,
    validate_actor_trust_bindings_v1,
    validate_bridge_attestation_bundle_v1,
    validate_execution_identity_linkage_v1,
    validate_external_input_attestation_bundle_v1,
    validate_hidden_access_ledger_genesis,
    validate_m3_output_roots_null,
    validate_opaque_id_registry_append_v1,
    validate_parent_absence_audit_bundle_v1,
)
from .strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
CONTAINER_PROFILE_PATH: Final = (
    PROJECT_ROOT / "config/phase3_container_actor_profile_v1.json"
)
CONTAINER_SECCOMP_PATH: Final = (
    PROJECT_ROOT / "config/phase3_internal_actor_seccomp_v1.json"
)
MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
CHILD_DSL_ID: Final = "hegel-old-dsl-v1.1.0"
AUTHORITY_CLASS: Final = "OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1"
TECHNICAL_ACTOR_DISCLOSURE_V1: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "same_admin_controller": True,
        "organizational_independence": False,
        "independent_human_actors": False,
        "technical_role_independence": True,
        "owner_accepted_threat_model": True,
        "remote_attestation": False,
        "hardware_key_nonexportability": False,
    }
)

CEREMONY_SCHEMA: Final = "hegel-phase3-m25-container-ceremony/1"
PUBLIC_BASIS_SCHEMA: Final = "hegel-phase3-m25-committed-public-basis/1"
GATE_REPORT_SCHEMA: Final = "hegel-phase3-m25-gates-15-24/1"
PUBLIC_ARTIFACT_KIND: Final = "FORMAL_PUBLIC_EVIDENCE_READY_NOT_RUN"

SPLIT_RESPONSE_SCHEMA_ID: Final = (
    b"hegel-phase3-split-calculator-fd3-response/2"
)
SPLIT_RESPONSE_VERSION: Final = 1
SPLIT_RESPONSE_MAX_FRAME_BYTES: Final = 1024
SPLIT_RESPONSE_ROWS: Final = (
    (1, 1, 192),
    (1, 2, 96),
    (1, 3, 192),
    (2, 1, 39),
    (2, 2, 20),
    (2, 3, 26),
)

ODD_UNIVERSE_ROOT: Final = bytes.fromhex(
    "b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05"
)
ODD_TRUTH_ROOT: Final = bytes.fromhex(
    "f5bbdc26bec62f9966e5ef31eaa800190ed52dedc73ee61545e0f9c122a1a506"
)
SINK_UNIVERSE_ROOT: Final = bytes.fromhex(
    "1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5"
)
SINK_TRUTH_ROOT: Final = bytes.fromhex(
    "9c0f5d75ea3c31f6cb1ea9917346a7a3f480ae9ce0ac0cb3bb21aac9d3bd7808"
)

GATE_NAMES: Final = MappingProxyType(
    {
        15: "SPLIT_SEED_FIRST_INSTANTIATION_SIGNED",
        16: "HIDDEN_ACCESS_LEDGER_GENESIS_ONLY",
        17: "PARENT_MANIFEST_ABSENCE_ATTESTED",
        18: "FORMAL_BINDING_MANIFESTS_CANONICALIZED",
        19: "FORMAL_SPEC_AND_REGISTRY_ROOTS_DUAL_EQUAL",
        20: "ODD_UNIVERSE_AND_TRUTH_ROOTS_DUAL_EQUAL",
        21: "SINK_UNIVERSE_AND_TRUTH_ROOTS_DUAL_EQUAL",
        22: "SPLIT_PARTITION_ROOTS_DUAL_EQUAL",
        23: "M3_STATE_AND_RECEIPT_WIRE_GOLDEN_TESTS_PASS",
        24: "M3_EXECUTION_MANIFEST_ROOT_NON_NULL_AND_15_OUTPUT_ROOTS_NULL",
    }
)

REQUIRED_STATIC_DUAL_ROOTS: Final = (
    "child_dsl_spec_root",
    "child_freeze_root",
    "operator_semantics_root",
    "identifier_registry_root",
    "canonical_ast_schema_root",
    "canonical_cbor_profile_root",
)

REQUIRED_CANONICAL_BINDING_OBJECTS: Final = (
    "NormativeApprovalManifestV1",
    "SplitBindingManifestV1",
    "CustodianBindingManifestV1",
    "SeedContinuityManifestV1",
    "DslShrinkTransitionFormalV1",
)

FAIL_CEREMONY_ELIGIBILITY: Final = "FAIL_M25_CONTAINER_ACTOR_NOT_ELIGIBLE"
FAIL_CEREMONY_BASIS_COMMIT: Final = "FAIL_M25_CEREMONY_BASIS_COMMIT_MISMATCH"
FAIL_CEREMONY_INPUT_UNCOMMITTED: Final = "FAIL_M25_CEREMONY_INPUT_UNCOMMITTED"
FAIL_PUBLIC_BASIS_INCOMPLETE: Final = "FAIL_M25_FORMAL_PUBLIC_BASIS_INCOMPLETE"
FAIL_PUBLIC_BASIS_REGISTRY_PREIMAGE: Final = (
    "FAIL_M25_FORMAL_PUBLIC_BASIS_REGISTRY_PREIMAGE_UNFROZEN"
)
FAIL_SPLIT_RESPONSE_FRAMING: Final = "FAIL_M25_SPLIT_FD5_RESPONSE_FRAMING"
FAIL_SPLIT_RESPONSE_SCHEMA: Final = "FAIL_M25_SPLIT_FD5_RESPONSE_SCHEMA"
FAIL_SPLIT_RESPONSE_SECRET_FIELD: Final = "FAIL_M25_SPLIT_FD5_SECRET_FIELD"
FAIL_SPLIT_FULL_ENDPOINT_REQUIRED: Final = (
    "FAIL_M25_SPLIT_FULL_ROOT_FD3_ENDPOINT_REQUIRED"
)
FAIL_MARKER_ATOMICITY: Final = "FAIL_M25_SPLIT_MARKER_ATOMICITY"
FAIL_MARKER_ALREADY_EXISTS: Final = "FAIL_SPLIT_SEED_ALREADY_INSTANTIATED"
FAIL_MARKER_RECOVERY_REQUIRED: Final = (
    "FAIL_SPLIT_SEED_PENDING_EXTERNAL_RECOVERY_REQUIRED"
)
FAIL_SIGNATURE_INVALID: Final = "FAIL_M25_EXTERNAL_SIGNATURE_INVALID"
FAIL_GATE_EVIDENCE: Final = "FAIL_M25_GATE_EVIDENCE_INVALID"
FAIL_GATE_NOT_ALL_PASS: Final = "FAIL_M25_GATES_15_24_NOT_ALL_PASS"
FAIL_FORMAL_PROMOTION_CONTEXT: Final = "FAIL_M25_FORMAL_PROMOTION_CONTEXT_INVALID"
FAIL_CONTAINER_INVOCATION_POLICY: Final = "FAIL_M25_CEREMONY_CONTAINER_POLICY"
FAIL_SECRET_TRANSPORT: Final = "FAIL_M25_CEREMONY_SECRET_TRANSPORT"


class M25ContainerCeremonyError(RuntimeError):
    """Stable fail-closed error for the owner-accepted ceremony boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise M25ContainerCeremonyError(code, detail)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_bytes(value: object, length: int, field: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        _fail(FAIL_GATE_EVIDENCE, f"{field} must be exactly {length} bytes")
    return value


def _require_lower_hex(value: object, length: int, field: str) -> str:
    if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        _fail(FAIL_GATE_EVIDENCE, f"{field} must be lowercase {length}-hex")
    return value


@dataclass(frozen=True, slots=True)
class ContainerActorInvocationV1:
    """One policy-complete Docker invocation with no secret transport."""

    purpose_id: int
    image_ref: str
    operation: str
    command: tuple[str, ...]
    environment: Mapping[str, str]
    stdin_payload: None = None


def validate_no_secret_transport_v1(
    *, argv: Sequence[str], environment: Mapping[str, str], stdin_payload: object
) -> None:
    """Forbid secret-bearing argv/env/stdin channels by construction."""

    if stdin_payload is not None:
        _fail(FAIL_SECRET_TRANSPORT, "ceremony actor stdin must be DEVNULL")
    forbidden_fragments = (
        "seed=",
        "seed_hex",
        "master_seed",
        "private_key",
        "private-key",
        "k_role",
        "derived_role",
    )
    lowered_argv = tuple(str(item).lower() for item in argv)
    if any(fragment in item for item in lowered_argv for fragment in forbidden_fragments):
        _fail(FAIL_SECRET_TRANSPORT, "ceremony argv contains a secret-bearing field")
    for key, value in environment.items():
        normalized_key = str(key).lower()
        normalized_value = str(value).lower()
        if any(fragment in normalized_key or fragment in normalized_value for fragment in forbidden_fragments):
            _fail(FAIL_SECRET_TRANSPORT, "ceremony environment contains a secret-bearing field")


ED25519_SPKI_DER_PREFIX: Final = bytes.fromhex("302a300506032b6570032100")


def parse_ed25519_spki_der_v1(payload: bytes) -> tuple[bytes, bytes]:
    """Parse the one allowed OpenSSL Ed25519 SubjectPublicKeyInfo form.

    Returns ``(raw_public_key, key_id)``.  No alternate ASN.1 encoding is
    accepted, keeping cross-container public-key identity byte-exact.
    """

    if (
        type(payload) is not bytes
        or len(payload) != len(ED25519_SPKI_DER_PREFIX) + 32
        or not payload.startswith(ED25519_SPKI_DER_PREFIX)
    ):
        _fail(FAIL_SIGNATURE_INVALID, "Ed25519 SPKI DER is not the exact 44-byte profile")
    public_key = payload[len(ED25519_SPKI_DER_PREFIX) :]
    key_id = hashlib.sha256(public_key).digest()[:16]
    return public_key, key_id


def build_actor_key_manifest_fields_v1(
    *,
    purpose_id: int,
    public_key: bytes,
    created_at_unix_seconds: int,
    basis_commit: str,
) -> dict[str, object]:
    """Build one epoch-0 actor-key manifest from a container public output."""

    _require_lower_hex(basis_commit, 40, "basis_commit")
    _require_bytes(public_key, 32, "actor public key")
    if purpose_id not in {1, 2, 3, 4}:
        _fail(FAIL_GATE_EVIDENCE, "pre-M4 actor key purpose must be 1,2,3,4")
    fields: dict[str, object] = {
        "purpose_id": purpose_id,
        "key_id": hashlib.sha256(public_key).digest()[:16],
        "public_key_32_bytes": public_key,
        "key_epoch": 0,
        "valid_from_unix_seconds": created_at_unix_seconds,
        "valid_until_unix_seconds_or_null": None,
        "repository_commit_id": git_sha1_commit_id(bytes.fromhex(basis_commit)),
    }
    build_formal_object("ActorKeyManifestV1", fields)
    return fields


def build_single_signature_envelope_fields_v1(
    *,
    enclosed_object_tag: int,
    enclosed_manifest_root: bytes,
    created_at_unix_seconds: int,
    signer_key_id: bytes,
    signature: bytes,
) -> dict[str, object]:
    """Build one epoch-0 envelope from a purpose-container public signature."""

    fields: dict[str, object] = {
        "enclosed_object_tag": enclosed_object_tag,
        "enclosed_manifest_root": _require_bytes(
            enclosed_manifest_root, 32, "enclosed manifest root"
        ),
        "created_at_unix_seconds": created_at_unix_seconds,
        "signer_key_epoch": 0,
        "signatures": (
            (
                _require_bytes(signer_key_id, 16, "signer key ID"),
                _require_bytes(signature, 64, "Ed25519 signature"),
            ),
        ),
    }
    build_formal_object("SignedManifestEnvelopeV1", fields)
    return fields


def build_offline_actor_invocation_v1(
    *,
    purpose_id: int,
    operation: str,
    read_only_input_directory: Path,
    private_state_volume: str,
    public_output_directory: Path,
    entrypoint: str,
) -> ContainerActorInvocationV1:
    """Build, but do not execute, one hardened purpose-container command.

    The state mount is a purpose-private Docker volume outside the repository;
    the public input is read-only and output is the only bind-mounted writable
    directory.  The command always uses local digest refs, ``--pull=never`` and
    ``--network=none``.  No registry/build operation exists in this API.
    """

    if purpose_id not in {1, 2, 3, 4}:
        _fail(FAIL_CONTAINER_INVOCATION_POLICY, "purpose must be 1,2,3,4")
    if operation not in {"keygen", "sign", "verify", "split-fd3"}:
        _fail(FAIL_CONTAINER_INVOCATION_POLICY, "actor operation is not allowlisted")
    if (
        not isinstance(private_state_volume, str)
        or re.fullmatch(r"hegel-m25-purpose-[1-4]-[0-9a-f]{32}", private_state_volume)
        is None
        or not isinstance(entrypoint, str)
        or not entrypoint.startswith("/input/")
        or ".." in Path(entrypoint).parts
    ):
        _fail(FAIL_CONTAINER_INVOCATION_POLICY, "volume or entrypoint identity is invalid")
    try:
        input_path = read_only_input_directory.resolve(strict=True)
        output_path = public_output_directory.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_CONTAINER_INVOCATION_POLICY, f"actor mount path is invalid: {exc}")
    if not input_path.is_dir() or not output_path.is_dir():
        _fail(FAIL_CONTAINER_INVOCATION_POLICY, "actor input/output mounts must be directories")
    if input_path == output_path:
        _fail(FAIL_CONTAINER_INVOCATION_POLICY, "actor input and output mounts must differ")
    try:
        profile = json.loads(CONTAINER_PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(FAIL_CONTAINER_INVOCATION_POLICY, f"container profile is invalid: {exc}")
    role_by_purpose = {
        1: "custodian",
        2: "python_attester",
        3: "rust_attester",
        4: "policy_auditor",
    }
    image_ref = profile.get("images", {}).get(role_by_purpose[purpose_id])
    if (
        type(image_ref) is not str
        or re.fullmatch(r"[a-z0-9._/-]+@sha256:[0-9a-f]{64}", image_ref) is None
    ):
        _fail(FAIL_CONTAINER_INVOCATION_POLICY, "actor image is not digest pinned")
    command = (
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--security-opt=seccomp={CONTAINER_SECCOMP_PATH.resolve()}",
        "--user=65534:65534",
        "--pids-limit=64",
        "--memory=512m",
        "--memory-swap=512m",
        "--ulimit=nofile=64:64",
        "--ipc=private",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=0700",
        f"--mount=type=bind,src={input_path},dst=/input,readonly",
        f"--mount=type=volume,src={private_state_volume},dst=/state",
        f"--mount=type=bind,src={output_path},dst=/output",
        "--env=LC_ALL=C.UTF-8",
        "--env=PATH=/usr/local/bin:/usr/bin:/bin",
        "--env=HEGEL_ACTOR_PROFILE_ID=hegel-owner-accepted-container-technical-actors-v1",
        f"--env=HEGEL_PURPOSE_ID={purpose_id}",
        "--entrypoint",
        entrypoint,
        image_ref,
        operation,
    )
    environment = MappingProxyType(
        {
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HEGEL_ACTOR_PROFILE_ID": (
                "hegel-owner-accepted-container-technical-actors-v1"
            ),
            "HEGEL_PURPOSE_ID": str(purpose_id),
        }
    )
    validate_no_secret_transport_v1(
        argv=command, environment=environment, stdin_payload=None
    )
    return ContainerActorInvocationV1(
        purpose_id=purpose_id,
        image_ref=image_ref,
        operation=operation,
        command=command,
        environment=environment,
    )


def _run_git(args: Sequence[str], *, binary: bool = True) -> bytes | str:
    completed = subprocess.run(
        ["/usr/bin/git", *args],
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_SSH_COMMAND": "false",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace")[-1000:]
        _fail(FAIL_CEREMONY_BASIS_COMMIT, f"git {' '.join(args)} failed: {detail}")
    if binary:
        return completed.stdout
    return completed.stdout.decode("ascii", "strict").strip()


def _repository_relative(path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except (OSError, ValueError) as exc:
        _fail(FAIL_CEREMONY_INPUT_UNCOMMITTED, f"input escapes repository: {path}: {exc}")


def _git_blob_bytes(commit: str, path: Path) -> bytes:
    relative = _repository_relative(path)
    raw = _run_git(["show", f"{commit}:{relative}"], binary=True)
    assert isinstance(raw, bytes)
    return raw


def _assert_worktree_blob_equals_commit(commit: str, path: Path) -> bytes:
    committed = _git_blob_bytes(commit, path)
    try:
        worktree = path.read_bytes()
    except OSError as exc:
        _fail(FAIL_CEREMONY_INPUT_UNCOMMITTED, f"cannot read ceremony input {path}: {exc}")
    if committed != worktree:
        _fail(
            FAIL_CEREMONY_INPUT_UNCOMMITTED,
            f"ceremony input differs from basis commit: {_repository_relative(path)}",
        )
    return committed


@dataclass(frozen=True, slots=True)
class SplitRootCommitment:
    """One public role/partition count and RFC6962 root."""

    role_id: int
    partition_id: int
    row_count: int
    root: bytes

    def formal_row(self) -> tuple[object, ...]:
        return (self.role_id, self.partition_id, self.row_count, self.root)


@dataclass(frozen=True, slots=True)
class SplitCalculatorPublicResponseV2:
    """The only accepted public result from a full FD-3 split calculator."""

    seed_commitment: bytes
    partitions: tuple[SplitRootCommitment, ...]

    def validate(self) -> "SplitCalculatorPublicResponseV2":
        _require_bytes(self.seed_commitment, 32, "seed commitment")
        actual = tuple(
            (row.role_id, row.partition_id, row.row_count) for row in self.partitions
        )
        if actual != SPLIT_RESPONSE_ROWS:
            _fail(
                FAIL_SPLIT_RESPONSE_SCHEMA,
                "split response role/partition/count registry differs from the freeze",
            )
        if any(type(row.root) is not bytes or len(row.root) != 32 for row in self.partitions):
            _fail(FAIL_SPLIT_RESPONSE_SCHEMA, "every split root must be 32 bytes")
        if len({(row.role_id, row.partition_id) for row in self.partitions}) != 6:
            _fail(FAIL_SPLIT_RESPONSE_SCHEMA, "split response partitions repeat")
        return self

    @property
    def roots(self) -> Mapping[str, bytes]:
        names = (
            "outside_discovery_split_root",
            "outside_validation_split_root",
            "outside_sealed_split_root",
            "null_discovery_split_root",
            "null_validation_split_root",
            "null_sealed_split_root",
        )
        self.validate()
        return MappingProxyType(
            {name: row.root for name, row in zip(names, self.partitions, strict=True)}
        )


def encode_split_calculator_public_payload_v2(
    response: SplitCalculatorPublicResponseV2,
) -> bytes:
    """Encode the exact public payload (without the uint64 frame length)."""

    response.validate()
    value = (
        SPLIT_RESPONSE_VERSION,
        SPLIT_RESPONSE_SCHEMA_ID,
        response.seed_commitment,
        tuple(row.formal_row() for row in response.partitions),
    )
    payload = canonical_cbor_encode(value)
    if len(payload) > SPLIT_RESPONSE_MAX_FRAME_BYTES:
        _fail(FAIL_SPLIT_RESPONSE_FRAMING, "split response exceeds the frozen limit")
    return payload


def encode_split_calculator_public_frame_v2(
    response: SplitCalculatorPublicResponseV2,
) -> bytes:
    payload = encode_split_calculator_public_payload_v2(response)
    return len(payload).to_bytes(8, "big") + payload


def decode_split_calculator_public_frame_v2(
    frame: bytes,
) -> SplitCalculatorPublicResponseV2:
    """Strictly decode one complete FD-5 length-prefixed response."""

    if type(frame) is not bytes or len(frame) < 9:
        _fail(FAIL_SPLIT_RESPONSE_FRAMING, "FD-5 response frame is truncated")
    length = int.from_bytes(frame[:8], "big")
    payload = frame[8:]
    if length != len(payload) or not 1 <= length <= SPLIT_RESPONSE_MAX_FRAME_BYTES:
        _fail(FAIL_SPLIT_RESPONSE_FRAMING, "FD-5 length prefix is not exact")
    try:
        value = canonical_cbor_decode(payload)
    except Exception as exc:
        _fail(FAIL_SPLIT_RESPONSE_SCHEMA, f"FD-5 payload is not strict CBOR: {exc}")
    if (
        not isinstance(value, tuple)
        or len(value) != 4
        or value[0] != SPLIT_RESPONSE_VERSION
        or value[1] != SPLIT_RESPONSE_SCHEMA_ID
        or type(value[2]) is not bytes
        or not isinstance(value[3], tuple)
        or len(value[3]) != 6
    ):
        _fail(FAIL_SPLIT_RESPONSE_SCHEMA, "FD-5 response prefix/arity differs")
    rows: list[SplitRootCommitment] = []
    for index, raw in enumerate(value[3]):
        if (
            not isinstance(raw, tuple)
            or len(raw) != 4
            or any(type(item) is not int for item in raw[:3])
            or type(raw[3]) is not bytes
        ):
            _fail(FAIL_SPLIT_RESPONSE_SCHEMA, f"FD-5 partition row {index} is invalid")
        rows.append(SplitRootCommitment(raw[0], raw[1], raw[2], raw[3]))
    result = SplitCalculatorPublicResponseV2(value[2], tuple(rows)).validate()
    if encode_split_calculator_public_frame_v2(result) != frame:
        _fail(FAIL_SPLIT_RESPONSE_FRAMING, "FD-5 response does not round-trip exactly")
    return result


def require_full_split_response_agreement_v2(
    python_frame: bytes,
    rust_frame: bytes,
) -> SplitCalculatorPublicResponseV2:
    """Require byte-identical Python/Rust v2 responses.

    A legacy commitment-only response is rejected with the dedicated endpoint
    gap code rather than being silently upgraded to Gate-22 evidence.
    """

    try:
        python_response = decode_split_calculator_public_frame_v2(python_frame)
        rust_response = decode_split_calculator_public_frame_v2(rust_frame)
    except M25ContainerCeremonyError as exc:
        if exc.code in {FAIL_SPLIT_RESPONSE_FRAMING, FAIL_SPLIT_RESPONSE_SCHEMA}:
            _fail(
                FAIL_SPLIT_FULL_ENDPOINT_REQUIRED,
                "both calculators must implement commitment plus six split roots/counts",
            )
        raise
    if python_frame != rust_frame or python_response != rust_response:
        _fail(FAIL_GATE_EVIDENCE, "Python/Rust full split responses differ")
    return python_response


def _normative_document_fields(
    *, commit_wire: tuple[int, bytes], relative_path: str, payload: bytes
) -> dict[str, object]:
    # Raw repository paths may contain spaces or non-ASCII bytes and therefore
    # are not themselves IdDigestV1 machine IDs.  Bind the exact UTF-8 path
    # through a valid, deterministic path-alias machine ID.
    path_alias = "repo-path-sha256:" + hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()
    return {
        "repository_relative_path_id_digest": id_digest_v1(path_alias),
        "raw_git_blob_bytes": payload,
        "repository_commit_id": commit_wire,
    }


def _source_profile_fields(
    *,
    identity_field: str,
    identity_machine_id: str,
    governing_root: bytes,
    selector_machine_id: str,
    section_bytes: bytes,
    commit_wire: tuple[int, bytes],
) -> dict[str, object]:
    return {
        identity_field: id_digest_v1(identity_machine_id),
        "governing_normative_document_root": governing_root,
        "section_selector_id_digest": id_digest_v1(selector_machine_id),
        "section_blob_sha256": hashlib.sha256(section_bytes).digest(),
        "section_byte_length": len(section_bytes),
        "repository_commit_id": commit_wire,
    }


def build_committed_public_basis_candidates_v1(basis_commit: str) -> dict[str, object]:
    """Construct the non-secret, already-unambiguous part of the formal basis.

    Every byte comes from ``git show <basis>:<path>`` or the frozen typed-row
    generator.  The result intentionally says ``complete=false`` because the
    freeze has not yet supplied an exact committed generator for all
    ``IdentifierRegistryEntryV1`` and ``OperatorSemanticsEntryV1`` preimages.
    Root-shaped placeholders are never synthesized.
    """

    _require_lower_hex(basis_commit, 40, "basis_commit")
    commit_wire = git_sha1_commit_id(bytes.fromhex(basis_commit))
    document_paths = {
        1: PROJECT_ROOT
        / "docs/Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md",
        2: PROJECT_ROOT
        / "docs/Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md",
        3: PROJECT_ROOT
        / "docs/Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md",
    }
    blobs: dict[int, bytes] = {
        role: _git_blob_bytes(basis_commit, path) for role, path in document_paths.items()
    }
    document_objects: dict[int, dict[str, object]] = {}
    document_roots: dict[int, bytes] = {}
    for role, path in document_paths.items():
        relative = _repository_relative(path)
        fields = _normative_document_fields(
            commit_wire=commit_wire, relative_path=relative, payload=blobs[role]
        )
        document_objects[role] = fields
        document_roots[role] = candidate_content_root("NormativeDocumentBlobV1", fields)

    bundle = {
        "bundle_id_digest": id_digest_v1(
            "hegel-m25-normative-document-bundle-v1.1.2"
        ),
        "document_entries": tuple(
            (role, document_roots[role]) for role in (1, 2, 3)
        ),
        "repository_commit_id": commit_wire,
    }
    normative_bundle_root = candidate_content_root("NormativeDocumentBundleV1", bundle)

    # The closure addendum explicitly permits ``section:entire-document``.
    # The specialized profile IDs keep the five uses distinct while binding
    # the exact committed normative bytes without line-ending normalization.
    profiles: dict[str, dict[str, object]] = {}
    profile_specs = (
        (
            "CanonicalAstProfileSpecV1",
            "profile_id_digest",
            "hegel-canonical-ast-v1",
            1,
        ),
        (
            "CanonicalCborProfileSpecV1",
            "profile_id_digest",
            "hegel-cbor-det-v1",
            1,
        ),
        (
            "Phase2BContractSpecV1",
            "contract_id_digest",
            "hegel-phase2b-contract-v1.1.2",
            1,
        ),
        (
            "MdlCodeTableSpecV1",
            "table_id_digest",
            "hegel-mdl-prefix-v1.0.0",
            1,
        ),
        (
            "HiddenArtifactScopeV1",
            "policy_id_digest",
            "hegel-hidden-artifact-scope-v1.1.2",
            1,
        ),
    )
    for schema_name, identity_field, identity_id, document_role in profile_specs:
        fields = _source_profile_fields(
            identity_field=identity_field,
            identity_machine_id=identity_id,
            governing_root=normative_bundle_root,
            selector_machine_id=(
                f"section:entire-document:{document_paths[document_role].name}"
            ),
            section_bytes=blobs[document_role],
            commit_wire=commit_wire,
        )
        profiles[schema_name] = fields
        candidate_content_root(schema_name, fields)

    odd = generate_odd_role_rows_v1()
    sink = generate_sink_role_rows_v1()
    typed_roots = {
        "outside_target_universe_root": odd.universe_root,
        "outside_target_truth_root": odd.truth_root,
        "null_control_universe_root": sink.universe_root,
        "null_control_truth_root": sink.truth_root,
    }
    if typed_roots != {
        "outside_target_universe_root": ODD_UNIVERSE_ROOT,
        "outside_target_truth_root": ODD_TRUTH_ROOT,
        "null_control_universe_root": SINK_UNIVERSE_ROOT,
        "null_control_truth_root": SINK_TRUTH_ROOT,
    }:
        _fail(FAIL_GATE_EVIDENCE, "typed row roots differ from the amendment")

    profile_roots = {
        name: candidate_content_root(name, fields) for name, fields in profiles.items()
    }
    return {
        "schema": PUBLIC_BASIS_SCHEMA,
        "basis_commit": basis_commit,
        "artifact_kind": "COMMITTED_CANDIDATE_PREIMAGES_NON_AUTHORITATIVE",
        "normative_document_objects": document_objects,
        "normative_document_roots": document_roots,
        "normative_document_bundle": bundle,
        "normative_document_bundle_root": normative_bundle_root,
        "source_profile_objects": profiles,
        "source_profile_roots": profile_roots,
        "typed_role_roots": typed_roots,
        "typed_role_row_counts": {"outside_target": 480, "null_control": 85},
        "complete": False,
        "formal_promotion_allowed": False,
        "missing_preimage_classes": [
            "IdentifierRegistryEntryV1[]",
            "OperatorSemanticsEntryV1[]",
            "complete child DSL/freeze/contract root DAG",
            "independent Rust full-split FD3 endpoint",
        ],
        "failure_code": FAIL_PUBLIC_BASIS_REGISTRY_PREIMAGE,
    }


def validate_ceremony_admission_v1(
    *,
    actor_qualification_report: Mapping[str, object],
    errata_qualification_report: Mapping[str, object],
    basis_commit: str,
    committed_input_paths: Sequence[Path],
) -> dict[str, object]:
    """Validate all pre-side-effect admission evidence.

    Callers must invoke this before marker/key/seed creation.  It checks both
    qualification reports and proves every ceremony runtime input is already
    byte-identical to the reported Commit A.
    """

    _require_lower_hex(basis_commit, 40, "basis_commit")
    actor = validate_qualification_report(actor_qualification_report)
    if actor["technical_actor_eligible"] is not True:
        _fail(FAIL_CEREMONY_ELIGIBILITY, "technical actor report is not eligible")
    if actor["basis_commit"] != basis_commit:
        _fail(FAIL_CEREMONY_BASIS_COMMIT, "actor report binds a different commit")
    try:
        validate_dual_errata_qualification_report(errata_qualification_report)
    except Exception as exc:
        _fail(FAIL_CEREMONY_ELIGIBILITY, f"dual exact-wire report is invalid: {exc}")
    if errata_qualification_report.get("implementation_basis_commit") != basis_commit:
        _fail(FAIL_CEREMONY_BASIS_COMMIT, "errata report binds a different commit")

    ancestry = subprocess.run(
        ["/usr/bin/git", "merge-base", "--is-ancestor", basis_commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_SSH_COMMAND": "false",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    if ancestry.returncode != 0:
        _fail(FAIL_CEREMONY_BASIS_COMMIT, "basis commit is not an ancestor of HEAD")
    if not committed_input_paths:
        _fail(FAIL_CEREMONY_INPUT_UNCOMMITTED, "ceremony input path list is empty")
    bindings: dict[str, str] = {}
    for path in committed_input_paths:
        payload = _assert_worktree_blob_equals_commit(basis_commit, path)
        bindings[_repository_relative(path)] = hashlib.sha256(payload).hexdigest()
    return {
        "basis_commit": basis_commit,
        "authority_class": AUTHORITY_CLASS,
        "technical_actor_eligible": True,
        "dual_exact_wire_qualified": True,
        "all_ceremony_inputs_committed": True,
        "input_sha256": dict(sorted(bindings.items())),
        "pre_side_effect_admission": True,
        "marker_created": False,
        "key_or_seed_generated": False,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _marker_payload(snapshot: MarkerSnapshot) -> bytes:
    validate_marker_snapshot(snapshot)
    return _canonical_json_bytes(
        {
            "schema": "hegel-phase3-split-seed-instantiation-marker/1",
            "state": snapshot.state,
            "split_version_digest_hex": snapshot.split_version_digest.hex(),
            "seed_commitment_manifest_root_hex_or_null": (
                None
                if snapshot.seed_commitment_manifest_root is None
                else snapshot.seed_commitment_manifest_root.hex()
            ),
            "custodian_key_id_hex": snapshot.custodian_key_id.hex(),
            "created_at_unix_seconds": snapshot.created_at_unix_seconds,
        }
    )


def read_marker_snapshot_v1(marker_path: Path) -> MarkerSnapshot:
    """Read one strict public-metadata marker without following a symlink."""

    if marker_path.is_symlink():
        _fail(FAIL_MARKER_ATOMICITY, "marker may not be a symlink")
    try:
        metadata = marker_path.stat()
        raw = marker_path.read_bytes()
    except OSError as exc:
        _fail(FAIL_MARKER_ATOMICITY, f"cannot read marker: {exc}")
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
        _fail(FAIL_MARKER_ATOMICITY, "marker must be a regular 0600 file")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(FAIL_MARKER_ATOMICITY, f"marker is not canonical JSON: {exc}")
    expected = {
        "schema",
        "state",
        "split_version_digest_hex",
        "seed_commitment_manifest_root_hex_or_null",
        "custodian_key_id_hex",
        "created_at_unix_seconds",
    }
    if type(value) is not dict or set(value) != expected:
        _fail(FAIL_MARKER_ATOMICITY, "marker field set differs")
    if _canonical_json_bytes(value) != raw:
        _fail(FAIL_MARKER_ATOMICITY, "marker JSON is not canonical")
    if value["schema"] != "hegel-phase3-split-seed-instantiation-marker/1":
        _fail(FAIL_MARKER_ATOMICITY, "marker schema differs")
    try:
        split_digest = bytes.fromhex(value["split_version_digest_hex"])
        key_id = bytes.fromhex(value["custodian_key_id_hex"])
        root_hex = value["seed_commitment_manifest_root_hex_or_null"]
        root = None if root_hex is None else bytes.fromhex(root_hex)
    except (TypeError, ValueError) as exc:
        _fail(FAIL_MARKER_ATOMICITY, f"marker hex field is invalid: {exc}")
    snapshot = MarkerSnapshot(
        state=value["state"],
        split_version_digest=split_digest,
        seed_commitment_manifest_root=root,
        custodian_key_id=key_id,
        created_at_unix_seconds=value["created_at_unix_seconds"],
    )
    return validate_marker_snapshot(snapshot)


def create_pending_marker_v1(
    *,
    secret_state_directory: Path,
    split_version_digest: bytes,
    custodian_key_id: bytes,
    created_at_unix_seconds: int,
) -> tuple[Path, MarkerSnapshot]:
    """Atomically persist PENDING before any caller performs seed CSPRNG."""

    state = validate_secret_state_directory(
        secret_state_directory, repository_root=REPOSITORY_ROOT
    )
    snapshot = validate_marker_snapshot(
        MarkerSnapshot(
            state="PENDING",
            split_version_digest=split_version_digest,
            seed_commitment_manifest_root=None,
            custodian_key_id=custodian_key_id,
            created_at_unix_seconds=created_at_unix_seconds,
        )
    )
    marker = state / "split_seed_instantiation.marker"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(marker, flags, 0o600)
    except FileExistsError:
        existing = read_marker_snapshot_v1(marker)
        if existing.state == "PENDING":
            _fail(FAIL_MARKER_RECOVERY_REQUIRED, "PENDING marker prohibits redraw")
        _fail(FAIL_MARKER_ALREADY_EXISTS, "COMPLETE marker prohibits redraw")
    except OSError as exc:
        _fail(FAIL_MARKER_ATOMICITY, f"cannot create marker: {exc}")
    try:
        payload = _marker_payload(snapshot)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail(FAIL_MARKER_ATOMICITY, "short marker write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(state)
    return marker, snapshot


def complete_marker_v1(
    *, marker_path: Path, seed_commitment_manifest_root: bytes
) -> MarkerSnapshot:
    """Atomically replace the exact PENDING marker with COMPLETE."""

    pending = read_marker_snapshot_v1(marker_path)
    if pending.state != "PENDING":
        _fail(FAIL_MARKER_ALREADY_EXISTS, "only PENDING may transition to COMPLETE")
    complete = validate_marker_snapshot(
        MarkerSnapshot(
            state="COMPLETE",
            split_version_digest=pending.split_version_digest,
            seed_commitment_manifest_root=_require_bytes(
                seed_commitment_manifest_root, 32, "seed commitment manifest root"
            ),
            custodian_key_id=pending.custodian_key_id,
            created_at_unix_seconds=pending.created_at_unix_seconds,
        )
    )
    temporary = marker_path.with_name(marker_path.name + ".complete.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
    except OSError as exc:
        _fail(FAIL_MARKER_ATOMICITY, f"cannot create COMPLETE temp marker: {exc}")
    try:
        payload = _marker_payload(complete)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail(FAIL_MARKER_ATOMICITY, "short COMPLETE marker write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, marker_path)
    _fsync_directory(marker_path.parent)
    return read_marker_snapshot_v1(marker_path)


def _verify_ed25519(public_key: bytes, signature: bytes, message: bytes) -> None:
    _require_bytes(public_key, 32, "Ed25519 public key")
    _require_bytes(signature, 64, "Ed25519 signature")
    if type(message) is not bytes:
        _fail(FAIL_SIGNATURE_INVALID, "signature message must be bytes")
    try:
        with tempfile.TemporaryDirectory(
            prefix="hegel-m25-ed25519-", dir="/tmp"
        ) as raw_private_directory:
            private_directory = Path(raw_private_directory)
            private_directory.chmod(0o700)
            verifier = make_openssl_ed25519_verifier_v1(private_directory)
            verifier(public_key, signature, message)
    except Exception as exc:
        _fail(FAIL_SIGNATURE_INVALID, f"Ed25519 verification failed: {exc}")


def validate_single_signature_envelope_v1(
    *,
    envelope_fields: Mapping[str, object],
    signer_purpose_id: int,
    signer_key_id: bytes,
    signer_public_key: bytes,
) -> bytes:
    """Replay one exact external envelope and its purpose-bound signature."""

    build_formal_object("SignedManifestEnvelopeV1", envelope_fields)
    signatures = envelope_fields["signatures"]
    if not isinstance(signatures, (tuple, list)) or len(signatures) != 1:
        _fail(FAIL_SIGNATURE_INVALID, "external envelope must contain one signature")
    record = signatures[0]
    if (
        not isinstance(record, (tuple, list))
        or len(record) != 2
        or record[0] != signer_key_id
    ):
        _fail(FAIL_SIGNATURE_INVALID, "external envelope key ID differs")
    preimage = external_signature_preimage_v1(
        envelope_fields["enclosed_object_tag"],  # type: ignore[arg-type]
        envelope_fields["enclosed_manifest_root"],  # type: ignore[arg-type]
        signer_purpose_id,
        envelope_fields["signer_key_epoch"],  # type: ignore[arg-type]
    )
    _verify_ed25519(signer_public_key, record[1], preimage)
    return candidate_content_root("SignedManifestEnvelopeV1", envelope_fields)


def _canonical_object_round_trip(name: str, fields: Mapping[str, object]) -> bytes:
    payload = encode_formal_object(name, fields)
    decoded = decode_formal_object(payload, expected_name=name)
    if encode_formal_object(name, decoded.fields) != payload:
        _fail(FAIL_GATE_EVIDENCE, f"{name} failed strict decode/re-encode")
    return candidate_content_root(name, fields)


@dataclass(frozen=True, slots=True)
class GateEvidenceInputsV1:
    """Complete public replay inputs for Gates 15--24.

    No field may contain a raw seed, private key, derived role key, or split
    assignment rows.  Those values are intentionally absent from the type.
    """

    basis_commit: str
    actor_qualification_report: Mapping[str, object]
    errata_qualification_report: Mapping[str, object]
    marker_snapshot: MarkerSnapshot
    actor_key_manifests: tuple[Mapping[str, object], ...]
    replacement_policy_fields: Mapping[str, object]
    trust_genesis_fields: Mapping[str, object]
    split_seed_commitment_fields: Mapping[str, object]
    ledger_genesis_fields: Mapping[str, object]
    parent_top_level_path_rows: tuple[Mapping[str, object], ...]
    parent_history_rows: tuple[Mapping[str, object], ...]
    parent_touched_rows: tuple[tuple[Mapping[str, object], ...], ...]
    parent_legacy_rows: tuple[Mapping[str, object], ...]
    parent_audit_bundle_fields: Mapping[str, object]
    parent_attestation_fields: Mapping[str, object]
    external_envelopes: tuple[tuple[int, Mapping[str, object]], ...]
    external_bundle_fields: Mapping[str, object]
    canonical_binding_objects: tuple[tuple[str, Mapping[str, object]], ...]
    python_static_roots: Mapping[str, bytes]
    rust_static_roots: Mapping[str, bytes]
    python_typed_roots: Mapping[str, bytes]
    rust_typed_roots: Mapping[str, bytes]
    python_split_frame: bytes
    rust_split_frame: bytes
    opaque_registration_intents: tuple[Mapping[str, object], ...]
    opaque_registry_records: tuple[Mapping[str, object], ...]
    opaque_registry_snapshots: tuple[Mapping[str, object], ...]
    execution_candidate_fields: Mapping[str, object]
    bridge_statement_fields: Mapping[str, object]
    bridge_envelopes: tuple[tuple[int, Mapping[str, object]], ...]
    bridge_bundle_fields: Mapping[str, object]
    execution_manifest_fields: Mapping[str, object]
    run_genesis_fields: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class QualifiedGateEvidenceV1:
    """Opaque-ish result created only by a complete successful replay."""

    basis_commit: str
    gate_report: Mapping[str, object]
    formal_roots: Mapping[str, bytes]
    _seal: object


_PROMOTION_SEAL = object()


def _actor_key_by_purpose(
    manifests: Sequence[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for manifest in manifests:
        purpose = manifest.get("purpose_id")
        if type(purpose) is not int or purpose in result:
            _fail(FAIL_GATE_EVIDENCE, "actor-key purpose set is invalid")
        result[purpose] = manifest
    if set(result) != {1, 2, 3, 4}:
        _fail(FAIL_GATE_EVIDENCE, "actor-key purposes must be exactly 1,2,3,4")
    return result


def _validate_report_basis(
    inputs: GateEvidenceInputsV1,
    *,
    prevalidated_actor_report: Mapping[str, object] | None = None,
    prevalidated_errata_report: Mapping[str, object] | None = None,
) -> None:
    if (prevalidated_actor_report is None) is not (
        prevalidated_errata_report is None
    ):
        _fail(FAIL_GATE_EVIDENCE, "prevalidated report basis is incomplete")
    if prevalidated_actor_report is None:
        actor = validate_qualification_report(inputs.actor_qualification_report)
        try:
            validate_dual_errata_qualification_report(
                inputs.errata_qualification_report
            )
        except Exception as exc:
            _fail(FAIL_GATE_EVIDENCE, f"errata qualification is invalid: {exc}")
        errata = inputs.errata_qualification_report
    else:
        assert prevalidated_errata_report is not None
        actor = dict(prevalidated_actor_report)
        errata = dict(prevalidated_errata_report)
        if (
            _canonical_json_bytes(actor)
            != _canonical_json_bytes(inputs.actor_qualification_report)
            or _canonical_json_bytes(errata)
            != _canonical_json_bytes(inputs.errata_qualification_report)
        ):
            _fail(
                FAIL_GATE_EVIDENCE,
                "prevalidated report basis differs from gate evidence inputs",
            )
    if actor["technical_actor_eligible"] is not True:
        _fail(FAIL_CEREMONY_ELIGIBILITY, "actor qualification is not eligible")
    if actor["basis_commit"] != inputs.basis_commit:
        _fail(FAIL_CEREMONY_BASIS_COMMIT, "actor report basis differs")
    if errata.get("implementation_basis_commit") != inputs.basis_commit:
        _fail(FAIL_CEREMONY_BASIS_COMMIT, "errata report basis differs")


def _evaluate_gates_15_24_impl_v1(
    inputs: GateEvidenceInputsV1,
    *,
    prevalidated_actor_report: Mapping[str, object] | None = None,
    prevalidated_errata_report: Mapping[str, object] | None = None,
) -> QualifiedGateEvidenceV1:
    """Replay and require every Gate 15--24 predicate.

    The function returns no partial success.  A single failure raises a stable
    machine error and no value capable of formal promotion is created.
    """

    if not isinstance(inputs, GateEvidenceInputsV1):
        raise TypeError("gate evidence must be GateEvidenceInputsV1")
    _require_lower_hex(inputs.basis_commit, 40, "basis_commit")
    assert_public_payload_contains_no_secret_fields(
        {field.name: getattr(inputs, field.name) for field in dataclass_fields(inputs)}
    )
    _validate_report_basis(
        inputs,
        prevalidated_actor_report=prevalidated_actor_report,
        prevalidated_errata_report=prevalidated_errata_report,
    )
    commit_wire = git_sha1_commit_id(bytes.fromhex(inputs.basis_commit))
    validate_marker_snapshot(inputs.marker_snapshot)
    keys = _actor_key_by_purpose(inputs.actor_key_manifests)
    trust_root = validate_actor_trust_bindings_v1(
        inputs.trust_genesis_fields,
        inputs.actor_key_manifests,
        inputs.replacement_policy_fields,
    )
    if inputs.trust_genesis_fields["repository_commit_id"] != commit_wire:
        _fail(FAIL_GATE_EVIDENCE, "actor trust does not bind Commit A")
    if any(
        manifest["repository_commit_id"] != commit_wire
        for manifest in inputs.actor_key_manifests
    ):
        _fail(FAIL_GATE_EVIDENCE, "an actor key manifest does not bind Commit A")

    external_by_tag: dict[int, tuple[int, Mapping[str, object]]] = {}
    for purpose, envelope in inputs.external_envelopes:
        tag = envelope["enclosed_object_tag"]
        if type(tag) is not int or tag in external_by_tag:
            _fail(FAIL_GATE_EVIDENCE, "external envelope tag set is invalid")
        key = keys[purpose]
        validate_single_signature_envelope_v1(
            envelope_fields=envelope,
            signer_purpose_id=purpose,
            signer_key_id=key["key_id"],  # type: ignore[arg-type]
            signer_public_key=key["public_key_32_bytes"],  # type: ignore[arg-type]
        )
        external_by_tag[tag] = (purpose, envelope)
    external_bundle_root = validate_external_input_attestation_bundle_v1(
        inputs.external_bundle_fields, inputs.external_envelopes
    )

    split_response = require_full_split_response_agreement_v2(
        inputs.python_split_frame, inputs.rust_split_frame
    )
    seed_root = _canonical_object_round_trip(
        "SplitSeedCommitmentManifestV1", inputs.split_seed_commitment_fields
    )
    if (
        inputs.split_seed_commitment_fields["repository_commit_id"] != commit_wire
        or
        inputs.split_seed_commitment_fields["split_seed_commitment_digest"]
        != split_response.seed_commitment
        or external_by_tag[OBJECT_TAGS["SplitSeedCommitmentManifestV1"]][1][
            "enclosed_manifest_root"
        ]
        != seed_root
        or inputs.marker_snapshot.state != "COMPLETE"
        or inputs.marker_snapshot.seed_commitment_manifest_root != seed_root
    ):
        _fail(FAIL_GATE_EVIDENCE, "Gate 15 seed/marker/envelope binding differs")

    ledger_root = validate_hidden_access_ledger_genesis(inputs.ledger_genesis_fields)
    if (
        inputs.ledger_genesis_fields["repository_commit_id"] != commit_wire
        or
        external_by_tag[OBJECT_TAGS["HiddenAccessLedgerRecordV1"]][1][
            "enclosed_manifest_root"
        ]
        != ledger_root
        or inputs.ledger_genesis_fields["subject_manifest_root"] != seed_root
    ):
        _fail(FAIL_GATE_EVIDENCE, "Gate 16 ledger signature binds another root")

    audit_root = validate_parent_absence_audit_bundle_v1(
        inputs.parent_top_level_path_rows,
        inputs.parent_history_rows,
        inputs.parent_touched_rows,
        inputs.parent_legacy_rows,
        inputs.parent_audit_bundle_fields,
    )
    parent_root = _canonical_object_round_trip(
        "ParentManifestAbsenceAttestationV2", inputs.parent_attestation_fields
    )
    if (
        inputs.parent_attestation_fields["parent_repository_commit_id"]
        != git_sha1_commit_id(AUDITED_PARENT_COMMIT_SHA1)
        or inputs.parent_attestation_fields["audit_bundle_root"] != audit_root
        or external_by_tag[OBJECT_TAGS["ParentManifestAbsenceAttestationV2"]][1][
            "enclosed_manifest_root"
        ]
        != parent_root
    ):
        _fail(FAIL_GATE_EVIDENCE, "Gate 17 parent audit binding differs")

    canonical_counts: dict[str, int] = {}
    seen_binding_names: list[str] = []
    for name, fields in inputs.canonical_binding_objects:
        _canonical_object_round_trip(name, fields)
        if "repository_commit_id" in fields and fields["repository_commit_id"] != commit_wire:
            _fail(FAIL_GATE_EVIDENCE, f"Gate 18 {name} does not bind Commit A")
        canonical_counts[name] = canonical_counts.get(name, 0) + 1
        seen_binding_names.append(name)
    if any(canonical_counts.get(name, 0) != 1 for name in REQUIRED_CANONICAL_BINDING_OBJECTS):
        _fail(FAIL_GATE_EVIDENCE, "Gate 18 required binding object set is incomplete")
    if canonical_counts.get("DslRoleBindingManifestV1", 0) != 2:
        _fail(FAIL_GATE_EVIDENCE, "Gate 18 requires exactly two role binding manifests")

    if set(inputs.python_static_roots) != set(inputs.rust_static_roots):
        _fail(FAIL_GATE_EVIDENCE, "Gate 19 dual static root name sets differ")
    for name in REQUIRED_STATIC_DUAL_ROOTS:
        if (
            name not in inputs.python_static_roots
            or _require_bytes(inputs.python_static_roots[name], 32, name)
            != _require_bytes(inputs.rust_static_roots[name], 32, name)
        ):
            _fail(FAIL_GATE_EVIDENCE, f"Gate 19 dual root differs: {name}")

    exact_typed = {
        "outside_target_universe_root": ODD_UNIVERSE_ROOT,
        "outside_target_truth_root": ODD_TRUTH_ROOT,
        "null_control_universe_root": SINK_UNIVERSE_ROOT,
        "null_control_truth_root": SINK_TRUTH_ROOT,
    }
    if dict(inputs.python_typed_roots) != exact_typed or dict(inputs.rust_typed_roots) != exact_typed:
        _fail(FAIL_GATE_EVIDENCE, "Gate 20/21 typed roots differ from frozen goldens")

    if (
        len(inputs.opaque_registration_intents) != 2
        or len(inputs.opaque_registry_records) != 2
        or len(inputs.opaque_registry_snapshots) != 2
    ):
        _fail(FAIL_GATE_EVIDENCE, "Gate 24 requires run and ledger ID registry appends")
    first_snapshot_root = validate_opaque_id_registry_append_v1(
        inputs.opaque_registration_intents[:1],
        inputs.opaque_registry_records[:1],
        inputs.opaque_registry_snapshots[0],
    )
    final_snapshot_root = validate_opaque_id_registry_append_v1(
        inputs.opaque_registration_intents,
        inputs.opaque_registry_records,
        inputs.opaque_registry_snapshots[1],
        previous_snapshot_fields=inputs.opaque_registry_snapshots[0],
    )
    del first_snapshot_root
    run_id = inputs.execution_candidate_fields["run_id"]
    ledger_id = inputs.ledger_genesis_fields["ledger_id"]
    if (
        inputs.opaque_registration_intents[0]["opaque_id_kind_id"] != 1
        or inputs.opaque_registration_intents[0]["opaque_id_16_bytes"] != run_id
        or inputs.opaque_registration_intents[1]["opaque_id_kind_id"] != 2
        or inputs.opaque_registration_intents[1]["opaque_id_16_bytes"] != ledger_id
    ):
        _fail(FAIL_GATE_EVIDENCE, "run/ledger IDs are not registered in frozen kind order")

    candidate_root, statement_root, manifest_root = validate_execution_identity_linkage_v1(
        inputs.execution_candidate_fields,
        inputs.bridge_statement_fields,
        inputs.bridge_bundle_fields,
        inputs.execution_manifest_fields,
        inputs.run_genesis_fields,
    )
    if inputs.execution_candidate_fields["repository_commit_id"] != commit_wire:
        _fail(FAIL_GATE_EVIDENCE, "Gate 24 execution candidate does not bind Commit A")
    bridge_bundle_root = validate_bridge_attestation_bundle_v1(
        statement_root, inputs.bridge_bundle_fields, inputs.bridge_envelopes
    )
    for purpose, envelope in inputs.bridge_envelopes:
        key = keys[purpose]
        validate_single_signature_envelope_v1(
            envelope_fields=envelope,
            signer_purpose_id=purpose,
            signer_key_id=key["key_id"],  # type: ignore[arg-type]
            signer_public_key=key["public_key_32_bytes"],  # type: ignore[arg-type]
        )
    if inputs.execution_manifest_fields["bridge_attestation_bundle_root"] != bridge_bundle_root:
        _fail(FAIL_GATE_EVIDENCE, "Gate 24 manifest bridge bundle differs")
    if inputs.execution_candidate_fields["custodian_attestation_bundle_root"] != external_bundle_root:
        _fail(FAIL_GATE_EVIDENCE, "execution candidate external bundle differs")
    if inputs.execution_candidate_fields["actor_trust_genesis_root"] != trust_root:
        _fail(FAIL_GATE_EVIDENCE, "execution candidate trust root differs")
    if (
        inputs.execution_candidate_fields["opaque_id_registry_snapshot_root"]
        != final_snapshot_root
    ):
        _fail(FAIL_GATE_EVIDENCE, "execution candidate opaque-ID snapshot differs")
    if (
        inputs.execution_candidate_fields["hidden_access_ledger_genesis_root"]
        != ledger_root
        or inputs.execution_candidate_fields["hidden_access_ledger_head_root"]
        != ledger_root
    ):
        _fail(FAIL_GATE_EVIDENCE, "Gate 16 ledger head is not genesis-only")
    for name, root in split_response.roots.items():
        if inputs.execution_candidate_fields[name] != root:
            _fail(FAIL_GATE_EVIDENCE, f"execution candidate split root differs: {name}")
    for name, root in exact_typed.items():
        if inputs.execution_candidate_fields[name] != root:
            _fail(FAIL_GATE_EVIDENCE, f"execution candidate typed root differs: {name}")
    output_slots = {
        name: inputs.run_genesis_fields[f"{name}_or_null"] for name in M3_RUN_OUTPUT_ROOTS
    }
    validate_m3_output_roots_null(output_slots)
    if inputs.run_genesis_fields["initial_state_id"] != 0:
        _fail(FAIL_GATE_EVIDENCE, "Gate 24 initial state is not NOT_RUN")

    gates = {number: True for number in GATE_NAMES}
    formal_roots = {
        **dict(inputs.python_static_roots),
        **exact_typed,
        **dict(split_response.roots),
        "actor_trust_genesis_root": trust_root,
        "external_input_attestation_bundle_root": external_bundle_root,
        "parent_absence_audit_bundle_root": audit_root,
        "parent_manifest_absence_attestation_root": parent_root,
        "split_seed_commitment_manifest_root": seed_root,
        "hidden_access_ledger_genesis_root": ledger_root,
        "m3_execution_candidate_root": candidate_root,
        "bridge_replay_statement_root": statement_root,
        "bridge_attestation_bundle_root": bridge_bundle_root,
        "m3_execution_manifest_root": manifest_root,
        "m3_run_genesis_root": candidate_content_root(
            "M3RunGenesisV1", inputs.run_genesis_fields
        ),
    }
    report: dict[str, object] = {
        "schema": GATE_REPORT_SCHEMA,
        "basis_commit": inputs.basis_commit,
        "authority_class": AUTHORITY_CLASS,
        "gates": [
            {"gate_number": number, "gate_name": name, "passed": gates[number]}
            for number, name in GATE_NAMES.items()
        ],
        "gates_before": 14,
        "gates_after": 24,
        "all_gates_15_24_passed": True,
        "m3_entry_qualified": True,
        "child_state": "NOT_RUN",
        "m3_run_started": False,
        "output_slot_count": 15,
        "all_output_slots_null": True,
        "formal_root_names": sorted(formal_roots),
    }
    report["gate_report_sha256"] = _canonical_json_sha256(report)
    return QualifiedGateEvidenceV1(
        basis_commit=inputs.basis_commit,
        gate_report=MappingProxyType(report),
        formal_roots=MappingProxyType(formal_roots),
        _seal=_PROMOTION_SEAL,
    )


def evaluate_gates_15_24_v1(
    inputs: GateEvidenceInputsV1,
) -> QualifiedGateEvidenceV1:
    """Replay Gate 15--24 with the live qualification validators."""

    return _evaluate_gates_15_24_impl_v1(inputs)


def _evaluate_gates_15_24_with_prevalidated_report_basis_v1(
    inputs: GateEvidenceInputsV1,
    *,
    actor_report: Mapping[str, object],
    errata_report: Mapping[str, object],
) -> QualifiedGateEvidenceV1:
    """Internal replay after a caller-specific pure report-basis validation.

    The injected copies must be byte-for-byte equal to those embedded in
    ``inputs``.  This private entry point exists for a supplied-repository
    post-commit verifier; normal ceremony callers retain the live validators.
    """

    return _evaluate_gates_15_24_impl_v1(
        inputs,
        prevalidated_actor_report=actor_report,
        prevalidated_errata_report=errata_report,
    )


def promote_gate_evidence_v1(
    evidence: QualifiedGateEvidenceV1,
) -> dict[str, object]:
    """Create a public-only 24/24 NOT_RUN artifact after complete replay."""

    if (
        not isinstance(evidence, QualifiedGateEvidenceV1)
        or evidence._seal is not _PROMOTION_SEAL
        or evidence.gate_report.get("all_gates_15_24_passed") is not True
        or evidence.gate_report.get("gates_after") != 24
        or evidence.gate_report.get("child_state") != "NOT_RUN"
        or evidence.gate_report.get("m3_run_started") is not False
    ):
        _fail(FAIL_FORMAL_PROMOTION_CONTEXT, "gate evidence is not promotable")
    payload: dict[str, object] = {
        "schema": CEREMONY_SCHEMA,
        "artifact_kind": PUBLIC_ARTIFACT_KIND,
        "basis_commit": evidence.basis_commit,
        "authority_class": AUTHORITY_CLASS,
        "authority_disclosure": dict(TECHNICAL_ACTOR_DISCLOSURE_V1),
        "formal_roots": {
            name: root.hex() for name, root in sorted(evidence.formal_roots.items())
        },
        "gate_report": dict(evidence.gate_report),
        "m3_entry_qualified": True,
        "child_state": "NOT_RUN",
        "m3_run_started": False,
        "phase3_m3_start_required_separately": True,
        "contains_private_key": False,
        "contains_raw_split_seed": False,
        "contains_split_assignment_rows": False,
    }
    assert_public_payload_contains_no_secret_fields(payload)
    payload["public_artifact_sha256"] = _canonical_json_sha256(payload)
    return payload


__all__ = [
    "AUTHORITY_CLASS",
    "CEREMONY_SCHEMA",
    "FAIL_CEREMONY_BASIS_COMMIT",
    "FAIL_CEREMONY_ELIGIBILITY",
    "FAIL_CEREMONY_INPUT_UNCOMMITTED",
    "FAIL_CONTAINER_INVOCATION_POLICY",
    "FAIL_FORMAL_PROMOTION_CONTEXT",
    "FAIL_GATE_EVIDENCE",
    "FAIL_MARKER_ALREADY_EXISTS",
    "FAIL_MARKER_RECOVERY_REQUIRED",
    "FAIL_PUBLIC_BASIS_INCOMPLETE",
    "FAIL_PUBLIC_BASIS_REGISTRY_PREIMAGE",
    "FAIL_SIGNATURE_INVALID",
    "FAIL_SECRET_TRANSPORT",
    "FAIL_SPLIT_FULL_ENDPOINT_REQUIRED",
    "FAIL_SPLIT_RESPONSE_FRAMING",
    "FAIL_SPLIT_RESPONSE_SCHEMA",
    "GATE_NAMES",
    "GateEvidenceInputsV1",
    "ContainerActorInvocationV1",
    "M25ContainerCeremonyError",
    "PUBLIC_BASIS_SCHEMA",
    "QualifiedGateEvidenceV1",
    "SPLIT_RESPONSE_ROWS",
    "SPLIT_RESPONSE_SCHEMA_ID",
    "TECHNICAL_ACTOR_DISCLOSURE_V1",
    "SplitCalculatorPublicResponseV2",
    "SplitRootCommitment",
    "build_committed_public_basis_candidates_v1",
    "build_actor_key_manifest_fields_v1",
    "build_offline_actor_invocation_v1",
    "build_single_signature_envelope_fields_v1",
    "complete_marker_v1",
    "create_pending_marker_v1",
    "decode_split_calculator_public_frame_v2",
    "encode_split_calculator_public_frame_v2",
    "encode_split_calculator_public_payload_v2",
    "evaluate_gates_15_24_v1",
    "promote_gate_evidence_v1",
    "parse_ed25519_spki_der_v1",
    "read_marker_snapshot_v1",
    "require_full_split_response_agreement_v2",
    "validate_ceremony_admission_v1",
    "validate_no_secret_transport_v1",
    "validate_single_signature_envelope_v1",
]
