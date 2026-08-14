"""Pure mechanics for three candidate Phase-2B baseline registrations.

The module closes and verifies caller-supplied candidate records for the three
frozen baselines.  It verifies fifteen Ed25519 signatures in memory and checks
content addressing, exact revision policies, artifact manifests, IF_ABSENT
``0 -> 1`` receipts, and a three-token declared-time chain.  It never reads or
writes a filesystem, downloads a model, builds an image, generates a key,
creates a signature, consults a clock, stores an object, allocates campaign or
attempt identifiers, or generates holdout cases.

Even a successful result is permanently mechanics-only.  Caller-supplied key
validity does not establish signer identity, organizational independence,
external authority, durable CAS, authoritative time, R05, R07, or R08.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import re
from typing import Callable, Final, NoReturn

from .rfc8785_v1 import canonicalize_rfc8785_v1

try:  # Optional verification backend; absence is a fail-closed result.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - tested through backend replacement.
    _InvalidSignature = Exception
    _Ed25519PublicKey = None


PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_SCHEMA: Final = (
    "hegel-machine-phase2b-c1-three-baseline-registration-profile/1"
)
PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_VERSION: Final = (
    "hegel-machine-phase2b-c1-three-baseline-registration-mechanics/1"
)
PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_CLAIM_LEVEL: Final = (
    "MECHANICS_ONLY_NOT_REGISTRATIONS"
)
PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_ED25519_BACKEND_AVAILABLE: Final = (
    _Ed25519PublicKey is not None
)

OWNER_AUTHORIZATION_ID_V1: Final = (
    "phase2b_c1_owner_actual_unsealed_960_authorization_v1_"
    "dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d"
)
READINESS_DAG_ID_V1: Final = (
    "phase2b_actual_unsealed_960_readiness_dag_v1_"
    "775b86cb499db3a991686634335ec444f7e44c7caea92b1c2b948036b585ea84"
)
PROTOCOL_ID_V1: Final = (
    "phase2b_protocol_"
    "62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a"
)
EXACT_FREEZE_ID_V1: Final = (
    "phase2b_exact_freeze_"
    "ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"
)
EXACT_FREEZE_VERSION_V1: Final = "hegel-freeze-p2b-p3-v1.0.1"

R05_NODE_ID_V1: Final = "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED"
R06_NODE_ID_V1: Final = "R06_GENERATOR_IMPLEMENTATION_QUALIFIED"
R07_NODE_ID_V1: Final = "R07_BASELINE_REGISTRATIONS_FROZEN"
R08_NODE_ID_V1: Final = "R08_STATISTICAL_VALIDATION_PASSED"

_PROFILE_DOMAIN: Final = b"HEGEL/PHASE2B/C1/THREE_BASELINE/PROFILE/V1\x00"
_PROFILE_PREFIX: Final = "phase2b_c1_three_baseline_registration_profile_v1_"
_FILE_MANIFEST_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/ARTIFACT_FILE_MANIFEST/V1\x00"
)
_R05_CANDIDATE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/R05_PREDECESSOR_CANDIDATE/V1\x00"
)
_KEY_DOMAIN: Final = b"HEGEL/PHASE2B/C1/THREE_BASELINE/KEY/V1\x00"
_IMPLEMENTATION_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/IMPLEMENTATION/V1\x00"
)
_BUILDER_SUBJECT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/BUILDER_SUBJECT/V1\x00"
)
_BUILDER_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/BUILDER_SIGNATURE/V1\x00"
)
_BUILDER_ATTESTATION_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/BUILDER_ATTESTATION/V1\x00"
)
_QUALIFIER_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/QUALIFIER_SIGNATURE/V1\x00"
)
_QUALIFIER_ATTESTATION_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/QUALIFIER_ATTESTATION/V1\x00"
)
_RECORD_DOMAIN: Final = b"HEGEL/PHASE2B/C1/THREE_BASELINE/RICH_RECORD/V1\x00"
_CUSTODIAN_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/CUSTODIAN_SIGNATURE/V1\x00"
)
_CUSTODIAN_ATTESTATION_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/CUSTODIAN_ATTESTATION/V1\x00"
)
_STORE_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/STORE_SIGNATURE/V1\x00"
)
_STORE_RECEIPT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/STORE_RECEIPT/V1\x00"
)
_TIME_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/TIME_SIGNATURE/V1\x00"
)
_TIME_TOKEN_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/TIME_TOKEN/V1\x00"
)
_FINAL_REGISTRATION_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/FINAL_REGISTRATION/V1\x00"
)
_REGISTRY_ROOT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/REGISTRY_ROOT/V1\x00"
)
_REGISTRY_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/REGISTRY_ID/V1\x00"
)
_RESULT_DOMAIN: Final = b"HEGEL/PHASE2B/C1/THREE_BASELINE/RESULT/V1\x00"
_LEGACY_PROJECTION_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/THREE_BASELINE/LEGACY_PROJECTION_CANDIDATE/V1\x00"
)

_FILE_MANIFEST_PREFIX: Final = (
    "phase2b_c1_baseline_artifact_file_manifest_v1_"
)
_R05_CANDIDATE_PREFIX: Final = (
    "phase2b_c1_r05_predecessor_candidate_v1_"
)
_KEY_PREFIX: Final = "phase2b_c1_baseline_candidate_key_v1_"
_IMPLEMENTATION_PREFIX: Final = "phase2b_c1_baseline_implementation_v1_"
_BUILDER_SUBJECT_PREFIX: Final = "phase2b_c1_baseline_builder_subject_v1_"
_BUILDER_ATTESTATION_PREFIX: Final = (
    "phase2b_c1_baseline_builder_attestation_candidate_v1_"
)
_QUALIFIER_ATTESTATION_PREFIX: Final = (
    "phase2b_c1_baseline_qualifier_attestation_candidate_v1_"
)
_RECORD_PREFIX: Final = "phase2b_c1_rich_baseline_registration_record_v1_"
_CUSTODIAN_ATTESTATION_PREFIX: Final = (
    "phase2b_c1_baseline_custodian_attestation_candidate_v1_"
)
_STORE_RECEIPT_PREFIX: Final = (
    "phase2b_c1_baseline_store_if_absent_receipt_candidate_v1_"
)
_TIME_TOKEN_PREFIX: Final = (
    "phase2b_c1_baseline_declared_time_token_candidate_v1_"
)
_FINAL_REGISTRATION_PREFIX: Final = (
    "phase2b_c1_final_baseline_registration_candidate_v1_"
)
_REGISTRY_PREFIX: Final = (
    "phase2b_c1_three_baseline_registration_candidate_registry_v1_"
)
_RESULT_PREFIX: Final = (
    "phase2b_c1_three_baseline_registration_mechanics_result_v1_"
)
_LEGACY_PROJECTION_PREFIX: Final = (
    "phase2b_c1_legacy_baseline_projection_candidate_v1_"
)

_QUALIFICATION_RECEIPT_PREFIX: Final = (
    "phase2b_formal_wire_covert_qualification_receipt_v2_"
)
_MAX_TEXT_BYTES: Final = 4_096
_MAX_PATH_BYTES: Final = 512
_MAX_FILES: Final = 4_096
_MAX_SAFE_INTEGER: Final = (1 << 53) - 1
_KEY_EPOCH: Final = 1
_RAW_PUBLIC_KEY_BYTES: Final = 32
_RAW_SIGNATURE_BYTES: Final = 64
_EXPECTED_SIGNATURES: Final = 15
_MAX_SNAPSHOT_DEPTH: Final = 32
_MAX_SNAPSHOT_NODES: Final = 32_768
_MAX_SNAPSHOT_TUPLE_ITEMS: Final = 4_096
_MAX_SNAPSHOT_TOTAL_TUPLE_ENTRIES: Final = 8_192
_MAX_SNAPSHOT_BYTES: Final = 4_096
_MAX_SNAPSHOT_TOTAL_UTF8_BYTES: Final = 1_048_576
_MAX_SNAPSHOT_TOTAL_RAW_BYTES: Final = 1_048_576
_MAX_SNAPSHOT_TOTAL_LEAF_BYTES: Final = 2_097_152
_MAX_SNAPSHOT_CANONICAL_BYTES: Final = 4_194_304
_SNAPSHOT_DOMAIN: Final = b"HEGEL/PHASE2B/C1/THREE_BASELINE/INPUT_SNAPSHOT/V1\x00"


class BaselineRegistrationKindV1(str, Enum):
    EMBEDDING_NEAREST_PROTOTYPE = "embedding_nearest_prototype"
    FROZEN_LLM_SEMANTIC_ONLY = "frozen_llm_semantic_only"
    FLAT_LEARNED_TYPED = "flat_learned_typed"


class BaselineRevisionPolicyV1(str, Enum):
    EXACT_40_HEX_COMMIT_REQUIRED = "exact_40_hex_commit_required"
    REPOSITORY_DEPENDENCY_LOCK_REQUIRED = "repository_dependency_lock_required"


class BaselineArtifactRoleV1(str, Enum):
    SOURCE_ARCHIVE = "SOURCE_ARCHIVE"
    MODEL_WEIGHTS = "MODEL_WEIGHTS"
    MODEL_CONFIGURATION = "MODEL_CONFIGURATION"
    TOKENIZER = "TOKENIZER"
    POOLING_CONFIGURATION = "POOLING_CONFIGURATION"
    GENERATION_CONFIGURATION = "GENERATION_CONFIGURATION"
    DECODER_CONFIGURATION = "DECODER_CONFIGURATION"
    DEPENDENCY_LOCK = "DEPENDENCY_LOCK"
    FEATURE_SCHEMA = "FEATURE_SCHEMA"
    SERIALIZED_ESTIMATOR = "SERIALIZED_ESTIMATOR"
    SBOM = "SBOM"
    DEVELOPMENT_STATE = "DEVELOPMENT_STATE"
    OFFLINE_QUALIFICATION_REPORT = "OFFLINE_QUALIFICATION_REPORT"
    LICENSE = "LICENSE"


class BaselineSignaturePurposeV1(str, Enum):
    BASELINE_RUNTIME_BUILDER_ATTESTATION = (
        "BASELINE_RUNTIME_BUILDER_ATTESTATION"
    )
    BASELINE_OFFLINE_QUALIFIER_ATTESTATION = (
        "BASELINE_OFFLINE_QUALIFIER_ATTESTATION"
    )
    BASELINE_REGISTRATION_CUSTODIAN_ATTESTATION = (
        "BASELINE_REGISTRATION_CUSTODIAN_ATTESTATION"
    )
    STORE_IF_ABSENT_RECEIPT = "STORE_IF_ABSENT_RECEIPT"
    DECLARED_TIME_TOKEN = "DECLARED_TIME_TOKEN"


class ThreeBaselineRegistrationDispositionV1(str, Enum):
    MECHANICS_ONLY_NOT_REGISTRATIONS = "MECHANICS_ONLY_NOT_REGISTRATIONS"
    REJECTED = "REJECTED"


class ThreeBaselineRegistrationReasonV1(str, Enum):
    CALLER_SUPPLIED_CANDIDATE_SIGNATURE_MECHANICS_VERIFIED = (
        "CALLER_SUPPLIED_CANDIDATE_SIGNATURE_MECHANICS_VERIFIED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    STRUCTURAL_CLOSURE_INVALID = "STRUCTURAL_CLOSURE_INVALID"
    SIGNATURE_BACKEND_UNAVAILABLE = "SIGNATURE_BACKEND_UNAVAILABLE"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    INPUT_MUTATED_DURING_VERIFICATION = "INPUT_MUTATED_DURING_VERIFICATION"
    RESOURCE_BUDGET_EXCEEDED = "RESOURCE_BUDGET_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_BASELINE_ORDER: Final = tuple(BaselineRegistrationKindV1)
_PURPOSE_ORDER: Final = tuple(BaselineSignaturePurposeV1)

_BASELINE_SPEC_IDS: Final = {
    BaselineRegistrationKindV1.EMBEDDING_NEAREST_PROTOTYPE: (
        "phase2b_baseline_spec_"
        "d14fa83a6843b92d1a502bba97b982323d1de8bf0abe22763d8c1cb67c97cfa4"
    ),
    BaselineRegistrationKindV1.FROZEN_LLM_SEMANTIC_ONLY: (
        "phase2b_baseline_spec_"
        "c07005a2e32423f5e11a2160367b089d8f69208289239bfc18821fb351f13fa8"
    ),
    BaselineRegistrationKindV1.FLAT_LEARNED_TYPED: (
        "phase2b_baseline_spec_"
        "0e7b098030bf7e5a36f2deb0befd40fbabc55ce538a78765be2aa33d6b7af6d2"
    ),
}
_IMPLEMENTATIONS: Final = {
    BaselineRegistrationKindV1.EMBEDDING_NEAREST_PROTOTYPE: (
        "sentence-transformers/all-mpnet-base-v2"
    ),
    BaselineRegistrationKindV1.FROZEN_LLM_SEMANTIC_ONLY: (
        "Qwen/Qwen2.5-7B-Instruct"
    ),
    BaselineRegistrationKindV1.FLAT_LEARNED_TYPED: (
        "sklearn.ensemble.HistGradientBoostingClassifier"
    ),
}
_REVISION_POLICIES: Final = {
    BaselineRegistrationKindV1.EMBEDDING_NEAREST_PROTOTYPE: (
        BaselineRevisionPolicyV1.EXACT_40_HEX_COMMIT_REQUIRED
    ),
    BaselineRegistrationKindV1.FROZEN_LLM_SEMANTIC_ONLY: (
        BaselineRevisionPolicyV1.EXACT_40_HEX_COMMIT_REQUIRED
    ),
    BaselineRegistrationKindV1.FLAT_LEARNED_TYPED: (
        BaselineRevisionPolicyV1.REPOSITORY_DEPENDENCY_LOCK_REQUIRED
    ),
}
_REQUIRED_ARTIFACT_ROLES: Final = {
    BaselineRegistrationKindV1.EMBEDDING_NEAREST_PROTOTYPE: frozenset(
        {
            BaselineArtifactRoleV1.SOURCE_ARCHIVE,
            BaselineArtifactRoleV1.MODEL_WEIGHTS,
            BaselineArtifactRoleV1.MODEL_CONFIGURATION,
            BaselineArtifactRoleV1.TOKENIZER,
            BaselineArtifactRoleV1.POOLING_CONFIGURATION,
            BaselineArtifactRoleV1.DECODER_CONFIGURATION,
            BaselineArtifactRoleV1.SBOM,
            BaselineArtifactRoleV1.DEVELOPMENT_STATE,
            BaselineArtifactRoleV1.OFFLINE_QUALIFICATION_REPORT,
            BaselineArtifactRoleV1.LICENSE,
        }
    ),
    BaselineRegistrationKindV1.FROZEN_LLM_SEMANTIC_ONLY: frozenset(
        {
            BaselineArtifactRoleV1.SOURCE_ARCHIVE,
            BaselineArtifactRoleV1.MODEL_WEIGHTS,
            BaselineArtifactRoleV1.MODEL_CONFIGURATION,
            BaselineArtifactRoleV1.TOKENIZER,
            BaselineArtifactRoleV1.GENERATION_CONFIGURATION,
            BaselineArtifactRoleV1.DECODER_CONFIGURATION,
            BaselineArtifactRoleV1.SBOM,
            BaselineArtifactRoleV1.DEVELOPMENT_STATE,
            BaselineArtifactRoleV1.OFFLINE_QUALIFICATION_REPORT,
            BaselineArtifactRoleV1.LICENSE,
        }
    ),
    BaselineRegistrationKindV1.FLAT_LEARNED_TYPED: frozenset(
        {
            BaselineArtifactRoleV1.SOURCE_ARCHIVE,
            BaselineArtifactRoleV1.DEPENDENCY_LOCK,
            BaselineArtifactRoleV1.FEATURE_SCHEMA,
            BaselineArtifactRoleV1.SERIALIZED_ESTIMATOR,
            BaselineArtifactRoleV1.DECODER_CONFIGURATION,
            BaselineArtifactRoleV1.SBOM,
            BaselineArtifactRoleV1.DEVELOPMENT_STATE,
            BaselineArtifactRoleV1.OFFLINE_QUALIFICATION_REPORT,
            BaselineArtifactRoleV1.LICENSE,
        }
    ),
}


def _fail(detail: str) -> NoReturn:
    raise ValueError(detail)


def _text(value: object, name: str, *, maximum: int = _MAX_TEXT_BYTES) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    if not 1 <= len(encoded) <= maximum:
        raise ValueError(f"{name} UTF-8 size drift")
    return value


def _sha256(value: object, name: str) -> str:
    text = _text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _sha1_commit(value: object, name: str) -> str:
    text = _text(value, name)
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise ValueError(f"{name} must be exact lowercase 40-hex")
    return text


def _prefixed_id(value: object, prefix: str, name: str) -> str:
    text = _text(value, name)
    if not text.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    _sha256(text[len(prefix) :], f"{name} digest")
    return text


def _image_digest(value: object, name: str) -> str:
    text = _text(value, name)
    if not text.startswith("sha256:"):
        raise ValueError(f"{name} algorithm drift")
    _sha256(text[7:], f"{name} digest")
    return text


def _raw(value: object, length: int, name: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        raise TypeError(f"{name} must be exact {length}-byte bytes")
    return bytes(value)


def _integer(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_SAFE_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} integer drift")
    return value


def _content_id(prefix: str, domain: bytes, body: dict[str, object]) -> str:
    return prefix + hashlib.sha256(domain + canonicalize_rfc8785_v1(body)).hexdigest()


def _root(domain: bytes, body: object) -> str:
    return hashlib.sha256(domain + canonicalize_rfc8785_v1(body)).hexdigest()


def _signature_preimage(domain: bytes, unsigned: dict[str, object]) -> bytes:
    return domain + canonicalize_rfc8785_v1(unsigned)


def _signed_candidate_id(
    prefix: str,
    domain: bytes,
    unsigned: dict[str, object],
    signature: bytes,
) -> str:
    return _content_id(
        prefix,
        domain,
        {**unsigned, "signature_hex": signature.hex()},
    )


def _profile_body_v1() -> dict[str, object]:
    baseline_rows = []
    for kind in _BASELINE_ORDER:
        baseline_rows.append(
            {
                "baseline_id": kind.value,
                "baseline_spec_id": _BASELINE_SPEC_IDS[kind],
                "implementation": _IMPLEMENTATIONS[kind],
                "revision_policy": _REVISION_POLICIES[kind].value,
                "required_artifact_roles": sorted(
                    role.value for role in _REQUIRED_ARTIFACT_ROLES[kind]
                ),
            }
        )
    return {
        "schema_version": PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_SCHEMA,
        "document_id": "hegel-phase2b-c1-three-baseline-registration-profile-v1",
        "status": PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_CLAIM_LEVEL,
        "claim_level": PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_CLAIM_LEVEL,
        "canonicalization": "RFC8785_CANONICAL_BODY_PLUS_ONE_LF",
        "exact_contract_bindings": {
            "owner_authorization_id": OWNER_AUTHORIZATION_ID_V1,
            "readiness_dag_id": READINESS_DAG_ID_V1,
            "protocol_id": PROTOCOL_ID_V1,
            "exact_freeze_id": EXACT_FREEZE_ID_V1,
            "exact_freeze_version": EXACT_FREEZE_VERSION_V1,
        },
        "readiness_contract": {
            "required_predecessor": R05_NODE_ID_V1,
            "target": R07_NODE_ID_V1,
            "parallel_node": R06_NODE_ID_V1,
            "successor": R08_NODE_ID_V1,
            "successor_dependencies_in_order": [R06_NODE_ID_V1, R07_NODE_ID_V1],
        },
        "baselines_in_order": baseline_rows,
        "revision_binding": {
            "hugging_face_revisions": "EXACT_LOWERCASE_40_HEX_COMMIT",
            "flat_learned_typed": (
                "EXACT_REPOSITORY_SOURCE_COMMIT_PLUS_DEPENDENCY_LOCK_SHA256"
            ),
        },
        "caller_supplied_candidate_bindings_in_order": [
            "source_root_candidate_sha256",
            "source_archive_root_candidate_sha256",
            "configuration_root_candidate_sha256",
            "decoder_root_candidate_sha256",
            "sbom_root_candidate_sha256",
            "runtime_image_candidate_digest",
            "development_state_root_candidate_sha256",
            "offline_qualification_root_candidate_sha256",
        ],
        "artifact_file_manifest": {
            "closed_role_enum": [role.value for role in BaselineArtifactRoleV1],
            "canonical_order": "RELATIVE_PATH_ASCENDING",
            "complete_required": True,
            "per_baseline_role_set_exact": True,
            "hgb_dependency_lock_entry_exactly_one_and_digest_bound": True,
            "root_formula": (
                "SHA256(HEGEL/PHASE2B/C1/THREE_BASELINE/"
                "ARTIFACT_FILE_MANIFEST/V1\\x00 || RFC8785(entries))"
            ),
            "maximum_files": _MAX_FILES,
        },
        "signature_candidate_contract": {
            "algorithm": "Ed25519",
            "key_epoch": _KEY_EPOCH,
            "raw_public_key_bytes": _RAW_PUBLIC_KEY_BYTES,
            "raw_signature_bytes": _RAW_SIGNATURE_BYTES,
            "purposes_in_order": [purpose.value for purpose in _PURPOSE_ORDER],
            "purpose_separated_key_ids_required": True,
            "purpose_separated_public_keys_required": True,
            "fresh_signature_verification_per_registration": True,
            "signatures_per_registration": len(_PURPOSE_ORDER),
        },
        "causal_contract": {
            "store_condition": "IF_ABSENT",
            "previous_generation": 0,
            "new_generation": 1,
            "declared_time_sequences_in_baseline_order": [0, 1, 2],
            "declared_time_previous_token_chain_required": True,
            "declared_time_fields": [
                "declared_unix_seconds",
                "declared_nanosecond",
            ],
            "declared_time_strictly_increasing": True,
            "maximum_safe_integer": _MAX_SAFE_INTEGER,
        },
        "safety_boundary": {
            "campaign_id": None,
            "attempt_id": None,
            "actual_holdout_case_count": 0,
            "actual_preservation_pair_count": 0,
            "actual_invalid_control_count": 0,
            "actual_baseline_registration_count": 0,
            "artifact_roots_replayed": False,
            "aggregate_snapshot_persisted": False,
            "r05_authority_verified": False,
            "r07_satisfied": False,
            "r08_unlocked": False,
            "external_authority_verified": False,
            "organizational_independence_verified": False,
            "durable_cas_verified": False,
            "authoritative_time_verified": False,
            "actual_execution_claimed": False,
        },
    }


PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID: Final = _content_id(
    _PROFILE_PREFIX,
    _PROFILE_DOMAIN,
    _profile_body_v1(),
)


def phase2b_c1_three_baseline_registration_profile_v1() -> dict[str, object]:
    """Return a detached exact profile wrapper without reading configuration."""

    body = _profile_body_v1()
    return {
        "profile_body": body,
        "profile_id": PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID,
    }


@dataclass(frozen=True, slots=True)
class BaselineArtifactFileV1:
    role: BaselineArtifactRoleV1
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.role) is not BaselineArtifactRoleV1:
            raise TypeError("artifact role exact type drift")
        path = _text(self.relative_path, "artifact relative path", maximum=_MAX_PATH_BYTES)
        if (
            path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or re.fullmatch(r"[A-Za-z0-9._+@/-]+", path) is None
        ):
            raise ValueError("artifact relative path is not closed portable syntax")
        _integer(self.size_bytes, "artifact size")
        _sha256(self.sha256, "artifact SHA-256")


def _artifact_file_projection(item: BaselineArtifactFileV1) -> dict[str, object]:
    item.__post_init__()
    return {
        "role": item.role.value,
        "relative_path": item.relative_path,
        "size_bytes": item.size_bytes,
        "sha256": item.sha256,
    }


def artifact_file_manifest_root_v1(
    files: tuple[BaselineArtifactFileV1, ...],
) -> str:
    if type(files) is not tuple or not 1 <= len(files) <= _MAX_FILES:
        raise TypeError("artifact files must be a bounded exact tuple")
    if any(type(item) is not BaselineArtifactFileV1 for item in files):
        raise TypeError("artifact file exact type drift")
    return _root(
        _FILE_MANIFEST_DOMAIN,
        [_artifact_file_projection(item) for item in files],
    )


@dataclass(frozen=True, slots=True)
class BaselineArtifactFileManifestV1:
    files: tuple[BaselineArtifactFileV1, ...]
    file_count: int
    total_bytes: int
    complete: bool
    manifest_root_sha256: str
    manifest_id: str

    def __post_init__(self) -> None:
        if type(self.files) is not tuple or not 1 <= len(self.files) <= _MAX_FILES:
            raise TypeError("artifact files must be a bounded exact tuple")
        if any(type(item) is not BaselineArtifactFileV1 for item in self.files):
            raise TypeError("artifact file exact type drift")
        for item in self.files:
            item.__post_init__()
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("artifact paths must be unique ascending values")
        roles = tuple(item.role for item in self.files)
        if len(roles) != len(set(roles)):
            raise ValueError("artifact roles must be unique")
        if type(self.file_count) is not int or self.file_count != len(self.files):
            raise ValueError("artifact file count drift")
        total = sum(item.size_bytes for item in self.files)
        if (
            type(self.total_bytes) is not int
            or self.total_bytes != total
            or total > _MAX_SAFE_INTEGER
        ):
            raise ValueError("artifact total bytes drift")
        if self.complete is not True:
            raise ValueError("artifact file manifest must be complete")
        expected_root = artifact_file_manifest_root_v1(self.files)
        if _sha256(self.manifest_root_sha256, "artifact manifest root") != expected_root:
            raise ValueError("artifact manifest root drift")
        expected_id = _FILE_MANIFEST_PREFIX + expected_root
        if _prefixed_id(self.manifest_id, _FILE_MANIFEST_PREFIX, "artifact manifest ID") != expected_id:
            raise ValueError("artifact manifest ID drift")


@dataclass(frozen=True, slots=True)
class BaselineCallerSuppliedCandidateBindingsV1:
    """Opaque caller-supplied digest candidates; only shape is checked here."""

    source_root_candidate_sha256: str
    source_archive_root_candidate_sha256: str
    configuration_root_candidate_sha256: str
    decoder_root_candidate_sha256: str
    sbom_root_candidate_sha256: str
    runtime_image_candidate_digest: str
    development_state_root_candidate_sha256: str
    offline_qualification_root_candidate_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "source_root_candidate_sha256",
            "source_archive_root_candidate_sha256",
            "configuration_root_candidate_sha256",
            "decoder_root_candidate_sha256",
            "sbom_root_candidate_sha256",
            "development_state_root_candidate_sha256",
            "offline_qualification_root_candidate_sha256",
        ):
            _sha256(getattr(self, field_name), field_name)
        _image_digest(
            self.runtime_image_candidate_digest,
            "runtime image candidate digest",
        )


def _candidate_bindings_projection(
    bindings: BaselineCallerSuppliedCandidateBindingsV1,
) -> dict[str, object]:
    bindings.__post_init__()
    return {
        "source_root_candidate_sha256": bindings.source_root_candidate_sha256,
        "source_archive_root_candidate_sha256": (
            bindings.source_archive_root_candidate_sha256
        ),
        "configuration_root_candidate_sha256": (
            bindings.configuration_root_candidate_sha256
        ),
        "decoder_root_candidate_sha256": bindings.decoder_root_candidate_sha256,
        "sbom_root_candidate_sha256": bindings.sbom_root_candidate_sha256,
        "runtime_image_candidate_digest": bindings.runtime_image_candidate_digest,
        "development_state_root_candidate_sha256": (
            bindings.development_state_root_candidate_sha256
        ),
        "offline_qualification_root_candidate_sha256": (
            bindings.offline_qualification_root_candidate_sha256
        ),
    }


def _r05_projection(value: "R05QualificationPredecessorCandidateV1") -> dict[str, object]:
    return {
        "readiness_node_id": value.readiness_node_id,
        "qualification_receipt_id": value.qualification_receipt_id,
        "source_commit_sha": value.source_commit_sha,
        "source_archive_root_sha256": value.source_archive_root_sha256,
        "dependency_lock_sha256": value.dependency_lock_sha256,
        "runtime_image_digest": value.runtime_image_digest,
        "synthetic_fixture_root_sha256": value.synthetic_fixture_root_sha256,
        "candidate_scope": value.candidate_scope,
        "actual_corpus_input_used": value.actual_corpus_input_used,
    }


@dataclass(frozen=True, slots=True)
class R05QualificationPredecessorCandidateV1:
    readiness_node_id: str
    qualification_receipt_id: str
    source_commit_sha: str
    source_archive_root_sha256: str
    dependency_lock_sha256: str
    runtime_image_digest: str
    synthetic_fixture_root_sha256: str
    candidate_scope: str
    actual_corpus_input_used: bool
    candidate_id: str

    def __post_init__(self) -> None:
        if self.readiness_node_id != R05_NODE_ID_V1:
            raise ValueError("R05 predecessor node drift")
        _prefixed_id(
            self.qualification_receipt_id,
            _QUALIFICATION_RECEIPT_PREFIX,
            "qualification receipt ID",
        )
        _sha1_commit(self.source_commit_sha, "R05 source commit")
        _sha256(self.source_archive_root_sha256, "R05 source archive root")
        _sha256(self.dependency_lock_sha256, "R05 dependency lock")
        _image_digest(self.runtime_image_digest, "R05 runtime image digest")
        _sha256(self.synthetic_fixture_root_sha256, "R05 synthetic fixture root")
        if self.candidate_scope != "SUPPLIED_R05_PREDECESSOR_CANDIDATE_NOT_AUTHORITY":
            raise ValueError("R05 candidate scope drift")
        if self.actual_corpus_input_used is not False:
            raise ValueError("R05 candidate must remain synthetic-dev-only")
        expected = _content_id(
            _R05_CANDIDATE_PREFIX,
            _R05_CANDIDATE_DOMAIN,
            _r05_projection(self),
        )
        if _prefixed_id(self.candidate_id, _R05_CANDIDATE_PREFIX, "R05 candidate ID") != expected:
            raise ValueError("R05 predecessor candidate ID drift")


def _key_projection(value: "BaselineCandidatePublicKeyV1") -> dict[str, object]:
    return {
        "purpose": value.purpose.value,
        "key_epoch": value.key_epoch,
        "algorithm": value.algorithm,
        "public_key_hex": value.public_key.hex(),
    }


@dataclass(frozen=True, slots=True)
class BaselineCandidatePublicKeyV1:
    purpose: BaselineSignaturePurposeV1
    key_epoch: int
    algorithm: str
    public_key: bytes
    key_id: str

    def __post_init__(self) -> None:
        if type(self.purpose) is not BaselineSignaturePurposeV1:
            raise TypeError("signature purpose exact type drift")
        if type(self.key_epoch) is not int or self.key_epoch != _KEY_EPOCH:
            raise ValueError("candidate key epoch drift")
        if self.algorithm != "Ed25519":
            raise ValueError("candidate key algorithm drift")
        _raw(self.public_key, _RAW_PUBLIC_KEY_BYTES, "Ed25519 public key")
        expected = _content_id(_KEY_PREFIX, _KEY_DOMAIN, _key_projection(self))
        if _prefixed_id(self.key_id, _KEY_PREFIX, "candidate key ID") != expected:
            raise ValueError("candidate key content address drift")


def _revision_projection(
    *,
    upstream_revision_40hex: str | None,
    repository_source_commit_40hex: str | None,
    repository_dependency_lock_sha256: str | None,
) -> dict[str, object]:
    return {
        "upstream_revision_40hex": upstream_revision_40hex,
        "repository_source_commit_40hex": repository_source_commit_40hex,
        "repository_dependency_lock_sha256": repository_dependency_lock_sha256,
    }


def _validate_revision_binding(
    kind: BaselineRegistrationKindV1,
    revision_policy: BaselineRevisionPolicyV1,
    upstream_revision_40hex: str | None,
    repository_source_commit_40hex: str | None,
    repository_dependency_lock_sha256: str | None,
) -> None:
    if revision_policy is not _REVISION_POLICIES[kind]:
        raise ValueError("baseline revision policy drift")
    if revision_policy is BaselineRevisionPolicyV1.EXACT_40_HEX_COMMIT_REQUIRED:
        _sha1_commit(upstream_revision_40hex, "Hugging Face revision")
        if repository_source_commit_40hex is not None or repository_dependency_lock_sha256 is not None:
            raise ValueError("Hugging Face revision closure drift")
    else:
        if upstream_revision_40hex is not None:
            raise ValueError("flat baseline cannot carry a Hugging Face revision")
        _sha1_commit(repository_source_commit_40hex, "repository source commit")
        _sha256(repository_dependency_lock_sha256, "repository dependency lock")


def baseline_implementation_id_v1(
    *,
    kind: BaselineRegistrationKindV1,
    baseline_spec_id: str,
    implementation: str,
    revision_policy: BaselineRevisionPolicyV1,
    upstream_revision_40hex: str | None,
    repository_source_commit_40hex: str | None,
    repository_dependency_lock_sha256: str | None,
    artifact_manifest_id: str,
    candidate_bindings: BaselineCallerSuppliedCandidateBindingsV1,
) -> str:
    return _content_id(
        _IMPLEMENTATION_PREFIX,
        _IMPLEMENTATION_DOMAIN,
        {
            "kind": kind.value,
            "baseline_spec_id": baseline_spec_id,
            "implementation": implementation,
            "revision_policy": revision_policy.value,
            "revision_binding": _revision_projection(
                upstream_revision_40hex=upstream_revision_40hex,
                repository_source_commit_40hex=repository_source_commit_40hex,
                repository_dependency_lock_sha256=repository_dependency_lock_sha256,
            ),
            "artifact_manifest_id": artifact_manifest_id,
            "caller_supplied_candidate_bindings": (
                _candidate_bindings_projection(candidate_bindings)
            ),
        },
    )


def _builder_unsigned(value: "BaselineRuntimeBuilderAttestationCandidateV1") -> dict[str, object]:
    return {
        "purpose": value.purpose.value,
        "key_epoch": value.key_epoch,
        "signer_key_id": value.signer_key_id,
        "profile_id": value.profile_id,
        "kind": value.kind.value,
        "baseline_spec_id": value.baseline_spec_id,
        "implementation_id": value.implementation_id,
        "artifact_manifest_id": value.artifact_manifest_id,
        "caller_supplied_candidate_bindings": (
            _candidate_bindings_projection(value.candidate_bindings)
        ),
        "builder_subject_root_sha256": value.builder_subject_root_sha256,
    }


def _builder_subject_projection(value: "BaselineRuntimeBuilderAttestationCandidateV1") -> dict[str, object]:
    return {
        "profile_id": value.profile_id,
        "kind": value.kind.value,
        "baseline_spec_id": value.baseline_spec_id,
        "implementation_id": value.implementation_id,
        "artifact_manifest_id": value.artifact_manifest_id,
        "caller_supplied_candidate_bindings": (
            _candidate_bindings_projection(value.candidate_bindings)
        ),
    }


def baseline_runtime_builder_signature_preimage_v1(
    value: "BaselineRuntimeBuilderAttestationCandidateV1",
) -> bytes:
    _validate_builder_attestation(value)
    return _signature_preimage(_BUILDER_SIGNATURE_DOMAIN, _builder_unsigned(value))


@dataclass(frozen=True, slots=True)
class BaselineRuntimeBuilderAttestationCandidateV1:
    purpose: BaselineSignaturePurposeV1
    key_epoch: int
    signer_key_id: str
    profile_id: str
    kind: BaselineRegistrationKindV1
    baseline_spec_id: str
    implementation_id: str
    artifact_manifest_id: str
    candidate_bindings: BaselineCallerSuppliedCandidateBindingsV1
    builder_subject_root_sha256: str
    signature: bytes
    attestation_id: str

    def __post_init__(self) -> None:
        _validate_builder_attestation(self)


def _validate_builder_attestation(value: BaselineRuntimeBuilderAttestationCandidateV1) -> None:
    if type(value) is not BaselineRuntimeBuilderAttestationCandidateV1:
        raise TypeError("runtime builder attestation exact type drift")
    if value.purpose is not BaselineSignaturePurposeV1.BASELINE_RUNTIME_BUILDER_ATTESTATION:
        raise ValueError("runtime builder purpose drift")
    if type(value.key_epoch) is not int or value.key_epoch != _KEY_EPOCH:
        raise ValueError("runtime builder key epoch drift")
    _prefixed_id(value.signer_key_id, _KEY_PREFIX, "runtime builder signer key ID")
    if value.profile_id != PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID:
        raise ValueError("runtime builder profile ID drift")
    if type(value.kind) is not BaselineRegistrationKindV1:
        raise TypeError("runtime builder baseline kind exact type drift")
    if value.baseline_spec_id != _BASELINE_SPEC_IDS[value.kind]:
        raise ValueError("runtime builder baseline spec ID drift")
    _prefixed_id(value.implementation_id, _IMPLEMENTATION_PREFIX, "implementation ID")
    _prefixed_id(value.artifact_manifest_id, _FILE_MANIFEST_PREFIX, "artifact manifest ID")
    if type(value.candidate_bindings) is not BaselineCallerSuppliedCandidateBindingsV1:
        raise TypeError("runtime builder candidate bindings exact type drift")
    value.candidate_bindings.__post_init__()
    expected_subject = _root(_BUILDER_SUBJECT_DOMAIN, _builder_subject_projection(value))
    if _sha256(value.builder_subject_root_sha256, "builder subject root") != expected_subject:
        raise ValueError("runtime builder subject root drift")
    signature = _raw(value.signature, _RAW_SIGNATURE_BYTES, "runtime builder signature")
    expected_id = _signed_candidate_id(
        _BUILDER_ATTESTATION_PREFIX,
        _BUILDER_ATTESTATION_DOMAIN,
        _builder_unsigned(value),
        signature,
    )
    if _prefixed_id(value.attestation_id, _BUILDER_ATTESTATION_PREFIX, "builder attestation ID") != expected_id:
        raise ValueError("runtime builder attestation ID drift")


def _qualifier_unsigned(value: "BaselineOfflineQualifierAttestationCandidateV1") -> dict[str, object]:
    return {
        "purpose": value.purpose.value,
        "key_epoch": value.key_epoch,
        "signer_key_id": value.signer_key_id,
        "profile_id": value.profile_id,
        "kind": value.kind.value,
        "baseline_spec_id": value.baseline_spec_id,
        "implementation_id": value.implementation_id,
        "builder_attestation_id": value.builder_attestation_id,
        "runtime_image_candidate_digest": value.runtime_image_candidate_digest,
        "offline_qualification_root_candidate_sha256": (
            value.offline_qualification_root_candidate_sha256
        ),
    }


def baseline_offline_qualifier_signature_preimage_v1(
    value: "BaselineOfflineQualifierAttestationCandidateV1",
) -> bytes:
    _validate_qualifier_attestation(value)
    return _signature_preimage(_QUALIFIER_SIGNATURE_DOMAIN, _qualifier_unsigned(value))


@dataclass(frozen=True, slots=True)
class BaselineOfflineQualifierAttestationCandidateV1:
    purpose: BaselineSignaturePurposeV1
    key_epoch: int
    signer_key_id: str
    profile_id: str
    kind: BaselineRegistrationKindV1
    baseline_spec_id: str
    implementation_id: str
    builder_attestation_id: str
    runtime_image_candidate_digest: str
    offline_qualification_root_candidate_sha256: str
    signature: bytes
    attestation_id: str

    def __post_init__(self) -> None:
        _validate_qualifier_attestation(self)


def _validate_qualifier_attestation(value: BaselineOfflineQualifierAttestationCandidateV1) -> None:
    if type(value) is not BaselineOfflineQualifierAttestationCandidateV1:
        raise TypeError("offline qualifier attestation exact type drift")
    if value.purpose is not BaselineSignaturePurposeV1.BASELINE_OFFLINE_QUALIFIER_ATTESTATION:
        raise ValueError("offline qualifier purpose drift")
    if type(value.key_epoch) is not int or value.key_epoch != _KEY_EPOCH:
        raise ValueError("offline qualifier key epoch drift")
    _prefixed_id(value.signer_key_id, _KEY_PREFIX, "offline qualifier signer key ID")
    if value.profile_id != PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID:
        raise ValueError("offline qualifier profile ID drift")
    if type(value.kind) is not BaselineRegistrationKindV1:
        raise TypeError("offline qualifier baseline kind exact type drift")
    if value.baseline_spec_id != _BASELINE_SPEC_IDS[value.kind]:
        raise ValueError("offline qualifier baseline spec ID drift")
    _prefixed_id(value.implementation_id, _IMPLEMENTATION_PREFIX, "implementation ID")
    _prefixed_id(value.builder_attestation_id, _BUILDER_ATTESTATION_PREFIX, "builder attestation ID")
    _image_digest(
        value.runtime_image_candidate_digest,
        "qualified runtime image candidate digest",
    )
    _sha256(
        value.offline_qualification_root_candidate_sha256,
        "offline qualification root candidate",
    )
    signature = _raw(value.signature, _RAW_SIGNATURE_BYTES, "offline qualifier signature")
    expected_id = _signed_candidate_id(
        _QUALIFIER_ATTESTATION_PREFIX,
        _QUALIFIER_ATTESTATION_DOMAIN,
        _qualifier_unsigned(value),
        signature,
    )
    if _prefixed_id(value.attestation_id, _QUALIFIER_ATTESTATION_PREFIX, "qualifier attestation ID") != expected_id:
        raise ValueError("offline qualifier attestation ID drift")


def _record_projection(value: "RichBaselineRegistrationRecordV1") -> dict[str, object]:
    return {
        "version": value.version,
        "claim_level": value.claim_level,
        "profile_id": value.profile_id,
        "protocol_id": value.protocol_id,
        "exact_freeze_id": value.exact_freeze_id,
        "owner_authorization_id": value.owner_authorization_id,
        "readiness_dag_id": value.readiness_dag_id,
        "r05_predecessor_candidate_id": value.r05_predecessor_candidate_id,
        "kind": value.kind.value,
        "baseline_spec_id": value.baseline_spec_id,
        "implementation": value.implementation,
        "revision_policy": value.revision_policy.value,
        "revision_binding": _revision_projection(
            upstream_revision_40hex=value.upstream_revision_40hex,
            repository_source_commit_40hex=value.repository_source_commit_40hex,
            repository_dependency_lock_sha256=value.repository_dependency_lock_sha256,
        ),
        "artifact_manifest_id": value.artifact_manifest.manifest_id,
        "artifact_manifest_root_sha256": value.artifact_manifest.manifest_root_sha256,
        "caller_supplied_candidate_bindings": (
            _candidate_bindings_projection(value.candidate_bindings)
        ),
        "implementation_id": value.implementation_id,
        "builder_attestation_id": value.builder_attestation_id,
        "qualifier_attestation_id": value.qualifier_attestation_id,
        "previous_final_registration_id": value.previous_final_registration_id,
        "holdout_generation_count": value.holdout_generation_count,
        "actual_case_count": value.actual_case_count,
        "campaign_id": value.campaign_id,
        "attempt_id": value.attempt_id,
    }


@dataclass(frozen=True, slots=True)
class RichBaselineRegistrationRecordV1:
    version: str
    claim_level: str
    profile_id: str
    protocol_id: str
    exact_freeze_id: str
    owner_authorization_id: str
    readiness_dag_id: str
    r05_predecessor_candidate_id: str
    kind: BaselineRegistrationKindV1
    baseline_spec_id: str
    implementation: str
    revision_policy: BaselineRevisionPolicyV1
    upstream_revision_40hex: str | None
    repository_source_commit_40hex: str | None
    repository_dependency_lock_sha256: str | None
    artifact_manifest: BaselineArtifactFileManifestV1
    candidate_bindings: BaselineCallerSuppliedCandidateBindingsV1
    implementation_id: str
    builder_attestation_id: str
    qualifier_attestation_id: str
    previous_final_registration_id: str | None
    holdout_generation_count: int
    actual_case_count: int
    campaign_id: None
    attempt_id: None
    record_id: str

    def __post_init__(self) -> None:
        _validate_record(self)


def _validate_record(value: RichBaselineRegistrationRecordV1) -> None:
    if type(value) is not RichBaselineRegistrationRecordV1:
        raise TypeError("rich registration record exact type drift")
    if value.version != PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_VERSION:
        raise ValueError("rich registration version drift")
    if value.claim_level != PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_CLAIM_LEVEL:
        raise ValueError("rich registration claim level drift")
    if value.profile_id != PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID:
        raise ValueError("rich registration profile ID drift")
    if value.protocol_id != PROTOCOL_ID_V1 or value.exact_freeze_id != EXACT_FREEZE_ID_V1:
        raise ValueError("rich registration frozen contract binding drift")
    if value.owner_authorization_id != OWNER_AUTHORIZATION_ID_V1 or value.readiness_dag_id != READINESS_DAG_ID_V1:
        raise ValueError("rich registration authority/readiness binding drift")
    _prefixed_id(value.r05_predecessor_candidate_id, _R05_CANDIDATE_PREFIX, "R05 predecessor candidate ID")
    if type(value.kind) is not BaselineRegistrationKindV1:
        raise TypeError("rich registration kind exact type drift")
    if value.baseline_spec_id != _BASELINE_SPEC_IDS[value.kind]:
        raise ValueError("rich registration baseline spec ID drift")
    if value.implementation != _IMPLEMENTATIONS[value.kind]:
        raise ValueError("rich registration implementation drift")
    if type(value.revision_policy) is not BaselineRevisionPolicyV1:
        raise TypeError("revision policy exact type drift")
    _validate_revision_binding(
        value.kind,
        value.revision_policy,
        value.upstream_revision_40hex,
        value.repository_source_commit_40hex,
        value.repository_dependency_lock_sha256,
    )
    if type(value.artifact_manifest) is not BaselineArtifactFileManifestV1:
        raise TypeError("artifact manifest exact type drift")
    value.artifact_manifest.__post_init__()
    actual_roles = frozenset(item.role for item in value.artifact_manifest.files)
    if actual_roles != _REQUIRED_ARTIFACT_ROLES[value.kind]:
        raise ValueError("baseline artifact role set is not exact")
    dependency_lock_entries = tuple(
        item
        for item in value.artifact_manifest.files
        if item.role is BaselineArtifactRoleV1.DEPENDENCY_LOCK
    )
    if value.kind is BaselineRegistrationKindV1.FLAT_LEARNED_TYPED:
        if (
            len(dependency_lock_entries) != 1
            or dependency_lock_entries[0].sha256
            != value.repository_dependency_lock_sha256
        ):
            raise ValueError("HGB dependency-lock artifact binding drift")
    elif dependency_lock_entries:
        raise ValueError("Hugging Face baseline cannot carry dependency-lock artifacts")
    if type(value.candidate_bindings) is not BaselineCallerSuppliedCandidateBindingsV1:
        raise TypeError("caller-supplied candidate bindings exact type drift")
    value.candidate_bindings.__post_init__()
    expected_implementation_id = baseline_implementation_id_v1(
        kind=value.kind,
        baseline_spec_id=value.baseline_spec_id,
        implementation=value.implementation,
        revision_policy=value.revision_policy,
        upstream_revision_40hex=value.upstream_revision_40hex,
        repository_source_commit_40hex=value.repository_source_commit_40hex,
        repository_dependency_lock_sha256=value.repository_dependency_lock_sha256,
        artifact_manifest_id=value.artifact_manifest.manifest_id,
        candidate_bindings=value.candidate_bindings,
    )
    if _prefixed_id(value.implementation_id, _IMPLEMENTATION_PREFIX, "implementation ID") != expected_implementation_id:
        raise ValueError("baseline implementation ID drift")
    _prefixed_id(value.builder_attestation_id, _BUILDER_ATTESTATION_PREFIX, "builder attestation ID")
    _prefixed_id(value.qualifier_attestation_id, _QUALIFIER_ATTESTATION_PREFIX, "qualifier attestation ID")
    if value.previous_final_registration_id is not None:
        _prefixed_id(value.previous_final_registration_id, _FINAL_REGISTRATION_PREFIX, "previous final registration ID")
    if type(value.holdout_generation_count) is not int or value.holdout_generation_count != 0:
        raise ValueError("candidate registration must predate holdout generation")
    if type(value.actual_case_count) is not int or value.actual_case_count != 0:
        raise ValueError("candidate registration cannot carry actual cases")
    if value.campaign_id is not None or value.attempt_id is not None:
        raise ValueError("candidate registration cannot allocate campaign or attempt IDs")
    expected_id = _content_id(_RECORD_PREFIX, _RECORD_DOMAIN, _record_projection(value))
    if _prefixed_id(value.record_id, _RECORD_PREFIX, "rich registration record ID") != expected_id:
        raise ValueError("rich registration record ID drift")


def rich_baseline_registration_record_bytes_v1(
    value: RichBaselineRegistrationRecordV1,
) -> bytes:
    _validate_record(value)
    return canonicalize_rfc8785_v1(_record_projection(value))


def _custodian_unsigned(value: "BaselineRegistrationCustodianAttestationCandidateV1") -> dict[str, object]:
    return {
        "purpose": value.purpose.value,
        "key_epoch": value.key_epoch,
        "signer_key_id": value.signer_key_id,
        "profile_id": value.profile_id,
        "kind": value.kind.value,
        "record_id": value.record_id,
        "r05_predecessor_candidate_id": value.r05_predecessor_candidate_id,
        "previous_final_registration_id": value.previous_final_registration_id,
    }


def baseline_registration_custodian_signature_preimage_v1(
    value: "BaselineRegistrationCustodianAttestationCandidateV1",
) -> bytes:
    _validate_custodian_attestation(value)
    return _signature_preimage(_CUSTODIAN_SIGNATURE_DOMAIN, _custodian_unsigned(value))


@dataclass(frozen=True, slots=True)
class BaselineRegistrationCustodianAttestationCandidateV1:
    purpose: BaselineSignaturePurposeV1
    key_epoch: int
    signer_key_id: str
    profile_id: str
    kind: BaselineRegistrationKindV1
    record_id: str
    r05_predecessor_candidate_id: str
    previous_final_registration_id: str | None
    signature: bytes
    attestation_id: str

    def __post_init__(self) -> None:
        _validate_custodian_attestation(self)


def _validate_custodian_attestation(value: BaselineRegistrationCustodianAttestationCandidateV1) -> None:
    if type(value) is not BaselineRegistrationCustodianAttestationCandidateV1:
        raise TypeError("custodian attestation exact type drift")
    if value.purpose is not BaselineSignaturePurposeV1.BASELINE_REGISTRATION_CUSTODIAN_ATTESTATION:
        raise ValueError("registration custodian purpose drift")
    if type(value.key_epoch) is not int or value.key_epoch != _KEY_EPOCH:
        raise ValueError("registration custodian key epoch drift")
    _prefixed_id(value.signer_key_id, _KEY_PREFIX, "custodian signer key ID")
    if value.profile_id != PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID:
        raise ValueError("registration custodian profile ID drift")
    if type(value.kind) is not BaselineRegistrationKindV1:
        raise TypeError("registration custodian kind exact type drift")
    _prefixed_id(value.record_id, _RECORD_PREFIX, "custodian record ID")
    _prefixed_id(value.r05_predecessor_candidate_id, _R05_CANDIDATE_PREFIX, "custodian R05 candidate ID")
    if value.previous_final_registration_id is not None:
        _prefixed_id(value.previous_final_registration_id, _FINAL_REGISTRATION_PREFIX, "custodian predecessor ID")
    signature = _raw(value.signature, _RAW_SIGNATURE_BYTES, "registration custodian signature")
    expected_id = _signed_candidate_id(
        _CUSTODIAN_ATTESTATION_PREFIX,
        _CUSTODIAN_ATTESTATION_DOMAIN,
        _custodian_unsigned(value),
        signature,
    )
    if _prefixed_id(value.attestation_id, _CUSTODIAN_ATTESTATION_PREFIX, "custodian attestation ID") != expected_id:
        raise ValueError("registration custodian attestation ID drift")


def _store_unsigned(value: "BaselineStoreIfAbsentReceiptCandidateV1") -> dict[str, object]:
    return {
        "purpose": value.purpose.value,
        "key_epoch": value.key_epoch,
        "signer_key_id": value.signer_key_id,
        "profile_id": value.profile_id,
        "kind": value.kind.value,
        "record_id": value.record_id,
        "custodian_attestation_id": value.custodian_attestation_id,
        "object_key": value.object_key,
        "previous_generation": value.previous_generation,
        "new_generation": value.new_generation,
        "condition": value.condition,
        "stored_content_sha256": value.stored_content_sha256,
    }


def baseline_store_if_absent_signature_preimage_v1(
    value: "BaselineStoreIfAbsentReceiptCandidateV1",
) -> bytes:
    _validate_store_receipt(value)
    return _signature_preimage(_STORE_SIGNATURE_DOMAIN, _store_unsigned(value))


@dataclass(frozen=True, slots=True)
class BaselineStoreIfAbsentReceiptCandidateV1:
    purpose: BaselineSignaturePurposeV1
    key_epoch: int
    signer_key_id: str
    profile_id: str
    kind: BaselineRegistrationKindV1
    record_id: str
    custodian_attestation_id: str
    object_key: str
    previous_generation: int
    new_generation: int
    condition: str
    stored_content_sha256: str
    signature: bytes
    receipt_id: str

    def __post_init__(self) -> None:
        _validate_store_receipt(self)


def _expected_object_key(kind: BaselineRegistrationKindV1, record_id: str) -> str:
    return f"phase2b/c1/baseline-registration/v1/{kind.value}/{record_id}.json"


def _validate_store_receipt(value: BaselineStoreIfAbsentReceiptCandidateV1) -> None:
    if type(value) is not BaselineStoreIfAbsentReceiptCandidateV1:
        raise TypeError("store receipt exact type drift")
    if value.purpose is not BaselineSignaturePurposeV1.STORE_IF_ABSENT_RECEIPT:
        raise ValueError("store receipt purpose drift")
    if type(value.key_epoch) is not int or value.key_epoch != _KEY_EPOCH:
        raise ValueError("store receipt key epoch drift")
    _prefixed_id(value.signer_key_id, _KEY_PREFIX, "store signer key ID")
    if value.profile_id != PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID:
        raise ValueError("store receipt profile ID drift")
    if type(value.kind) is not BaselineRegistrationKindV1:
        raise TypeError("store receipt kind exact type drift")
    _prefixed_id(value.record_id, _RECORD_PREFIX, "store record ID")
    _prefixed_id(value.custodian_attestation_id, _CUSTODIAN_ATTESTATION_PREFIX, "store custodian attestation ID")
    if value.object_key != _expected_object_key(value.kind, value.record_id):
        raise ValueError("store object key drift")
    if (
        type(value.previous_generation) is not int
        or type(value.new_generation) is not int
        or value.previous_generation != 0
        or value.new_generation != 1
        or value.condition != "IF_ABSENT"
    ):
        raise ValueError("STORE IF_ABSENT 0-to-1 transition drift")
    _sha256(value.stored_content_sha256, "stored record content SHA-256")
    signature = _raw(value.signature, _RAW_SIGNATURE_BYTES, "store receipt signature")
    expected_id = _signed_candidate_id(
        _STORE_RECEIPT_PREFIX,
        _STORE_RECEIPT_DOMAIN,
        _store_unsigned(value),
        signature,
    )
    if _prefixed_id(value.receipt_id, _STORE_RECEIPT_PREFIX, "store receipt ID") != expected_id:
        raise ValueError("store receipt ID drift")


def _time_unsigned(value: "BaselineDeclaredTimeTokenCandidateV1") -> dict[str, object]:
    return {
        "purpose": value.purpose.value,
        "key_epoch": value.key_epoch,
        "signer_key_id": value.signer_key_id,
        "profile_id": value.profile_id,
        "kind": value.kind.value,
        "record_id": value.record_id,
        "custodian_attestation_id": value.custodian_attestation_id,
        "store_receipt_id": value.store_receipt_id,
        "logical_sequence": value.logical_sequence,
        "previous_time_token_id": value.previous_time_token_id,
        "declared_unix_seconds": value.declared_unix_seconds,
        "declared_nanosecond": value.declared_nanosecond,
    }


def baseline_declared_time_signature_preimage_v1(
    value: "BaselineDeclaredTimeTokenCandidateV1",
) -> bytes:
    _validate_time_token(value)
    return _signature_preimage(_TIME_SIGNATURE_DOMAIN, _time_unsigned(value))


@dataclass(frozen=True, slots=True)
class BaselineDeclaredTimeTokenCandidateV1:
    purpose: BaselineSignaturePurposeV1
    key_epoch: int
    signer_key_id: str
    profile_id: str
    kind: BaselineRegistrationKindV1
    record_id: str
    custodian_attestation_id: str
    store_receipt_id: str
    logical_sequence: int
    previous_time_token_id: str | None
    declared_unix_seconds: int
    declared_nanosecond: int
    signature: bytes
    token_id: str

    def __post_init__(self) -> None:
        _validate_time_token(self)


def _validate_time_token(value: BaselineDeclaredTimeTokenCandidateV1) -> None:
    if type(value) is not BaselineDeclaredTimeTokenCandidateV1:
        raise TypeError("declared-time token exact type drift")
    if value.purpose is not BaselineSignaturePurposeV1.DECLARED_TIME_TOKEN:
        raise ValueError("declared-time purpose drift")
    if type(value.key_epoch) is not int or value.key_epoch != _KEY_EPOCH:
        raise ValueError("declared-time key epoch drift")
    _prefixed_id(value.signer_key_id, _KEY_PREFIX, "time signer key ID")
    if value.profile_id != PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID:
        raise ValueError("declared-time profile ID drift")
    if type(value.kind) is not BaselineRegistrationKindV1:
        raise TypeError("declared-time kind exact type drift")
    _prefixed_id(value.record_id, _RECORD_PREFIX, "time record ID")
    _prefixed_id(value.custodian_attestation_id, _CUSTODIAN_ATTESTATION_PREFIX, "time custodian attestation ID")
    _prefixed_id(value.store_receipt_id, _STORE_RECEIPT_PREFIX, "time store receipt ID")
    _integer(value.logical_sequence, "declared-time logical sequence")
    if value.previous_time_token_id is not None:
        _prefixed_id(value.previous_time_token_id, _TIME_TOKEN_PREFIX, "previous time token ID")
    _integer(value.declared_unix_seconds, "declared Unix seconds")
    _integer(
        value.declared_nanosecond,
        "declared nanosecond",
        maximum=999_999_999,
    )
    signature = _raw(value.signature, _RAW_SIGNATURE_BYTES, "declared-time signature")
    expected_id = _signed_candidate_id(
        _TIME_TOKEN_PREFIX,
        _TIME_TOKEN_DOMAIN,
        _time_unsigned(value),
        signature,
    )
    if _prefixed_id(value.token_id, _TIME_TOKEN_PREFIX, "declared-time token ID") != expected_id:
        raise ValueError("declared-time token ID drift")


def _final_projection(value: "FinalBaselineRegistrationCandidateV1") -> dict[str, object]:
    return {
        "profile_id": PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID,
        "kind": value.record.kind.value,
        "record_id": value.record.record_id,
        "builder_attestation_id": value.builder_attestation.attestation_id,
        "qualifier_attestation_id": value.qualifier_attestation.attestation_id,
        "custodian_attestation_id": value.custodian_attestation.attestation_id,
        "store_receipt_id": value.store_receipt.receipt_id,
        "time_token_id": value.time_token.token_id,
    }


@dataclass(frozen=True, slots=True)
class FinalBaselineRegistrationCandidateV1:
    record: RichBaselineRegistrationRecordV1
    builder_attestation: BaselineRuntimeBuilderAttestationCandidateV1
    qualifier_attestation: BaselineOfflineQualifierAttestationCandidateV1
    custodian_attestation: BaselineRegistrationCustodianAttestationCandidateV1
    store_receipt: BaselineStoreIfAbsentReceiptCandidateV1
    time_token: BaselineDeclaredTimeTokenCandidateV1
    final_registration_id: str

    def __post_init__(self) -> None:
        _validate_final_registration(self)


def _validate_final_registration(value: FinalBaselineRegistrationCandidateV1) -> None:
    if type(value) is not FinalBaselineRegistrationCandidateV1:
        raise TypeError("final registration candidate exact type drift")
    if type(value.record) is not RichBaselineRegistrationRecordV1:
        raise TypeError("final registration record exact type drift")
    if type(value.builder_attestation) is not BaselineRuntimeBuilderAttestationCandidateV1:
        raise TypeError("final builder attestation exact type drift")
    if type(value.qualifier_attestation) is not BaselineOfflineQualifierAttestationCandidateV1:
        raise TypeError("final qualifier attestation exact type drift")
    if type(value.custodian_attestation) is not BaselineRegistrationCustodianAttestationCandidateV1:
        raise TypeError("final custodian attestation exact type drift")
    if type(value.store_receipt) is not BaselineStoreIfAbsentReceiptCandidateV1:
        raise TypeError("final store receipt exact type drift")
    if type(value.time_token) is not BaselineDeclaredTimeTokenCandidateV1:
        raise TypeError("final time token exact type drift")
    value.record.__post_init__()
    _validate_builder_attestation(value.builder_attestation)
    _validate_qualifier_attestation(value.qualifier_attestation)
    _validate_custodian_attestation(value.custodian_attestation)
    _validate_store_receipt(value.store_receipt)
    _validate_time_token(value.time_token)
    record = value.record
    builder = value.builder_attestation
    qualifier = value.qualifier_attestation
    custodian = value.custodian_attestation
    store = value.store_receipt
    time = value.time_token
    if (
        builder.kind is not record.kind
        or qualifier.kind is not record.kind
        or custodian.kind is not record.kind
        or store.kind is not record.kind
        or time.kind is not record.kind
    ):
        raise ValueError("cross-component baseline kind splice")
    if builder.baseline_spec_id != record.baseline_spec_id or qualifier.baseline_spec_id != record.baseline_spec_id:
        raise ValueError("cross-component baseline spec splice")
    if builder.implementation_id != record.implementation_id or qualifier.implementation_id != record.implementation_id:
        raise ValueError("cross-component implementation splice")
    if (
        builder.artifact_manifest_id != record.artifact_manifest.manifest_id
        or builder.candidate_bindings != record.candidate_bindings
    ):
        raise ValueError("builder artifact binding splice")
    if qualifier.builder_attestation_id != builder.attestation_id:
        raise ValueError("qualifier-to-builder linkage drift")
    if (
        qualifier.runtime_image_candidate_digest
        != record.candidate_bindings.runtime_image_candidate_digest
        or qualifier.offline_qualification_root_candidate_sha256
        != record.candidate_bindings.offline_qualification_root_candidate_sha256
    ):
        raise ValueError("qualifier runtime/qualification linkage drift")
    if record.builder_attestation_id != builder.attestation_id or record.qualifier_attestation_id != qualifier.attestation_id:
        raise ValueError("rich record attestation linkage drift")
    if custodian.record_id != record.record_id or custodian.r05_predecessor_candidate_id != record.r05_predecessor_candidate_id or custodian.previous_final_registration_id != record.previous_final_registration_id:
        raise ValueError("custodian rich-record linkage drift")
    if store.record_id != record.record_id or store.custodian_attestation_id != custodian.attestation_id:
        raise ValueError("store rich-record linkage drift")
    stored_digest = hashlib.sha256(rich_baseline_registration_record_bytes_v1(record)).hexdigest()
    if store.stored_content_sha256 != stored_digest:
        raise ValueError("store content digest does not bind the rich record")
    if (
        time.record_id != record.record_id
        or time.custodian_attestation_id != custodian.attestation_id
        or time.store_receipt_id != store.receipt_id
    ):
        raise ValueError("declared-time signed-rich/store linkage drift")
    expected_id = _content_id(
        _FINAL_REGISTRATION_PREFIX,
        _FINAL_REGISTRATION_DOMAIN,
        _final_projection(value),
    )
    if _prefixed_id(value.final_registration_id, _FINAL_REGISTRATION_PREFIX, "final registration candidate ID") != expected_id:
        raise ValueError("final registration candidate ID drift")


def _legacy_projection_body(
    value: "LegacyBaselineProjectionCandidateV1",
) -> dict[str, object]:
    return {
        "kind": value.kind.value,
        "baseline_spec_id": value.baseline_spec_id,
        "implementation_id": value.implementation_id,
        "artifact_manifest_root_sha256": value.artifact_manifest_root_sha256,
        "candidate_precedes_holdout_generation": (
            value.candidate_precedes_holdout_generation
        ),
        "eligible_for_execution_freeze": value.eligible_for_execution_freeze,
        "r07_registration_verified": value.r07_registration_verified,
    }


@dataclass(frozen=True, slots=True)
class LegacyBaselineProjectionCandidateV1:
    """Ineligible projection candidate; never a consumer-accepted freeze DTO."""

    kind: BaselineRegistrationKindV1
    baseline_spec_id: str
    implementation_id: str
    artifact_manifest_root_sha256: str
    candidate_precedes_holdout_generation: bool
    eligible_for_execution_freeze: bool
    r07_registration_verified: bool
    projection_id: str

    def __post_init__(self) -> None:
        if type(self.kind) is not BaselineRegistrationKindV1:
            raise TypeError("legacy projection candidate kind exact type drift")
        if self.baseline_spec_id != _BASELINE_SPEC_IDS[self.kind]:
            raise ValueError("legacy projection candidate baseline spec drift")
        _prefixed_id(
            self.implementation_id,
            _IMPLEMENTATION_PREFIX,
            "legacy projection candidate implementation ID",
        )
        _sha256(
            self.artifact_manifest_root_sha256,
            "legacy projection candidate manifest root",
        )
        if self.candidate_precedes_holdout_generation is not True:
            raise ValueError("legacy projection candidate causal polarity drift")
        if self.eligible_for_execution_freeze is not False:
            raise ValueError("legacy projection candidate must remain ineligible")
        if self.r07_registration_verified is not False:
            raise ValueError("legacy projection candidate cannot assert R07")
        expected = _content_id(
            _LEGACY_PROJECTION_PREFIX,
            _LEGACY_PROJECTION_DOMAIN,
            _legacy_projection_body(self),
        )
        if (
            _prefixed_id(
                self.projection_id,
                _LEGACY_PROJECTION_PREFIX,
                "legacy projection candidate ID",
            )
            != expected
        ):
            raise ValueError("legacy projection candidate ID drift")


def project_rich_candidate_to_ineligible_legacy_projection_v1(
    value: RichBaselineRegistrationRecordV1,
) -> LegacyBaselineProjectionCandidateV1:
    """Derive an explicitly ineligible projection from a closed rich candidate."""

    _validate_record(value)
    body = {
        "kind": value.kind.value,
        "baseline_spec_id": value.baseline_spec_id,
        "implementation_id": value.implementation_id,
        "artifact_manifest_root_sha256": (
            value.artifact_manifest.manifest_root_sha256
        ),
        "candidate_precedes_holdout_generation": True,
        "eligible_for_execution_freeze": False,
        "r07_registration_verified": False,
    }
    return LegacyBaselineProjectionCandidateV1(
        kind=value.kind,
        baseline_spec_id=value.baseline_spec_id,
        implementation_id=value.implementation_id,
        artifact_manifest_root_sha256=value.artifact_manifest.manifest_root_sha256,
        candidate_precedes_holdout_generation=True,
        eligible_for_execution_freeze=False,
        r07_registration_verified=False,
        projection_id=_content_id(
            _LEGACY_PROJECTION_PREFIX,
            _LEGACY_PROJECTION_DOMAIN,
            body,
        ),
    )


def _verify_ed25519(public_key: bytes, signature: bytes, message: bytes) -> None:
    if _Ed25519PublicKey is None:
        raise _SignatureBackendUnavailable
    try:
        _Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except _InvalidSignature as exc:
        raise _SignatureInvalid from exc
    except (TypeError, ValueError) as exc:
        raise _SignatureInvalid from exc


class _SignatureBackendUnavailable(Exception):
    pass


class _SignatureInvalid(Exception):
    pass


class _InputMutationDetected(Exception):
    pass


_TRUE_CLAIMS: Final = (
    "profile_and_exact_baseline_order_verified",
    "rich_record_field_closure_verified",
    "artifact_file_manifest_roots_recomputed",
    "artifact_file_required_roles_verified",
    "caller_supplied_candidate_binding_shapes_verified",
    "exact_revision_bindings_verified",
    "five_purpose_distinct_candidate_keys_verified",
    "fifteen_caller_supplied_ed25519_signatures_verified",
    "runtime_builder_attestations_mechanically_verified",
    "offline_qualifier_attestations_mechanically_verified",
    "registration_custodian_attestations_mechanically_verified",
    "store_if_absent_zero_to_one_candidates_verified",
    "declared_time_sequence_and_predecessor_chain_verified",
    "cross_baseline_predecessor_chain_verified",
    "ineligible_legacy_projection_candidates_derived",
    "aggregate_candidate_registry_content_address_verified",
    "input_snapshot_commitment_derived",
    "cooperative_computation_only",
    "downstream_fresh_reverification_required",
    "atomic_fail_closed_rejection_verified",
)

_FALSE_CLAIMS: Final = (
    "r05_authority_verified",
    "r07_baseline_registrations_frozen",
    "r08_statistical_validation_unlocked",
    "aggregate_snapshot_persisted",
    "baseline_registrations_exist",
    "artifact_roots_replayed",
    "result_object_unforgeable",
    "external_trust_anchor_verified",
    "key_registry_authority_verified",
    "runtime_builder_identity_verified",
    "offline_qualifier_identity_verified",
    "registration_custodian_identity_verified",
    "signer_organizational_independence_verified",
    "external_store_identity_verified",
    "external_store_authority_verified",
    "durable_cas_verified",
    "rollback_resistance_verified",
    "authoritative_time_source_verified",
    "authoritative_time_verified",
    "timeline_observed",
    "campaign_id_allocated",
    "attempt_id_allocated",
    "actual_holdout_generated",
    "actual_execution_claimed",
    "network_accessed",
    "docker_invoked",
    "keys_generated",
    "signatures_generated",
)


@dataclass(frozen=True, slots=True)
class ThreeBaselineRegistrationMechanicsV1:
    disposition: ThreeBaselineRegistrationDispositionV1
    reason: ThreeBaselineRegistrationReasonV1
    version: str
    claim_level: str
    profile_id: str
    result_id: str
    candidate_registry_id: str
    candidate_registry_root_sha256: str
    input_snapshot_commitment_sha256: str
    final_registration_candidate_ids: tuple[str, ...]
    legacy_projection_candidates: tuple[LegacyBaselineProjectionCandidateV1, ...]
    required_signature_count: int
    verified_signature_count: int
    profile_and_exact_baseline_order_verified: bool
    rich_record_field_closure_verified: bool
    artifact_file_manifest_roots_recomputed: bool
    artifact_file_required_roles_verified: bool
    caller_supplied_candidate_binding_shapes_verified: bool
    exact_revision_bindings_verified: bool
    five_purpose_distinct_candidate_keys_verified: bool
    fifteen_caller_supplied_ed25519_signatures_verified: bool
    runtime_builder_attestations_mechanically_verified: bool
    offline_qualifier_attestations_mechanically_verified: bool
    registration_custodian_attestations_mechanically_verified: bool
    store_if_absent_zero_to_one_candidates_verified: bool
    declared_time_sequence_and_predecessor_chain_verified: bool
    cross_baseline_predecessor_chain_verified: bool
    ineligible_legacy_projection_candidates_derived: bool
    aggregate_candidate_registry_content_address_verified: bool
    input_snapshot_commitment_derived: bool
    cooperative_computation_only: bool
    downstream_fresh_reverification_required: bool
    atomic_fail_closed_rejection_verified: bool
    r05_authority_verified: bool
    r07_baseline_registrations_frozen: bool
    r08_statistical_validation_unlocked: bool
    aggregate_snapshot_persisted: bool
    baseline_registrations_exist: bool
    artifact_roots_replayed: bool
    result_object_unforgeable: bool
    external_trust_anchor_verified: bool
    key_registry_authority_verified: bool
    runtime_builder_identity_verified: bool
    offline_qualifier_identity_verified: bool
    registration_custodian_identity_verified: bool
    signer_organizational_independence_verified: bool
    external_store_identity_verified: bool
    external_store_authority_verified: bool
    durable_cas_verified: bool
    rollback_resistance_verified: bool
    authoritative_time_source_verified: bool
    authoritative_time_verified: bool
    timeline_observed: bool
    campaign_id_allocated: bool
    attempt_id_allocated: bool
    actual_holdout_generated: bool
    actual_execution_claimed: bool
    network_accessed: bool
    docker_invoked: bool
    keys_generated: bool
    signatures_generated: bool


@dataclass(frozen=True, slots=True)
class ThreeBaselineRegistrationRejectionV1:
    disposition: ThreeBaselineRegistrationDispositionV1
    reason: ThreeBaselineRegistrationReasonV1
    version: None
    claim_level: None
    profile_id: None
    result_id: None
    candidate_registry_id: None
    candidate_registry_root_sha256: None
    input_snapshot_commitment_sha256: None
    final_registration_candidate_ids: tuple[()]
    legacy_projection_candidates: tuple[()]
    required_signature_count: int
    verified_signature_count: int
    partial_output_published: bool


@dataclass(frozen=True, slots=True)
class _FrozenValueSnapshotV1:
    tree: tuple[object, ...]
    canonical_bytes: bytes
    node_count: int
    utf8_bytes: int
    raw_bytes: int
    tuple_entry_count: int
    max_depth: int
    canonical_upper_bound: int


@dataclass(frozen=True, slots=True)
class _FrozenGraphSnapshotV1(_FrozenValueSnapshotV1):
    commitment_sha256: str


@dataclass(slots=True)
class _SnapshotBudgetV1:
    node_count: int = 0
    utf8_bytes: int = 0
    raw_bytes: int = 0
    tuple_entry_count: int = 0
    max_depth: int = 0


class _SnapshotBudgetExceeded(ValueError):
    pass


def _snapshot_dataclass_types() -> tuple[type[object], ...]:
    return (
        BaselineArtifactFileV1,
        BaselineArtifactFileManifestV1,
        BaselineCallerSuppliedCandidateBindingsV1,
        R05QualificationPredecessorCandidateV1,
        BaselineCandidatePublicKeyV1,
        BaselineRuntimeBuilderAttestationCandidateV1,
        BaselineOfflineQualifierAttestationCandidateV1,
        RichBaselineRegistrationRecordV1,
        BaselineRegistrationCustodianAttestationCandidateV1,
        BaselineStoreIfAbsentReceiptCandidateV1,
        BaselineDeclaredTimeTokenCandidateV1,
        FinalBaselineRegistrationCandidateV1,
        LegacyBaselineProjectionCandidateV1,
        ThreeBaselineRegistrationMechanicsV1,
        ThreeBaselineRegistrationRejectionV1,
    )


def _snapshot_enum_types() -> tuple[type[Enum], ...]:
    return (
        BaselineRegistrationKindV1,
        BaselineRevisionPolicyV1,
        BaselineArtifactRoleV1,
        BaselineSignaturePurposeV1,
        ThreeBaselineRegistrationDispositionV1,
        ThreeBaselineRegistrationReasonV1,
    )


def _freeze_snapshot_node(
    value: object,
    *,
    depth: int,
    active_ids: set[int],
    budget: _SnapshotBudgetV1,
) -> tuple[object, ...]:
    """Bound and type-check each node before copying or descending."""

    if type(depth) is not int or not 0 <= depth <= _MAX_SNAPSHOT_DEPTH:
        raise _SnapshotBudgetExceeded("snapshot depth bound exceeded")
    budget.max_depth = max(budget.max_depth, depth)
    budget.node_count += 1
    if budget.node_count > _MAX_SNAPSHOT_NODES:
        raise _SnapshotBudgetExceeded("snapshot node bound exceeded")
    value_type = type(value)
    if value is None:
        return ("none",)
    if value_type is bool:
        return ("bool", value)
    if value_type is int:
        _integer(value, "snapshot integer")
        return ("int", value)
    if value_type is str:
        if len(value) > _MAX_TEXT_BYTES:
            raise _SnapshotBudgetExceeded("snapshot text bound exceeded")
        encoded = value.encode("utf-8")
        if len(encoded) > _MAX_TEXT_BYTES:
            raise _SnapshotBudgetExceeded("snapshot text bound exceeded")
        budget.utf8_bytes += len(encoded)
        if (
            budget.utf8_bytes > _MAX_SNAPSHOT_TOTAL_UTF8_BYTES
            or budget.utf8_bytes + budget.raw_bytes
            > _MAX_SNAPSHOT_TOTAL_LEAF_BYTES
        ):
            raise _SnapshotBudgetExceeded("snapshot cumulative text bound exceeded")
        return ("str", value)
    if value_type is bytes:
        if len(value) > _MAX_SNAPSHOT_BYTES:
            raise _SnapshotBudgetExceeded("snapshot bytes bound exceeded")
        budget.raw_bytes += len(value)
        if (
            budget.raw_bytes > _MAX_SNAPSHOT_TOTAL_RAW_BYTES
            or budget.utf8_bytes + budget.raw_bytes
            > _MAX_SNAPSHOT_TOTAL_LEAF_BYTES
        ):
            raise _SnapshotBudgetExceeded("snapshot cumulative bytes bound exceeded")
        copied = memoryview(value).tobytes()
        return ("bytes", copied)
    enum_types = _snapshot_enum_types()
    if value_type in enum_types:
        enum_value = object.__getattribute__(value, "_value_")
        if type(enum_value) is not str:
            raise TypeError("snapshot enum value type drift")
        if len(enum_value) > _MAX_TEXT_BYTES:
            raise _SnapshotBudgetExceeded("snapshot enum value bound exceeded")
        encoded = enum_value.encode("utf-8")
        if not 1 <= len(encoded) <= _MAX_TEXT_BYTES:
            raise _SnapshotBudgetExceeded("snapshot enum value bound exceeded")
        budget.utf8_bytes += len(encoded)
        if (
            budget.utf8_bytes > _MAX_SNAPSHOT_TOTAL_UTF8_BYTES
            or budget.utf8_bytes + budget.raw_bytes
            > _MAX_SNAPSHOT_TOTAL_LEAF_BYTES
        ):
            raise _SnapshotBudgetExceeded("snapshot cumulative text bound exceeded")
        return ("enum", value_type.__name__, enum_value)
    if value_type is tuple:
        if len(value) > _MAX_SNAPSHOT_TUPLE_ITEMS:
            raise _SnapshotBudgetExceeded("snapshot tuple bound exceeded")
        budget.tuple_entry_count += len(value)
        if budget.tuple_entry_count > _MAX_SNAPSHOT_TOTAL_TUPLE_ENTRIES:
            raise _SnapshotBudgetExceeded("snapshot cumulative tuple bound exceeded")
        object_id = id(value)
        if object_id in active_ids:
            raise ValueError("snapshot tuple cycle rejected")
        active_ids.add(object_id)
        try:
            children = tuple(
                _freeze_snapshot_node(
                    item,
                    depth=depth + 1,
                    active_ids=active_ids,
                    budget=budget,
                )
                for item in value
            )
        finally:
            active_ids.remove(object_id)
        return ("tuple", children)
    dataclass_types = _snapshot_dataclass_types()
    if value_type in dataclass_types:
        object_id = id(value)
        if object_id in active_ids:
            raise ValueError("snapshot dataclass cycle rejected")
        active_ids.add(object_id)
        try:
            frozen_fields = []
            for item in fields(value_type):
                field_value = object.__getattribute__(value, item.name)
                frozen_fields.append(
                    (
                        item.name,
                        _freeze_snapshot_node(
                            field_value,
                            depth=depth + 1,
                            active_ids=active_ids,
                            budget=budget,
                        ),
                    )
                )
        finally:
            active_ids.remove(object_id)
        return ("dataclass", value_type.__name__, tuple(frozen_fields))
    raise TypeError("snapshot exact input type rejected")


def _snapshot_json_value(node: tuple[object, ...]) -> object:
    tag = node[0]
    if tag == "none":
        return {"type": "none"}
    if tag in {"bool", "int", "str"}:
        return {"type": tag, "value": node[1]}
    if tag == "bytes":
        return {"type": "bytes", "value_hex": node[1].hex()}
    if tag == "enum":
        return {"type": "enum", "class": node[1], "value": node[2]}
    if tag == "tuple":
        return {
            "type": "tuple",
            "items": [_snapshot_json_value(item) for item in node[1]],
        }
    if tag == "dataclass":
        return {
            "type": "dataclass",
            "class": node[1],
            "fields": [
                {"name": name, "value": _snapshot_json_value(child)}
                for name, child in node[2]
            ],
        }
    raise ValueError("trusted snapshot tag drift")


def _snapshot_canonical_upper_bound(budget: _SnapshotBudgetV1) -> int:
    if type(budget) is not _SnapshotBudgetV1:
        raise TypeError("snapshot budget exact type drift")
    upper_bound = (
        4_096
        + budget.utf8_bytes * 6
        + budget.raw_bytes * 2
        + budget.node_count * 256
        + budget.tuple_entry_count * 16
    )
    if upper_bound > _MAX_SNAPSHOT_CANONICAL_BYTES:
        raise _SnapshotBudgetExceeded("snapshot canonical byte bound exceeded")
    return upper_bound


def _freeze_bounded_value(value: object) -> _FrozenValueSnapshotV1:
    budget = _SnapshotBudgetV1()
    tree = _freeze_snapshot_node(
        value,
        depth=0,
        active_ids=set(),
        budget=budget,
    )
    canonical_upper_bound = _snapshot_canonical_upper_bound(budget)
    # The conservative upper bound is checked before materializing the tagged
    # JSON projection or invoking RFC 8785 canonicalization.
    canonical_bytes = canonicalize_rfc8785_v1(_snapshot_json_value(tree))
    if (
        len(canonical_bytes) > canonical_upper_bound
        or len(canonical_bytes) > _MAX_SNAPSHOT_CANONICAL_BYTES
    ):
        raise _SnapshotBudgetExceeded("snapshot canonical estimator drift")
    return _FrozenValueSnapshotV1(
        tree=tree,
        canonical_bytes=memoryview(canonical_bytes).tobytes(),
        node_count=budget.node_count,
        utf8_bytes=budget.utf8_bytes,
        raw_bytes=budget.raw_bytes,
        tuple_entry_count=budget.tuple_entry_count,
        max_depth=budget.max_depth,
        canonical_upper_bound=canonical_upper_bound,
    )


def _measure_frozen_tree_nonrecursive(
    tree: object,
) -> tuple[_SnapshotBudgetV1, int]:
    """Validate and measure a tagged tree before recursive conversion."""

    budget = _SnapshotBudgetV1()
    stack: list[tuple[object, int]] = [(tree, 0)]
    dataclass_by_name = {
        item.__name__: item for item in _snapshot_dataclass_types()
    }
    enum_by_name = {item.__name__: item for item in _snapshot_enum_types()}

    def consume_text(value: object, *, allow_empty: bool) -> None:
        if type(value) is not str or len(value) > _MAX_TEXT_BYTES:
            raise TypeError("frozen snapshot text type or size drift")
        encoded = value.encode("utf-8")
        if len(encoded) > _MAX_TEXT_BYTES or (not allow_empty and not encoded):
            raise _SnapshotBudgetExceeded("frozen snapshot text bound exceeded")
        budget.utf8_bytes += len(encoded)
        if (
            budget.utf8_bytes > _MAX_SNAPSHOT_TOTAL_UTF8_BYTES
            or budget.utf8_bytes + budget.raw_bytes
            > _MAX_SNAPSHOT_TOTAL_LEAF_BYTES
        ):
            raise _SnapshotBudgetExceeded(
                "frozen snapshot cumulative text bound exceeded"
            )

    while stack:
        node, depth = stack.pop()
        if type(depth) is not int or not 0 <= depth <= _MAX_SNAPSHOT_DEPTH:
            raise _SnapshotBudgetExceeded("frozen snapshot depth bound exceeded")
        if type(node) is not tuple or not node or type(node[0]) is not str:
            raise TypeError("frozen snapshot tagged node type drift")
        budget.max_depth = max(budget.max_depth, depth)
        budget.node_count += 1
        if budget.node_count > _MAX_SNAPSHOT_NODES:
            raise _SnapshotBudgetExceeded("frozen snapshot node bound exceeded")
        tag = node[0]
        if len(tag) > 16:
            raise ValueError("frozen snapshot tag size drift")
        if tag == "none":
            if len(node) != 1:
                raise ValueError("frozen none node shape drift")
        elif tag == "bool":
            if len(node) != 2 or type(node[1]) is not bool:
                raise TypeError("frozen bool node shape drift")
        elif tag == "int":
            if len(node) != 2:
                raise ValueError("frozen int node shape drift")
            _integer(node[1], "frozen snapshot integer")
        elif tag == "str":
            if len(node) != 2:
                raise ValueError("frozen string node shape drift")
            consume_text(node[1], allow_empty=True)
        elif tag == "bytes":
            if (
                len(node) != 2
                or type(node[1]) is not bytes
                or len(node[1]) > _MAX_SNAPSHOT_BYTES
            ):
                raise TypeError("frozen bytes node shape drift")
            budget.raw_bytes += len(node[1])
            if (
                budget.raw_bytes > _MAX_SNAPSHOT_TOTAL_RAW_BYTES
                or budget.utf8_bytes + budget.raw_bytes
                > _MAX_SNAPSHOT_TOTAL_LEAF_BYTES
            ):
                raise _SnapshotBudgetExceeded(
                    "frozen snapshot cumulative bytes bound exceeded"
                )
        elif tag == "enum":
            if (
                len(node) != 3
                or type(node[1]) is not str
                or len(node[1]) > 128
            ):
                raise TypeError("frozen enum node shape drift")
            enum_cls = enum_by_name.get(node[1])
            if enum_cls is None or type(node[2]) is not str:
                raise ValueError("frozen enum class drift")
            enum_cls(node[2])
            consume_text(node[2], allow_empty=False)
        elif tag == "tuple":
            if len(node) != 2 or type(node[1]) is not tuple:
                raise TypeError("frozen tuple node shape drift")
            children = node[1]
            if len(children) > _MAX_SNAPSHOT_TUPLE_ITEMS:
                raise _SnapshotBudgetExceeded("frozen tuple bound exceeded")
            budget.tuple_entry_count += len(children)
            if budget.tuple_entry_count > _MAX_SNAPSHOT_TOTAL_TUPLE_ENTRIES:
                raise _SnapshotBudgetExceeded(
                    "frozen cumulative tuple bound exceeded"
                )
            for child in reversed(children):
                stack.append((child, depth + 1))
        elif tag == "dataclass":
            if (
                len(node) != 3
                or type(node[1]) is not str
                or len(node[1]) > 128
                or type(node[2]) is not tuple
            ):
                raise TypeError("frozen dataclass node shape drift")
            cls = dataclass_by_name.get(node[1])
            if cls is None:
                raise ValueError("frozen dataclass class drift")
            expected_fields = fields(cls)
            if len(node[2]) != len(expected_fields):
                raise ValueError("frozen dataclass field count drift")
            for index in range(len(expected_fields) - 1, -1, -1):
                pair = node[2][index]
                expected_field = expected_fields[index]
                if (
                    type(pair) is not tuple
                    or len(pair) != 2
                    or type(pair[0]) is not str
                    or len(pair[0]) > 128
                    or pair[0] != expected_field.name
                ):
                    raise ValueError("frozen dataclass field order drift")
                stack.append((pair[1], depth + 1))
        else:
            raise ValueError("frozen snapshot tag drift")
    return budget, _snapshot_canonical_upper_bound(budget)


def _freeze_graph_inputs(
    r05_predecessor: object,
    candidate_keys: object,
    final_registrations: object,
) -> _FrozenGraphSnapshotV1:
    frozen = _freeze_bounded_value(
        (r05_predecessor, candidate_keys, final_registrations)
    )
    commitment = hashlib.sha256(
        _SNAPSHOT_DOMAIN + frozen.canonical_bytes
    ).hexdigest()
    return _FrozenGraphSnapshotV1(
        tree=frozen.tree,
        canonical_bytes=frozen.canonical_bytes,
        node_count=frozen.node_count,
        utf8_bytes=frozen.utf8_bytes,
        raw_bytes=frozen.raw_bytes,
        tuple_entry_count=frozen.tuple_entry_count,
        max_depth=frozen.max_depth,
        canonical_upper_bound=frozen.canonical_upper_bound,
        commitment_sha256=commitment,
    )


def _thaw_snapshot_node(node: tuple[object, ...]) -> object:
    tag = node[0]
    if tag == "none":
        return None
    if tag in {"bool", "int", "str"}:
        return node[1]
    if tag == "bytes":
        return memoryview(node[1]).tobytes()
    if tag == "enum":
        enum_by_name = {item.__name__: item for item in _snapshot_enum_types()}
        return enum_by_name[node[1]](node[2])
    if tag == "tuple":
        return tuple(_thaw_snapshot_node(item) for item in node[1])
    if tag == "dataclass":
        dataclass_by_name = {
            item.__name__: item for item in _snapshot_dataclass_types()
        }
        cls = dataclass_by_name[node[1]]
        kwargs = {name: _thaw_snapshot_node(child) for name, child in node[2]}
        return cls(**kwargs)
    raise ValueError("trusted snapshot tag drift")


def _thaw_graph_inputs(
    snapshot: _FrozenGraphSnapshotV1,
) -> tuple[
    R05QualificationPredecessorCandidateV1,
    tuple[BaselineCandidatePublicKeyV1, ...],
    tuple[FinalBaselineRegistrationCandidateV1, ...],
]:
    if type(snapshot) is not _FrozenGraphSnapshotV1:
        raise TypeError("frozen graph snapshot exact type drift")
    # A cooperative caller can reach private names in Python.  Re-apply the
    # exact-type/depth/node/tuple/text/bytes/integer bounds to the tagged tree
    # before any recursive JSON conversion or thawing.
    measured, measured_upper_bound = _measure_frozen_tree_nonrecursive(
        snapshot.tree
    )
    if (
        snapshot.node_count != measured.node_count
        or snapshot.utf8_bytes != measured.utf8_bytes
        or snapshot.raw_bytes != measured.raw_bytes
        or snapshot.tuple_entry_count != measured.tuple_entry_count
        or snapshot.max_depth != measured.max_depth
        or snapshot.canonical_upper_bound != measured_upper_bound
    ):
        raise ValueError("frozen graph snapshot measurement drift")
    expected_bytes = canonicalize_rfc8785_v1(_snapshot_json_value(snapshot.tree))
    expected_commitment = hashlib.sha256(_SNAPSHOT_DOMAIN + expected_bytes).hexdigest()
    if (
        type(snapshot.canonical_bytes) is not bytes
        or len(snapshot.canonical_bytes) > _MAX_SNAPSHOT_CANONICAL_BYTES
        or snapshot.canonical_bytes != expected_bytes
        or _sha256(snapshot.commitment_sha256, "input snapshot commitment")
        != expected_commitment
        or not 1 <= snapshot.node_count
    ):
        raise ValueError("frozen graph snapshot integrity drift")
    thawed = _thaw_snapshot_node(snapshot.tree)
    if type(thawed) is not tuple or len(thawed) != 3:
        raise TypeError("frozen graph root shape drift")
    r05_predecessor, candidate_keys, final_registrations = thawed
    if type(r05_predecessor) is not R05QualificationPredecessorCandidateV1:
        raise TypeError("R05 snapshot type drift")
    if type(candidate_keys) is not tuple or type(final_registrations) is not tuple:
        raise TypeError("candidate graph snapshot tuple type drift")
    return r05_predecessor, candidate_keys, final_registrations


def _assert_graph_inputs_unchanged(
    snapshot: _FrozenGraphSnapshotV1,
    *,
    r05_predecessor: object,
    candidate_keys: object,
    final_registrations: object,
) -> None:
    try:
        replay = _freeze_graph_inputs(
            r05_predecessor,
            candidate_keys,
            final_registrations,
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise _InputMutationDetected from exc
    if (
        replay.tree != snapshot.tree
        or replay.canonical_bytes != snapshot.canonical_bytes
        or replay.commitment_sha256 != snapshot.commitment_sha256
        or replay.node_count != snapshot.node_count
        or replay.utf8_bytes != snapshot.utf8_bytes
        or replay.raw_bytes != snapshot.raw_bytes
        or replay.tuple_entry_count != snapshot.tuple_entry_count
        or replay.max_depth != snapshot.max_depth
        or replay.canonical_upper_bound != snapshot.canonical_upper_bound
    ):
        raise _InputMutationDetected


def _freeze_result_for_comparison(value: object) -> _FrozenValueSnapshotV1:
    return _freeze_bounded_value(value)


def _rejection(
    reason: ThreeBaselineRegistrationReasonV1,
) -> ThreeBaselineRegistrationRejectionV1:
    return ThreeBaselineRegistrationRejectionV1(
        disposition=ThreeBaselineRegistrationDispositionV1.REJECTED,
        reason=reason,
        version=None,
        claim_level=None,
        profile_id=None,
        result_id=None,
        candidate_registry_id=None,
        candidate_registry_root_sha256=None,
        input_snapshot_commitment_sha256=None,
        final_registration_candidate_ids=(),
        legacy_projection_candidates=(),
        required_signature_count=_EXPECTED_SIGNATURES,
        verified_signature_count=0,
        partial_output_published=False,
    )


def _verify_key_registry(keys: tuple[BaselineCandidatePublicKeyV1, ...]) -> dict[BaselineSignaturePurposeV1, BaselineCandidatePublicKeyV1]:
    if type(keys) is not tuple or len(keys) != len(_PURPOSE_ORDER):
        raise TypeError("candidate key registry must be an exact five-item tuple")
    if any(type(item) is not BaselineCandidatePublicKeyV1 for item in keys):
        raise TypeError("candidate key exact type drift")
    for item in keys:
        item.__post_init__()
    if tuple(item.purpose for item in keys) != _PURPOSE_ORDER:
        raise ValueError("candidate key purpose order drift")
    if len({item.key_id for item in keys}) != len(keys):
        raise ValueError("candidate key IDs must be pairwise distinct")
    if len({item.public_key for item in keys}) != len(keys):
        raise ValueError("candidate public keys must be pairwise distinct")
    return {item.purpose: item for item in keys}


def _fresh_verify_five_signatures(
    value: FinalBaselineRegistrationCandidateV1,
    key_by_purpose: dict[BaselineSignaturePurposeV1, BaselineCandidatePublicKeyV1],
    after_each_verify: Callable[[], None],
) -> None:
    candidates = (
        (
            value.builder_attestation,
            BaselineSignaturePurposeV1.BASELINE_RUNTIME_BUILDER_ATTESTATION,
            baseline_runtime_builder_signature_preimage_v1(value.builder_attestation),
        ),
        (
            value.qualifier_attestation,
            BaselineSignaturePurposeV1.BASELINE_OFFLINE_QUALIFIER_ATTESTATION,
            baseline_offline_qualifier_signature_preimage_v1(value.qualifier_attestation),
        ),
        (
            value.custodian_attestation,
            BaselineSignaturePurposeV1.BASELINE_REGISTRATION_CUSTODIAN_ATTESTATION,
            baseline_registration_custodian_signature_preimage_v1(value.custodian_attestation),
        ),
        (
            value.store_receipt,
            BaselineSignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
            baseline_store_if_absent_signature_preimage_v1(value.store_receipt),
        ),
        (
            value.time_token,
            BaselineSignaturePurposeV1.DECLARED_TIME_TOKEN,
            baseline_declared_time_signature_preimage_v1(value.time_token),
        ),
    )
    for candidate, purpose, preimage in candidates:
        key = key_by_purpose[purpose]
        if candidate.signer_key_id != key.key_id or candidate.key_epoch != key.key_epoch:
            raise ValueError("signature candidate key/purpose binding drift")
        _verify_ed25519(key.public_key, candidate.signature, preimage)
        after_each_verify()


def _guard_graph_top_level_shape(
    r05_predecessor: object,
    candidate_keys: object,
    final_registrations: object,
) -> None:
    """Check the fixed graph shell without recursively walking caller data."""

    if type(r05_predecessor) is not R05QualificationPredecessorCandidateV1:
        raise TypeError("R05 top-level exact type drift")
    if type(candidate_keys) is not tuple or type(final_registrations) is not tuple:
        raise TypeError("candidate graph top-level tuple type drift")
    if len(candidate_keys) != len(_PURPOSE_ORDER):
        raise ValueError("candidate key top-level length drift")
    if len(final_registrations) != len(_BASELINE_ORDER):
        raise ValueError("final registration top-level length drift")
    for index, purpose in enumerate(_PURPOSE_ORDER):
        key = candidate_keys[index]
        if type(key) is not BaselineCandidatePublicKeyV1:
            raise TypeError("candidate key top-level exact type drift")
        if object.__getattribute__(key, "purpose") is not purpose:
            raise ValueError("candidate key top-level purpose order drift")
    for index, kind in enumerate(_BASELINE_ORDER):
        final = final_registrations[index]
        if type(final) is not FinalBaselineRegistrationCandidateV1:
            raise TypeError("final registration top-level exact type drift")
        record = object.__getattribute__(final, "record")
        if type(record) is not RichBaselineRegistrationRecordV1:
            raise TypeError("rich record top-level exact type drift")
        if object.__getattribute__(record, "kind") is not kind:
            raise ValueError("final registration top-level baseline order drift")


def _bounded_top_text_equals(actual: object, expected: str) -> bool:
    if type(actual) is not str or len(actual) > _MAX_TEXT_BYTES:
        return False
    try:
        if len(actual.encode("utf-8")) > _MAX_TEXT_BYTES:
            return False
    except UnicodeEncodeError:
        return False
    return actual == expected


def _projection_top_level_equals(
    actual: object,
    expected: LegacyBaselineProjectionCandidateV1,
) -> bool:
    if type(actual) is not LegacyBaselineProjectionCandidateV1:
        return False
    try:
        for item in fields(LegacyBaselineProjectionCandidateV1):
            actual_value = object.__getattribute__(actual, item.name)
            expected_value = object.__getattribute__(expected, item.name)
            if type(expected_value) is str:
                if not _bounded_top_text_equals(actual_value, expected_value):
                    return False
            elif isinstance(expected_value, Enum):
                if type(actual_value) is not type(expected_value) or actual_value is not expected_value:
                    return False
            elif type(expected_value) is bool:
                if actual_value is not expected_value:
                    return False
            else:
                return False
    except AttributeError:
        return False
    return True


def _result_top_level_equals(
    actual: object,
    expected: ThreeBaselineRegistrationMechanicsV1,
) -> bool:
    """Exact fixed-field comparison before any result snapshot traversal."""

    if type(actual) is not ThreeBaselineRegistrationMechanicsV1:
        return False
    try:
        for item in fields(ThreeBaselineRegistrationMechanicsV1):
            actual_value = object.__getattribute__(actual, item.name)
            expected_value = object.__getattribute__(expected, item.name)
            if type(expected_value) is str:
                if not _bounded_top_text_equals(actual_value, expected_value):
                    return False
            elif isinstance(expected_value, Enum):
                if type(actual_value) is not type(expected_value) or actual_value is not expected_value:
                    return False
            elif type(expected_value) is bool:
                if actual_value is not expected_value:
                    return False
            elif type(expected_value) is int:
                if type(actual_value) is not int or actual_value != expected_value:
                    return False
            elif type(expected_value) is tuple:
                if type(actual_value) is not tuple or len(actual_value) != len(expected_value):
                    return False
                if item.name == "final_registration_candidate_ids":
                    if any(
                        not _bounded_top_text_equals(actual_item, expected_item)
                        for actual_item, expected_item in zip(
                            actual_value,
                            expected_value,
                            strict=True,
                        )
                    ):
                        return False
                elif item.name == "legacy_projection_candidates":
                    if any(
                        not _projection_top_level_equals(actual_item, expected_item)
                        for actual_item, expected_item in zip(
                            actual_value,
                            expected_value,
                            strict=True,
                        )
                    ):
                        return False
                else:
                    return False
            else:
                return False
    except AttributeError:
        return False
    return True


def _success(snapshot: _FrozenGraphSnapshotV1) -> ThreeBaselineRegistrationMechanicsV1:
    """Derive a cooperative result only from one complete frozen snapshot."""

    r05_predecessor, _candidate_keys, final_registrations = _thaw_graph_inputs(
        snapshot
    )
    final_ids = tuple(item.final_registration_id for item in final_registrations)
    registry_root = _root(_REGISTRY_ROOT_DOMAIN, list(final_ids))
    registry_id = _content_id(
        _REGISTRY_PREFIX,
        _REGISTRY_ID_DOMAIN,
        {
            "profile_id": PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID,
            "r05_predecessor_candidate_id": r05_predecessor.candidate_id,
            "final_registration_candidate_ids": list(final_ids),
            "candidate_registry_root_sha256": registry_root,
        },
    )
    projections = tuple(
        project_rich_candidate_to_ineligible_legacy_projection_v1(item.record)
        for item in final_registrations
    )
    result_body = {
        "disposition": (
            ThreeBaselineRegistrationDispositionV1.MECHANICS_ONLY_NOT_REGISTRATIONS.value
        ),
        "reason": (
            ThreeBaselineRegistrationReasonV1.CALLER_SUPPLIED_CANDIDATE_SIGNATURE_MECHANICS_VERIFIED.value
        ),
        "version": PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_VERSION,
        "claim_level": PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_CLAIM_LEVEL,
        "profile_id": PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID,
        "candidate_registry_id": registry_id,
        "candidate_registry_root_sha256": registry_root,
        "input_snapshot_commitment_sha256": snapshot.commitment_sha256,
        "final_registration_candidate_ids": list(final_ids),
        "legacy_projection_candidate_ids": [
            item.projection_id for item in projections
        ],
        "required_signature_count": _EXPECTED_SIGNATURES,
        "verified_signature_count": _EXPECTED_SIGNATURES,
        "true_claims": list(_TRUE_CLAIMS),
        "false_claims": list(_FALSE_CLAIMS),
    }
    result_id = _content_id(_RESULT_PREFIX, _RESULT_DOMAIN, result_body)
    values: dict[str, object] = {
        "disposition": (
            ThreeBaselineRegistrationDispositionV1.MECHANICS_ONLY_NOT_REGISTRATIONS
        ),
        "reason": (
            ThreeBaselineRegistrationReasonV1.CALLER_SUPPLIED_CANDIDATE_SIGNATURE_MECHANICS_VERIFIED
        ),
        "version": PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_VERSION,
        "claim_level": PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_CLAIM_LEVEL,
        "profile_id": PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID,
        "result_id": result_id,
        "candidate_registry_id": registry_id,
        "candidate_registry_root_sha256": registry_root,
        "input_snapshot_commitment_sha256": snapshot.commitment_sha256,
        "final_registration_candidate_ids": final_ids,
        "legacy_projection_candidates": projections,
        "required_signature_count": _EXPECTED_SIGNATURES,
        "verified_signature_count": _EXPECTED_SIGNATURES,
    }
    values.update({name: True for name in _TRUE_CLAIMS})
    values.update({name: False for name in _FALSE_CLAIMS})
    return ThreeBaselineRegistrationMechanicsV1(**values)


def verify_three_baseline_registration_mechanics_v1(
    *,
    r05_predecessor: R05QualificationPredecessorCandidateV1,
    candidate_keys: tuple[BaselineCandidatePublicKeyV1, ...],
    final_registrations: tuple[FinalBaselineRegistrationCandidateV1, ...],
) -> ThreeBaselineRegistrationMechanicsV1 | ThreeBaselineRegistrationRejectionV1:
    """Fresh-verify the closed three-candidate graph atomically in memory."""

    if (
        type(r05_predecessor) is not R05QualificationPredecessorCandidateV1
        or type(candidate_keys) is not tuple
        or type(final_registrations) is not tuple
    ):
        return _rejection(ThreeBaselineRegistrationReasonV1.WRONG_INPUT_TYPE)
    try:
        _guard_graph_top_level_shape(
            r05_predecessor,
            candidate_keys,
            final_registrations,
        )
    except MemoryError:
        return _rejection(
            ThreeBaselineRegistrationReasonV1.RESOURCE_BUDGET_EXCEEDED
        )
    except (AttributeError, TypeError, ValueError):
        return _rejection(
            ThreeBaselineRegistrationReasonV1.STRUCTURAL_CLOSURE_INVALID
        )

    try:
        snapshot = _freeze_graph_inputs(
            r05_predecessor,
            candidate_keys,
            final_registrations,
        )
    except (_SnapshotBudgetExceeded, MemoryError):
        return _rejection(
            ThreeBaselineRegistrationReasonV1.RESOURCE_BUDGET_EXCEEDED
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return _rejection(
            ThreeBaselineRegistrationReasonV1.STRUCTURAL_CLOSURE_INVALID
        )
    except Exception:
        return _rejection(ThreeBaselineRegistrationReasonV1.INTERNAL_ERROR)

    try:
        detached_r05, detached_keys, detached_finals = _thaw_graph_inputs(snapshot)
        detached_r05.__post_init__()
        key_by_purpose = _verify_key_registry(detached_keys)
        if len(detached_finals) != len(_BASELINE_ORDER) or any(
            type(item) is not FinalBaselineRegistrationCandidateV1
            for item in detached_finals
        ):
            raise TypeError("final registration candidate tuple shape drift")
        for item in detached_finals:
            _validate_final_registration(item)
        if tuple(item.record.kind for item in detached_finals) != _BASELINE_ORDER:
            raise ValueError("final registration baseline order drift")

        previous_final_id: str | None = None
        previous_time_token_id: str | None = None
        previous_declared_time: tuple[int, int] | None = None
        for sequence, item in enumerate(detached_finals):
            record = item.record
            time_token = item.time_token
            if record.r05_predecessor_candidate_id != detached_r05.candidate_id:
                raise ValueError("registration does not bind the supplied R05 candidate")
            if record.previous_final_registration_id != previous_final_id:
                raise ValueError("cross-baseline final-registration predecessor chain drift")
            if time_token.logical_sequence != sequence:
                raise ValueError("declared-time baseline sequence drift")
            if time_token.previous_time_token_id != previous_time_token_id:
                raise ValueError("declared-time predecessor token chain drift")
            declared_time = (
                time_token.declared_unix_seconds,
                time_token.declared_nanosecond,
            )
            if previous_declared_time is not None and declared_time <= previous_declared_time:
                raise ValueError("declared Unix times must be strictly increasing")
            previous_final_id = item.final_registration_id
            previous_time_token_id = time_token.token_id
            previous_declared_time = declared_time

        if len({item.final_registration_id for item in detached_finals}) != 3:
            raise ValueError("final registration candidate replay")
        if len({item.store_receipt.receipt_id for item in detached_finals}) != 3:
            raise ValueError("store receipt replay")
        if len({item.store_receipt.object_key for item in detached_finals}) != 3:
            raise ValueError("store object-key replay")
        if len({item.time_token.token_id for item in detached_finals}) != 3:
            raise ValueError("declared-time token replay")

        # Signature verification is deliberately fresh and follows the exact
        # baseline then purpose order only after the entire structural graph closes.
        verified_count = 0

        def after_each_verify() -> None:
            nonlocal verified_count
            _assert_graph_inputs_unchanged(
                snapshot,
                r05_predecessor=r05_predecessor,
                candidate_keys=candidate_keys,
                final_registrations=final_registrations,
            )
            verified_count += 1

        for item in detached_finals:
            _fresh_verify_five_signatures(
                item,
                key_by_purpose,
                after_each_verify,
            )
        if verified_count != _EXPECTED_SIGNATURES:
            raise ValueError("fresh signature verification count drift")
        _assert_graph_inputs_unchanged(
            snapshot,
            r05_predecessor=r05_predecessor,
            candidate_keys=candidate_keys,
            final_registrations=final_registrations,
        )
        return _success(snapshot)
    except _InputMutationDetected:
        return _rejection(
            ThreeBaselineRegistrationReasonV1.INPUT_MUTATED_DURING_VERIFICATION
        )
    except _SignatureBackendUnavailable:
        return _rejection(
            ThreeBaselineRegistrationReasonV1.SIGNATURE_BACKEND_UNAVAILABLE
        )
    except _SignatureInvalid:
        return _rejection(ThreeBaselineRegistrationReasonV1.SIGNATURE_INVALID)
    except (_SnapshotBudgetExceeded, MemoryError):
        return _rejection(
            ThreeBaselineRegistrationReasonV1.RESOURCE_BUDGET_EXCEEDED
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return _rejection(
            ThreeBaselineRegistrationReasonV1.STRUCTURAL_CLOSURE_INVALID
        )
    except Exception:
        return _rejection(ThreeBaselineRegistrationReasonV1.INTERNAL_ERROR)


def verify_three_baseline_registration_mechanics_result_v1(
    *,
    r05_predecessor: R05QualificationPredecessorCandidateV1,
    candidate_keys: tuple[BaselineCandidatePublicKeyV1, ...],
    final_registrations: tuple[FinalBaselineRegistrationCandidateV1, ...],
    result: object,
) -> bool:
    """Fresh-reverify the full graph, then exact-compare every result field."""

    fresh = verify_three_baseline_registration_mechanics_v1(
        r05_predecessor=r05_predecessor,
        candidate_keys=candidate_keys,
        final_registrations=final_registrations,
    )
    if type(fresh) is not ThreeBaselineRegistrationMechanicsV1:
        return False
    try:
        result_shell_matches = _result_top_level_equals(result, fresh)
    except MemoryError:
        return False
    if not result_shell_matches:
        return False
    try:
        supplied_first = _freeze_result_for_comparison(result)
        supplied_second = _freeze_result_for_comparison(result)
        expected = _freeze_result_for_comparison(fresh)
    except (
        AttributeError,
        KeyError,
        MemoryError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False
    return supplied_first == supplied_second == expected


__all__ = [
    "PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_SCHEMA",
    "PHASE2B_C1_THREE_BASELINE_REGISTRATION_PROFILE_V1_ID",
    "PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_VERSION",
    "PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_CLAIM_LEVEL",
    "PHASE2B_C1_THREE_BASELINE_REGISTRATION_MECHANICS_V1_ED25519_BACKEND_AVAILABLE",
    "BaselineRegistrationKindV1",
    "BaselineRevisionPolicyV1",
    "BaselineArtifactRoleV1",
    "BaselineSignaturePurposeV1",
    "ThreeBaselineRegistrationDispositionV1",
    "ThreeBaselineRegistrationReasonV1",
    "BaselineArtifactFileV1",
    "BaselineArtifactFileManifestV1",
    "BaselineCallerSuppliedCandidateBindingsV1",
    "R05QualificationPredecessorCandidateV1",
    "BaselineCandidatePublicKeyV1",
    "BaselineRuntimeBuilderAttestationCandidateV1",
    "BaselineOfflineQualifierAttestationCandidateV1",
    "RichBaselineRegistrationRecordV1",
    "BaselineRegistrationCustodianAttestationCandidateV1",
    "BaselineStoreIfAbsentReceiptCandidateV1",
    "BaselineDeclaredTimeTokenCandidateV1",
    "FinalBaselineRegistrationCandidateV1",
    "LegacyBaselineProjectionCandidateV1",
    "ThreeBaselineRegistrationMechanicsV1",
    "ThreeBaselineRegistrationRejectionV1",
    "phase2b_c1_three_baseline_registration_profile_v1",
    "artifact_file_manifest_root_v1",
    "baseline_implementation_id_v1",
    "baseline_runtime_builder_signature_preimage_v1",
    "baseline_offline_qualifier_signature_preimage_v1",
    "rich_baseline_registration_record_bytes_v1",
    "baseline_registration_custodian_signature_preimage_v1",
    "baseline_store_if_absent_signature_preimage_v1",
    "baseline_declared_time_signature_preimage_v1",
    "project_rich_candidate_to_ineligible_legacy_projection_v1",
    "verify_three_baseline_registration_mechanics_v1",
    "verify_three_baseline_registration_mechanics_result_v1",
]
