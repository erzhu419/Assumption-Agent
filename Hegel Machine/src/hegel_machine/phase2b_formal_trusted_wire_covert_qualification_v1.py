"""Synthetic qualification scaffold for the Phase-2B trusted-wire stack.

The module is deliberately narrower than a formal corpus auditor.  It loads a
single closed profile and exercises a deterministic two-authority development
known-answer vector through the existing public V2 builder, public typed
replay, namespace-aware parser, and frozen covert-audit mechanics.  The
covert-audit vector must abstain for insufficient data; that abstention is the
expected fail-closed result, not a metadata-predictivity pass.

No entry point reads files, generates keys, invokes Docker, uses a network,
allocates campaign identities, or accepts an actual corpus.  Source SHA-256
values are manifest bindings only.  Tests may independently rehash files, but
this module cannot claim that live source files were rehashed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import sys
from types import MappingProxyType
from typing import Final, Mapping, NoReturn

from .phase2b_covert_audit_v1 import (
    CovertAuditDisposition,
    CovertAuditMechanicsReceipt,
    CovertAuditStructuralReceipt,
    DEFAULT_COVERT_AUDIT_POLICY,
    EnvelopeAuditRow,
    InvarianceAuditBatch,
    AuditorLabels,
    AuditorPermutationStrata,
    SEMANTICS_ID,
    run_frozen_envelope_covert_audit_mechanics,
    validate_frozen_covert_audit_structure,
)
from .phase2b_exact_transform_semantics_v1 import (
    PublicTransformEvidenceBundleV2,
)
from .phase2b_trusted_wire_batch_v2 import (
    TRUSTED_WIRE_BATCH_V2_POLICY_ID,
    TrustedWireBatchDispositionV2,
    TrustedWireBatchRejectionV2,
    TrustedWireBatchV2,
    TrustedWireKeySourcesV2,
    build_trusted_wire_batch_v2,
)
from .phase2b_trusted_wire_typed_replay_v2 import (
    TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
    TypedTrustedWireBatchReplayRejectionV2,
    TypedTrustedWireBatchReplayV2,
    replay_typed_trusted_wire_batch_v2,
)
from .phase2b_trusted_wire_typed_authority_v2 import (
    COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
)
from .phase2b_trusted_wire_v1 import (
    ENVELOPE_BYTES,
    FIELD_MANIFEST_ID,
    JCS_PROFILE_ID,
)


PROFILE_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-trusted-wire-covert-qualification/1"
)
PROFILE_STATUS: Final = "SYNTHETIC_IMPLEMENTATION_QUALIFICATION_SCAFFOLD_FROZEN"
PROFILE_CLAIM_LEVEL: Final = (
    "SYNTHETIC_DEV_KAT_ONLY_NOT_FORMAL_WIRE_NOT_FORMAL_CORPUS_AUDIT_NOT_EXECUTION"
)
PROFILE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE_COVERT_QUALIFICATION/PROFILE/V1\x00"
)
PROFILE_ID_PREFIX: Final = (
    "phase2b_formal_trusted_wire_covert_qualification_profile_v1_"
)
PROFILE_ID: Final = (
    PROFILE_ID_PREFIX
    + "de3f7edfacc8115c565b30a60bd285500cb0027cf4e7640749c8c1b954ae5c63"
)

QUALIFICATION_RECEIPT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE_COVERT_QUALIFICATION/RECEIPT/V1\x00"
)
QUALIFICATION_RECEIPT_PREFIX: Final = (
    "phase2b_formal_trusted_wire_covert_qualification_receipt_v1_"
)

FAIL_INPUT: Final = "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V1_INPUT"
FAIL_JSON: Final = "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V1_JSON"
FAIL_CANONICAL: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V1_CANONICAL"
)
FAIL_PROFILE: Final = "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V1_PROFILE"
FAIL_CONTENT_ID: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V1_CONTENT_ID"
)
FAIL_KAT_INPUT: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V1_KAT_INPUT"
)
FAIL_PUBLIC_MECHANICS: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V1_PUBLIC_MECHANICS"
)
FAIL_NAMESPACE: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V1_NAMESPACE"
)
FAIL_COVERT_STRUCTURE: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V1_COVERT_STRUCTURE"
)
FAIL_CLAIM_BOUNDARY: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V1_CLAIM_BOUNDARY"
)

_MAXIMUM_PROFILE_BYTES: Final = 128 * 1024
_ROOT_FIELDS: Final = frozenset({"qualification_profile", "qualification_profile_id"})

_SOURCE_IDENTITY_MANIFEST: Final = [
    {
        "path": "src/hegel_machine/hashing.py",
        "sha256": "6fcd988ca4e8b75b9aa9e5a8b38c9ddd73e2685140e2cd0095d68f8bda054159",
    },
    {
        "path": "src/hegel_machine/schema.py",
        "sha256": "6497a93f8a68f3313d294bc99dc3b8f43508e93772d748521d09ea716b8ae10a",
    },
    {
        "path": "src/hegel_machine/phase3_dsl_v1.py",
        "sha256": "c0e3ef208d08c7415dd934f8066f6c845a943272e1b9033db3b2a4083694ae27",
    },
    {
        "path": "src/hegel_machine/phase2b_freeze_v1.py",
        "sha256": "39a289ef0c000c1c62eb3a7a5f97ddfae3e51eb7904757642b0c2d167067f543",
    },
    {
        "path": "src/hegel_machine/phase2b_wire.py",
        "sha256": "0330d205b6bdfd9452063e636873e99cb8b8d902decd4124e52207061294d9db",
    },
    {
        "path": "src/hegel_machine/phase2b_uncertainty_compiler.py",
        "sha256": "529c3d8e7dc2baec20c76ec5f9f21c30c31c1842305872499ba95e6dbb584d35",
    },
    {
        "path": "src/hegel_machine/phase2b_exact_transform_semantics_v1.py",
        "sha256": "353b10a7464da54553cf84e738bd53c010274ad1b15e9c36729420ea11f8e55c",
    },
    {
        "path": "src/hegel_machine/phase2b_trusted_wire_v1.py",
        "sha256": "98defeb8455944144e318879688174488e1c13aab2153c019a6db4b67946b5c2",
    },
    {
        "path": "src/hegel_machine/phase2b_trusted_wire_typed_authority_v1.py",
        "sha256": "d6e39f81257026281fd81d191fcc6e37be8ff130a6bde6ac0d8e417cb6c11f63",
    },
    {
        "path": "src/hegel_machine/phase2b_trusted_wire_batch_v1.py",
        "sha256": "0b2606bd7ba4e41253beaadb9fee5e5d3a301382dac5a6025f51756e5cd1eeaa",
    },
    {
        "path": "src/hegel_machine/phase2b_trusted_wire_batch_v2.py",
        "sha256": "42e4fba7ff3829f57eb60b4f8d1ffe87099deed8b751cc2b3101bcfa02f703b5",
    },
    {
        "path": "src/hegel_machine/phase2b_trusted_wire_typed_authority_v2.py",
        "sha256": "6d6cc72547a8f00402e126b4a339272e6de0353525e7627c082ef62faa407521",
    },
    {
        "path": "src/hegel_machine/phase2b_trusted_wire_typed_replay_v2.py",
        "sha256": "0608113116bf0f548acd17d509a7ca0e13889d518260e1b9f4a3aaf8ab942309",
    },
    {
        "path": "src/hegel_machine/phase2b_covert_audit_v1.py",
        "sha256": "96585dbf4c9e2eaf019b228b6c3ec13e5bf951b0d761a6277f72fe628045a3a8",
    },
]

_PUBLIC_MECHANICS_BINDINGS: Final = {
    "accepted_jcs_profile_id": JCS_PROFILE_ID,
    "compact_typed_authority_codec_policy_id": (
        COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID
    ),
    "covert_audit_policy_id": DEFAULT_COVERT_AUDIT_POLICY.policy_id,
    "covert_audit_semantics_id": SEMANTICS_ID,
    "envelope_bytes": ENVELOPE_BYTES,
    "namespace_field_manifest_id": FIELD_MANIFEST_ID,
    "trusted_wire_batch_v2_policy_id": TRUSTED_WIRE_BATCH_V2_POLICY_ID,
    "typed_trusted_wire_replay_v2_policy_id": (
        TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID
    ),
    "loaded_public_mechanics_basis_commit": "0cdf729e80f828cf2fc1cf2fd1ca67319103d829",
    "qualification_runtime": {
        "implementation": "cpython",
        "major": 3,
        "minor": 10,
        "micro": 12,
    },
}

_KAT_INPUTS: Final = {
    "fixture_class": "DECLARED_SYNTHETIC_DEV_KAT_NOT_ORIGIN_AUTHENTICATED",
    "source_authority_content_ids_in_order": [
        "phase2b_public_transform_evidence_e61977454b06c77added4d3059a99eea7b59a29d878c116af3f7c0deccf14073",
        "phase2b_public_transform_evidence_7d7cbea8942625e44bc949189c3bbb03b3529a2cef8919a06e40a4336a9d5b6b",
    ],
    "run_id_hex": "50e9ed3d8f4126d17566865ee40e9ede560c82ecdd87f5017327bc8f3172c494",
    "shuffle_ikm_hex": "a6a0e6f539332c38d883124893a457bcd3e91be544ad6f4e211f8539e17bcaea",
    "id_ikm_hex": "e35b1f110b64e900804e47da1688d6fd0782975cb622e36496ffa9fdaa4bb439",
    "padding_ikm_hex": "d196ef9335af26147ffcb5700976f6ce2194f14b6362c8b70e548d316750a30b",
}

_KAT_EXPECTED_OUTPUTS: Final = {
    "batch_id": "phase2b_trusted_wire_batch_v2_fb3b5932c4993443b1f884861611af88d66b13fc940ee0da9006f334161a84eb",
    "replay_receipt_id": "phase2b_typed_trusted_wire_batch_replay_v2_61675b28e9a734b0d5b51f412720f693dbd154543517e2cf34815923ae9a2e9a",
    "envelope_ids_in_order": [
        "phase2b_trusted_envelope_v2_658e656b0804dd30a6392911cd7cc5b8df0eaeb40ade88f10b18936f803c018e",
        "phase2b_trusted_envelope_v2_a6978c6bd6f5e3c8584ce440ec63e1ca55e76411fb0a3cfa273a5ca17c23a41e",
    ],
    "public_authority_content_ids_in_order": [
        "phase2b_public_transform_evidence_e4ea331749bdcd99dffc7d2b81a576e468400c896ca4d5fb3466c36a9952d48b",
        "phase2b_public_transform_evidence_b3841f99c12f32055aef734942b1a43ce004950f834e298dfd5472f8c19fc22b",
    ],
    "namespace_audit_ids_in_order": [
        "phase2b_namespace_audit_v2_81e43431a7fbaf487760724e066bfe1175c46426971364d3580208ca9f5182bf",
        "phase2b_namespace_audit_v2_376a357e87d4b5eba64983455cd758e0ff827e80e9e9bf478320cea314883ecf",
    ],
    "structural_receipt_id": "phase2b_covert_audit_structural_receipt_6889e089a4b336a2181f55b3bbe357a2a79beae7a904bbd27e634f9abb8bd343",
    "covert_mechanics_receipt_id": "phase2b_covert_audit_mechanics_receipt_736bbda5e052ec3ce37a20b230538af6c56e858f29a678075dbb3e9d7f3b0c79",
    "covert_disposition": "ABSTAIN",
    "covert_statistics_executed": False,
    "metadata_predictivity_gate_passed": False,
}

_QUALIFIED_SCAFFOLD_SURFACES: Final = [
    "CLOSED_CANONICAL_PROFILE_AND_CONTENT_ID",
    "SCHEMA_CLOSED_ASCII_SAFE_INTEGER_ACCEPTED_JCS_KAT",
    "FIXED_65536_BYTE_ENVELOPE_AND_CLOSED_WRAPPER_KAT",
    "THREE_DISTINCT_PURPOSE_KEY_INPUT_POLICY_KAT",
    "POST_SHUFFLE_UUIDV4_ALLOCATION_KAT",
    "NAMESPACE_AWARE_FIELD_MANIFEST_PARSE_KAT",
    "PUBLIC_TYPED_REPLAY_AND_EXACT_TRANSFORM_KAT",
    "GLOBAL_RENAMING_32_RECEIPT_COVERAGE_AND_DECISION_EQUALITY_KAT",
    "CASE_ORDER_32_AND_OBSERVATION_ORDER_16_RECEIPT_COVERAGE_KAT",
    "INSUFFICIENT_METADATA_PREDICTIVITY_INPUT_ABSTAINS_KAT",
    "ATOMIC_SUCCESS_OR_EMPTY_ROOT_REJECTION_API",
]

_REMAINING_FORMAL_GAPS: Final = [
    "FULL_RFC8785_DOMAIN_CANONICALIZER_NOT_IMPLEMENTED_ACCEPTED_JCS_SUBSET_ONLY",
    "ACTUAL_960_CORPUS_NOT_LOADED_OR_AUDITED",
    "1024_AUTHORITY_CAPACITY_BOUNDARY_NOT_QUALIFIED",
    "COVERT_STATISTICS_IMPLEMENTATION_CAPPED_AT_64_ROWS",
    "PAIRWISE_DISTINCT_KAT_IKM_DOES_NOT_PROVE_KEY_SOURCE_INDEPENDENCE",
    "NAMESPACE_CLOSED_LEAF_CLASSIFICATION_AND_TEN_NAMESPACE_FEATURE_COVERAGE_NOT_IMPLEMENTED",
    "ALL_UUID_FIELD_LEVEL_PREDICTIVITY_FEATURES_NOT_IMPLEMENTED",
    "MISSINGNESS_WIRE_REPRESENTATION_DOES_NOT_MATCH_EXACT_FREEZE",
    "MISSINGNESS_FIELD_LEVEL_PREDICTIVITY_FEATURES_NOT_IMPLEMENTED",
    "PER_CASE_TRANSFORM_CATALOG_REPRESENTATION_CONFLICTS_WITH_EXACT_FREEZE",
    "FILENAME_AND_ARCHIVE_ORDER_PREDICTIVITY_FEATURES_NOT_IMPLEMENTED",
    "PROVENANCE_AND_ELIGIBILITY_MASK_PREDICTIVITY_FEATURES_NOT_IMPLEMENTED",
    "PAYLOAD_LENGTH_HEADER_CHANNEL_NOT_AUDITED",
    "PADDING_AND_UNUSED_TRANSFORM_SLOT_CHANNEL_FEATURES_NOT_IMPLEMENTED",
    "DERIVED_PRIVATE_ANSWER_CHANNEL_FEATURES_NOT_IMPLEMENTED",
    "GLOBAL_RENAMING_32_CASE_ORDER_32_OBSERVATION_ORDER_16_ARE_CALLER_SUPPLIED_RECEIPTS_NOT_ACTUAL_RECOGNIZER_RERUNS",
    "GLOBAL_CONSISTENT_RENAMING_GENERATOR_AND_RECOGNIZER_EXECUTION_NOT_IMPLEMENTED",
    "ID_MESSAGE_FRAMING_DOCUMENTATION_AND_IMPLEMENTATION_BINDING_NOT_CLOSED",
    "STANDARD_ERROR_UNSUPPORTED_WHOLE_BATCH_ATOMIC_PATH_NOT_QUALIFIED",
    "PYTHON_BINARY64_STATISTICS_HAVE_NO_CROSS_RUNTIME_FORMAL_REPLAY",
    "FORMAL_CORPUS_AUDITOR_AUTHORITY_ORIGIN_AND_INDEPENDENCE_NOT_IMPLEMENTED",
]

_PERMANENT_NONCLAIMS: Final = {
    "source_files_rehashed": False,
    "git_commit_live_verified": False,
    "full_rfc8785_domain_qualified": False,
    "formal_wire_builder_implemented": False,
    "formal_corpus_auditor_implemented": False,
    "metadata_predictivity_statistics_executed": False,
    "metadata_predictivity_gate_passed": False,
    "key_source_independence_verified": False,
    "namespace_closed_leaf_classification_qualified": False,
    "ten_namespace_feature_coverage_qualified": False,
    "all_uuid_field_predictivity_qualified": False,
    "missingness_freeze_representation_aligned": False,
    "per_case_transform_catalog_freeze_aligned": False,
    "payload_length_channel_audited": False,
    "standard_error_unsupported_atomicity_qualified": False,
    "binary64_cross_runtime_formal_replay_qualified": False,
    "global_consistent_renaming_execution_verified": False,
    "1024_authority_capacity_qualified": False,
    "recognizer_capacity_evidence": False,
    "actual_corpus_loaded": False,
    "actual_corpus_audited": False,
    "origin_authenticated": False,
    "formal_uuid_audit": False,
    "formal_covert_audit": False,
    "formal_auditor_authority_verified": False,
    "sealed_holdout_eligible": False,
    "actual_execution_claimed": False,
    "execution_authorized": False,
    "formal_holdout_generation_authorized": False,
    "exact_freeze_blocker_closed": False,
    "c1_exit_evidence": False,
    "docker_invoked": False,
    "filesystem_accessed": False,
    "network_accessed": False,
    "keys_generated": False,
}

_EXPECTED_PROFILE: Final = {
    "schema_version": PROFILE_SCHEMA_VERSION,
    "status": PROFILE_STATUS,
    "claim_level": PROFILE_CLAIM_LEVEL,
    "canonicalization": "SORTED_KEYS_ASCII_JSON_NO_WHITESPACE_SINGLE_LF_TERMINATOR_ALLOW_NAN_FALSE",
    "rfc8785_scope": {
        "profile": "RFC8785_DERIVED_SCHEMA_CLOSED_ASCII_SAFE_INTEGER_ACCEPTED_JCS_SUBSET",
        "unicode_and_ieee754_full_domain_supported": False,
        "duplicate_keys_rejected": True,
        "floats_rejected": True,
        "non_ascii_rejected": True,
        "safe_integers_only": True,
    },
    "source_identity_manifest_semantics": (
        "STATIC_EXPECTED_CRITICAL_TRANSITIVE_SHA256_BINDINGS_NOT_LIVE_SOURCE_REHASH_EVIDENCE"
    ),
    "permanent_nonclaims_scope": (
        "FILESYSTEM_DOCKER_NETWORK_AND_KEY_GENERATION_FALSE_APPLY_TO_QUALIFICATION_ENTRYPOINT_NOT_TEST_HARNESS_OR_CALLER_FIXTURE_CONSTRUCTION"
    ),
    "source_identity_manifest_in_order": _SOURCE_IDENTITY_MANIFEST,
    "public_mechanics_bindings": _PUBLIC_MECHANICS_BINDINGS,
    "known_answer_inputs": _KAT_INPUTS,
    "known_answer_expected_outputs": _KAT_EXPECTED_OUTPUTS,
    "qualified_scaffold_surfaces_in_order": _QUALIFIED_SCAFFOLD_SURFACES,
    "remaining_formal_gaps_in_order": _REMAINING_FORMAL_GAPS,
    "permanent_nonclaims": _PERMANENT_NONCLAIMS,
}


class Phase2BFormalTrustedWireCovertQualificationErrorV1(ValueError):
    """Stable fail-closed error for malformed profile material."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BFormalTrustedWireCovertQualificationErrorV1(code, detail)


def _exact_json_equal(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(
            _exact_json_equal(value[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _exact_json_equal(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return value == expected


def _detached_json_copy(value: object) -> object:
    if type(value) is dict:
        return {key: _detached_json_copy(item) for key, item in value.items()}
    if type(value) is list:
        return [_detached_json_copy(item) for item in value]
    return value


def _deep_freeze(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_deep_thaw(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        _fail(FAIL_CANONICAL, f"value is outside canonical JSON: {exc}")


def _reject_float(_: str) -> NoReturn:
    _fail(FAIL_JSON, "floating-point values are forbidden")


def _reject_constant(_: str) -> NoReturn:
    _fail(FAIL_JSON, "non-finite JSON constants are forbidden")


def _pairs_to_exact_dict(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(FAIL_JSON, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def qualification_profile_id_v1(profile: object) -> str:
    if not _exact_json_equal(profile, _EXPECTED_PROFILE):
        _fail(FAIL_PROFILE, "qualification profile differs from the frozen profile")
    canonical = _canonical_json_bytes(profile)
    return PROFILE_ID_PREFIX + hashlib.sha256(PROFILE_DOMAIN + canonical).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class Phase2BFormalTrustedWireCovertProfileV1:
    profile_id: str
    profile: Mapping[str, object] = field(repr=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("qualification profiles are privately issued")

    @classmethod
    def _issue(cls, profile: dict[str, object]) -> "Phase2BFormalTrustedWireCovertProfileV1":
        value = object.__new__(cls)
        object.__setattr__(value, "profile_id", qualification_profile_id_v1(profile))
        object.__setattr__(value, "profile", _deep_freeze(_detached_json_copy(profile)))
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not Phase2BFormalTrustedWireCovertProfileV1:
            raise TypeError("qualification profile must use the exact type")
        thawed = _deep_thaw(self.profile)
        expected_id = qualification_profile_id_v1(thawed)
        if self.profile_id != expected_id or self.profile_id != PROFILE_ID:
            raise ValueError("qualification profile content ID drift")


def load_phase2b_formal_trusted_wire_covert_qualification_profile_v1(
    payload: bytes,
) -> Phase2BFormalTrustedWireCovertProfileV1:
    """Load the one canonical profile from exact bytes; never access files."""

    if type(payload) is not bytes:
        _fail(FAIL_INPUT, "profile payload must use exact bytes")
    if not payload or len(payload) > _MAXIMUM_PROFILE_BYTES:
        _fail(FAIL_INPUT, "profile payload length is outside the frozen budget")
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        _fail(FAIL_JSON, f"profile payload must be ASCII: {exc}")
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_pairs_to_exact_dict,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except Phase2BFormalTrustedWireCovertQualificationErrorV1:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        _fail(FAIL_JSON, f"profile payload is invalid JSON: {exc}")
    if type(decoded) is not dict or set(decoded) != _ROOT_FIELDS:
        _fail(FAIL_PROFILE, "profile root is not the exact closed two-field object")
    profile = decoded["qualification_profile"]
    profile_id = decoded["qualification_profile_id"]
    if not _exact_json_equal(profile, _EXPECTED_PROFILE):
        _fail(FAIL_PROFILE, "qualification profile content drift")
    expected_id = qualification_profile_id_v1(profile)
    if type(profile_id) is not str or profile_id != expected_id or profile_id != PROFILE_ID:
        _fail(FAIL_CONTENT_ID, "qualification profile ID mismatch")
    if _canonical_json_bytes(decoded) + b"\n" != payload:
        _fail(FAIL_CANONICAL, "qualification profile bytes are not canonical")
    return Phase2BFormalTrustedWireCovertProfileV1._issue(profile)


class QualificationDispositionV1(str, Enum):
    SYNTHETIC_SCAFFOLD_QUALIFIED = "SYNTHETIC_SCAFFOLD_QUALIFIED"
    REJECTED = "REJECTED"


_QUALIFIED_TRUE_CLAIMS: Final = (
    "implementation_profile_bound",
    "loaded_public_mechanics_exercised",
    "accepted_jcs_subset_roundtrip_kat_qualified",
    "fixed_envelope_and_closed_wrapper_kat_qualified",
    "purpose_separated_key_input_policy_kat_qualified",
    "post_shuffle_uuidv4_allocation_kat_qualified",
    "namespace_aware_field_manifest_parse_kat_qualified",
    "public_typed_replay_kat_qualified",
    "global_renaming_receipt_coverage_and_decision_equality_kat_qualified",
    "case_and_observation_order_receipt_coverage_kat_qualified",
    "insufficient_metadata_predictivity_input_abstains_kat_qualified",
    "atomic_success_or_empty_rejection_api_qualified",
)
_QUALIFIED_FALSE_CLAIMS: Final = tuple(_PERMANENT_NONCLAIMS)


@dataclass(frozen=True, slots=True)
class Phase2BFormalTrustedWireCovertQualificationRejectionV1:
    disposition: QualificationDispositionV1
    reason: str
    profile_id: str | None
    authority_count: int
    batch_id: None = None
    replay_receipt_id: None = None
    envelope_ids: tuple[()] = ()
    namespace_audit_ids: tuple[()] = ()
    structural_receipt_id: None = None
    covert_mechanics_receipt_id: None = None
    qualification_receipt_id: None = None

    def __post_init__(self) -> None:
        if type(self) is not Phase2BFormalTrustedWireCovertQualificationRejectionV1:
            raise TypeError("qualification rejection must use the exact type")
        if self.disposition is not QualificationDispositionV1.REJECTED:
            raise ValueError("qualification rejection disposition drift")
        if type(self.reason) is not str or not self.reason or not self.reason.isascii():
            raise ValueError("qualification rejection reason must be nonempty ASCII")
        if type(self.authority_count) is not int or self.authority_count < 0:
            raise ValueError("qualification rejection authority count is invalid")
        if self.profile_id is not None and self.profile_id != PROFILE_ID:
            raise ValueError("qualification rejection profile ID drift")
        if any(
            value is not None
            for value in (
                self.batch_id,
                self.replay_receipt_id,
                self.structural_receipt_id,
                self.covert_mechanics_receipt_id,
                self.qualification_receipt_id,
            )
        ) or self.envelope_ids != () or self.namespace_audit_ids != ():
            raise ValueError("qualification rejection exposes partial roots")


_RECEIPT_ISSUE_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class Phase2BFormalTrustedWireCovertQualificationReceiptV1:
    disposition: QualificationDispositionV1
    profile: Phase2BFormalTrustedWireCovertProfileV1 = field(repr=False)
    batch: TrustedWireBatchV2 = field(repr=False)
    replay: TypedTrustedWireBatchReplayV2 = field(repr=False)
    structural_receipt: CovertAuditStructuralReceipt = field(repr=False)
    covert_mechanics_receipt: CovertAuditMechanicsReceipt = field(repr=False)
    profile_id: str
    source_authority_content_ids: tuple[str, ...]
    batch_id: str
    replay_receipt_id: str
    envelope_ids: tuple[str, ...]
    namespace_audit_ids: tuple[str, ...]
    structural_receipt_id: str
    covert_mechanics_receipt_id: str
    claim_level: str
    implementation_profile_bound: bool
    loaded_public_mechanics_exercised: bool
    accepted_jcs_subset_roundtrip_kat_qualified: bool
    fixed_envelope_and_closed_wrapper_kat_qualified: bool
    purpose_separated_key_input_policy_kat_qualified: bool
    post_shuffle_uuidv4_allocation_kat_qualified: bool
    namespace_aware_field_manifest_parse_kat_qualified: bool
    public_typed_replay_kat_qualified: bool
    global_renaming_receipt_coverage_and_decision_equality_kat_qualified: bool
    case_and_observation_order_receipt_coverage_kat_qualified: bool
    insufficient_metadata_predictivity_input_abstains_kat_qualified: bool
    atomic_success_or_empty_rejection_api_qualified: bool
    source_files_rehashed: bool
    git_commit_live_verified: bool
    full_rfc8785_domain_qualified: bool
    formal_wire_builder_implemented: bool
    formal_corpus_auditor_implemented: bool
    metadata_predictivity_statistics_executed: bool
    metadata_predictivity_gate_passed: bool
    key_source_independence_verified: bool
    namespace_closed_leaf_classification_qualified: bool
    ten_namespace_feature_coverage_qualified: bool
    all_uuid_field_predictivity_qualified: bool
    missingness_freeze_representation_aligned: bool
    per_case_transform_catalog_freeze_aligned: bool
    payload_length_channel_audited: bool
    standard_error_unsupported_atomicity_qualified: bool
    binary64_cross_runtime_formal_replay_qualified: bool
    global_consistent_renaming_execution_verified: bool
    capacity_1024_authority_qualified: bool
    recognizer_capacity_evidence: bool
    actual_corpus_loaded: bool
    actual_corpus_audited: bool
    origin_authenticated: bool
    formal_uuid_audit: bool
    formal_covert_audit: bool
    formal_auditor_authority_verified: bool
    sealed_holdout_eligible: bool
    actual_execution_claimed: bool
    execution_authorized: bool
    formal_holdout_generation_authorized: bool
    exact_freeze_blocker_closed: bool
    c1_exit_evidence: bool
    docker_invoked: bool
    filesystem_accessed: bool
    network_accessed: bool
    keys_generated: bool
    qualification_receipt_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("qualification receipts are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        profile: Phase2BFormalTrustedWireCovertProfileV1,
        source_authority_content_ids: tuple[str, ...],
        batch: TrustedWireBatchV2,
        replay: TypedTrustedWireBatchReplayV2,
        structural_receipt: CovertAuditStructuralReceipt,
        covert_mechanics_receipt: CovertAuditMechanicsReceipt,
    ) -> "Phase2BFormalTrustedWireCovertQualificationReceiptV1":
        if token is not _RECEIPT_ISSUE_TOKEN:
            raise TypeError("qualification receipt issuer token mismatch")
        value = object.__new__(cls)
        fixed: tuple[tuple[str, object], ...] = (
            ("disposition", QualificationDispositionV1.SYNTHETIC_SCAFFOLD_QUALIFIED),
            ("profile", profile),
            ("batch", batch),
            ("replay", replay),
            ("structural_receipt", structural_receipt),
            ("covert_mechanics_receipt", covert_mechanics_receipt),
            ("profile_id", profile.profile_id),
            ("source_authority_content_ids", source_authority_content_ids),
            ("batch_id", batch.batch_id),
            ("replay_receipt_id", replay.receipt_id),
            ("envelope_ids", batch.envelope_ids),
            (
                "namespace_audit_ids",
                tuple(row.namespace_audit_id for row in replay.rows),
            ),
            ("structural_receipt_id", structural_receipt.content_id),
            ("covert_mechanics_receipt_id", covert_mechanics_receipt.content_id),
            ("claim_level", PROFILE_CLAIM_LEVEL),
        )
        for name, item in fixed:
            object.__setattr__(value, name, item)
        for name in _QUALIFIED_TRUE_CLAIMS:
            object.__setattr__(value, name, True)
        for name in _QUALIFIED_FALSE_CLAIMS:
            receipt_name = (
                "capacity_1024_authority_qualified"
                if name == "1024_authority_capacity_qualified"
                else name
            )
            object.__setattr__(value, receipt_name, False)
        object.__setattr__(value, "qualification_receipt_id", value._recompute_id())
        value._validate()
        return value

    def _id_payload(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "source_authority_content_ids": self.source_authority_content_ids,
            "batch_id": self.batch_id,
            "replay_receipt_id": self.replay_receipt_id,
            "envelope_ids": self.envelope_ids,
            "namespace_audit_ids": self.namespace_audit_ids,
            "structural_receipt_id": self.structural_receipt_id,
            "covert_mechanics_receipt_id": self.covert_mechanics_receipt_id,
            "claim_level": self.claim_level,
            "qualified_true_claims": {
                name: getattr(self, name) for name in _QUALIFIED_TRUE_CLAIMS
            },
            "permanent_nonclaims": {
                name: getattr(
                    self,
                    "capacity_1024_authority_qualified"
                    if name == "1024_authority_capacity_qualified"
                    else name,
                )
                for name in _QUALIFIED_FALSE_CLAIMS
            },
        }

    def _recompute_id(self) -> str:
        payload = _canonical_json_bytes(self._id_payload())
        return QUALIFICATION_RECEIPT_PREFIX + hashlib.sha256(
            QUALIFICATION_RECEIPT_DOMAIN + payload
        ).hexdigest()

    def _validate(self) -> None:
        if type(self) is not Phase2BFormalTrustedWireCovertQualificationReceiptV1:
            raise TypeError("qualification receipt must use the exact type")
        self.profile._validate()
        if self.disposition is not QualificationDispositionV1.SYNTHETIC_SCAFFOLD_QUALIFIED:
            raise ValueError("qualification receipt disposition drift")
        if self.profile_id != PROFILE_ID or self.profile_id != self.profile.profile_id:
            raise ValueError("qualification receipt profile binding drift")
        replayed = replay_typed_trusted_wire_batch_v2(batch=self.batch)
        if type(replayed) is not TypedTrustedWireBatchReplayV2 or replayed != self.replay:
            raise ValueError("qualification receipt public replay drift")
        self.structural_receipt.__post_init__()
        self.covert_mechanics_receipt.__post_init__()
        expected = _KAT_EXPECTED_OUTPUTS
        if (
            self.source_authority_content_ids
            != tuple(_KAT_INPUTS["source_authority_content_ids_in_order"])
            or self.batch_id != expected["batch_id"]
            or self.replay_receipt_id != expected["replay_receipt_id"]
            or self.envelope_ids != tuple(expected["envelope_ids_in_order"])
            or self.namespace_audit_ids
            != tuple(expected["namespace_audit_ids_in_order"])
            or self.structural_receipt_id != expected["structural_receipt_id"]
            or self.covert_mechanics_receipt_id
            != expected["covert_mechanics_receipt_id"]
            or self.batch_id != self.batch.batch_id
            or self.replay_receipt_id != self.replay.receipt_id
            or self.structural_receipt_id != self.structural_receipt.content_id
            or self.covert_mechanics_receipt_id
            != self.covert_mechanics_receipt.content_id
            or self.covert_mechanics_receipt.structural_receipt_content_id
            != self.structural_receipt_id
        ):
            raise ValueError("qualification receipt known-answer roots drift")
        if self.claim_level != PROFILE_CLAIM_LEVEL:
            raise ValueError("qualification receipt claim level drift")
        all_claim_names = (*_QUALIFIED_TRUE_CLAIMS, *(
            "capacity_1024_authority_qualified"
            if name == "1024_authority_capacity_qualified"
            else name
            for name in _QUALIFIED_FALSE_CLAIMS
        ))
        if any(type(getattr(self, name)) is not bool for name in all_claim_names):
            raise TypeError("qualification receipt claims must use exact booleans")
        if not all(getattr(self, name) for name in _QUALIFIED_TRUE_CLAIMS):
            raise ValueError("qualification receipt true-claim boundary drift")
        if any(getattr(self, name) for name in all_claim_names[len(_QUALIFIED_TRUE_CLAIMS):]):
            raise ValueError("qualification receipt permanent nonclaim drift")
        if (
            self.covert_mechanics_receipt.disposition is not CovertAuditDisposition.ABSTAIN
            or self.covert_mechanics_receipt.permutations_executed_per_target != 0
            or self.covert_mechanics_receipt.results != ()
            or self.covert_mechanics_receipt.envelope_feature_gate_acceptable
        ):
            raise ValueError("qualification KAT must fail closed at metadata sufficiency")
        if self.qualification_receipt_id != self._recompute_id():
            raise ValueError("qualification receipt content ID drift")


def _rejection(
    reason: str,
    profile_id: str | None,
    authority_count: int,
) -> Phase2BFormalTrustedWireCovertQualificationRejectionV1:
    return Phase2BFormalTrustedWireCovertQualificationRejectionV1(
        QualificationDispositionV1.REJECTED,
        reason,
        profile_id,
        authority_count,
    )


def qualify_phase2b_formal_trusted_wire_covert_scaffold_v1(
    *,
    profile: Phase2BFormalTrustedWireCovertProfileV1,
    authorities: tuple[PublicTransformEvidenceBundleV2, ...],
    run_id: bytes,
    key_sources: TrustedWireKeySourcesV2,
    auditor_labels: tuple[AuditorLabels, ...],
    auditor_strata: tuple[AuditorPermutationStrata, ...],
    invariance: InvarianceAuditBatch,
) -> (
    Phase2BFormalTrustedWireCovertQualificationReceiptV1
    | Phase2BFormalTrustedWireCovertQualificationRejectionV1
):
    """Exercise the exact synthetic KAT atomically through public mechanics."""

    profile_id: str | None = None
    authority_count = len(authorities) if type(authorities) is tuple else 0
    try:
        if type(profile) is not Phase2BFormalTrustedWireCovertProfileV1:
            raise TypeError("qualification requires the exact loaded profile")
        profile._validate()
        profile_id = profile.profile_id
        runtime = _PUBLIC_MECHANICS_BINDINGS["qualification_runtime"]
        if (
            type(authorities) is not tuple
            or len(authorities) != 2
            or any(type(item) is not PublicTransformEvidenceBundleV2 for item in authorities)
            or tuple(item.content_id for item in authorities)
            != tuple(_KAT_INPUTS["source_authority_content_ids_in_order"])
            or type(run_id) is not bytes
            or run_id.hex() != _KAT_INPUTS["run_id_hex"]
            or type(key_sources) is not TrustedWireKeySourcesV2
            or key_sources.shuffle_ikm.hex() != _KAT_INPUTS["shuffle_ikm_hex"]
            or key_sources.id_ikm.hex() != _KAT_INPUTS["id_ikm_hex"]
            or key_sources.padding_ikm.hex() != _KAT_INPUTS["padding_ikm_hex"]
            or sys.implementation.name != runtime["implementation"]
            or sys.version_info[:3]
            != (runtime["major"], runtime["minor"], runtime["micro"])
        ):
            return _rejection(FAIL_KAT_INPUT, profile_id, authority_count)
        key_sources.__post_init__()
        if (
            type(auditor_labels) is not tuple
            or type(auditor_strata) is not tuple
            or len(auditor_labels) != 2
            or len(auditor_strata) != 2
            or any(type(item) is not AuditorLabels for item in auditor_labels)
            or any(type(item) is not AuditorPermutationStrata for item in auditor_strata)
            or type(invariance) is not InvarianceAuditBatch
        ):
            return _rejection(FAIL_KAT_INPUT, profile_id, authority_count)

        built = build_trusted_wire_batch_v2(
            authorities=authorities,
            run_id=run_id,
            key_sources=key_sources,
        )
        if type(built) is TrustedWireBatchRejectionV2:
            return _rejection(FAIL_PUBLIC_MECHANICS, profile_id, authority_count)
        if (
            type(built) is not TrustedWireBatchV2
            or built.disposition is not TrustedWireBatchDispositionV2.COMPLETE
            or built.batch_id != _KAT_EXPECTED_OUTPUTS["batch_id"]
        ):
            return _rejection(FAIL_PUBLIC_MECHANICS, profile_id, authority_count)
        replay = replay_typed_trusted_wire_batch_v2(batch=built)
        if type(replay) is TypedTrustedWireBatchReplayRejectionV2:
            return _rejection(FAIL_PUBLIC_MECHANICS, profile_id, authority_count)
        if (
            type(replay) is not TypedTrustedWireBatchReplayV2
            or replay.receipt_id != _KAT_EXPECTED_OUTPUTS["replay_receipt_id"]
        ):
            return _rejection(FAIL_PUBLIC_MECHANICS, profile_id, authority_count)

        for row in replay.rows:
            if (
                len(row.envelope) != ENVELOPE_BYTES
                or row.namespace_audit.manifest_id != FIELD_MANIFEST_ID
                or row.namespace_audit.formal_uuid_namespace_field_audit
                or row.formal_uuid_audit
                or row.formal_covert_audit
            ):
                return _rejection(FAIL_NAMESPACE, profile_id, authority_count)
        audit_rows = tuple(
            EnvelopeAuditRow(index, replay.rows[index].envelope, labels, auditor_strata[index])
            for index, labels in enumerate(auditor_labels)
        )
        structural = validate_frozen_covert_audit_structure(audit_rows, invariance)
        mechanics = run_frozen_envelope_covert_audit_mechanics(audit_rows, invariance)
        if (
            structural.content_id != _KAT_EXPECTED_OUTPUTS["structural_receipt_id"]
            or mechanics.content_id
            != _KAT_EXPECTED_OUTPUTS["covert_mechanics_receipt_id"]
            or mechanics.disposition is not CovertAuditDisposition.ABSTAIN
            or mechanics.permutations_executed_per_target != 0
            or mechanics.results != ()
            or mechanics.envelope_feature_gate_acceptable
        ):
            return _rejection(FAIL_COVERT_STRUCTURE, profile_id, authority_count)
        return Phase2BFormalTrustedWireCovertQualificationReceiptV1._issue(
            _RECEIPT_ISSUE_TOKEN,
            profile=profile,
            source_authority_content_ids=tuple(item.content_id for item in authorities),
            batch=built,
            replay=replay,
            structural_receipt=structural,
            covert_mechanics_receipt=mechanics,
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return _rejection(FAIL_COVERT_STRUCTURE, profile_id, authority_count)


def verify_phase2b_formal_trusted_wire_covert_qualification_receipt_v1(
    receipt: Phase2BFormalTrustedWireCovertQualificationReceiptV1,
) -> Phase2BFormalTrustedWireCovertQualificationReceiptV1:
    """Revalidate a success receipt and its retained public mechanics roots."""

    if type(receipt) is not Phase2BFormalTrustedWireCovertQualificationReceiptV1:
        raise TypeError("qualification receipt verifier requires the exact success type")
    receipt._validate()
    return receipt


__all__ = (
    "FAIL_CANONICAL",
    "FAIL_CLAIM_BOUNDARY",
    "FAIL_CONTENT_ID",
    "FAIL_COVERT_STRUCTURE",
    "FAIL_INPUT",
    "FAIL_JSON",
    "FAIL_KAT_INPUT",
    "FAIL_NAMESPACE",
    "FAIL_PROFILE",
    "FAIL_PUBLIC_MECHANICS",
    "PROFILE_CLAIM_LEVEL",
    "PROFILE_ID",
    "PROFILE_ID_PREFIX",
    "PROFILE_SCHEMA_VERSION",
    "PROFILE_STATUS",
    "Phase2BFormalTrustedWireCovertProfileV1",
    "Phase2BFormalTrustedWireCovertQualificationErrorV1",
    "Phase2BFormalTrustedWireCovertQualificationReceiptV1",
    "Phase2BFormalTrustedWireCovertQualificationRejectionV1",
    "QualificationDispositionV1",
    "load_phase2b_formal_trusted_wire_covert_qualification_profile_v1",
    "qualification_profile_id_v1",
    "qualify_phase2b_formal_trusted_wire_covert_scaffold_v1",
    "verify_phase2b_formal_trusted_wire_covert_qualification_receipt_v1",
)
