"""Side-effect-free external-genesis qualification for Phase-3A M2.5.

The v1.1.2 E1--E12 ambiguities are resolved deterministic prerequisites.  They
are no longer specification blockers.  External genesis is nevertheless
fail-closed until all ten committed Python/Rust errata-golden checks pass.

This module is deliberately incapable of generating randomness, creating an
Ed25519 key, writing an instantiation marker, signing a root, minting a formal
root, advancing an M3 gate, or starting M3.  The start guard returns only a
frozen, side-effect-free authorization value.  The storage, FD-3, marker,
actor-ID, and publication helpers remain read-only or pure validators.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Final, Mapping, NoReturn, Sequence

from .hashing import stable_hash


MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
CHILD_DSL_ID: Final = "hegel-old-dsl-v1.1.0"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
ERRATA_RESOLUTION_DOCUMENT: Final = (
    PROJECT_ROOT
    / "docs"
    / "Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md"
)
IMPLEMENTATION_ADDENDUM_DOCUMENT: Final = (
    PROJECT_ROOT
    / "docs"
    / "Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md"
)

ARTIFACT_NAME: Final = "phase3_m25_external_preflight_v2"
REPORT_SCHEMA: Final = "hegel-phase3-m25-external-preflight/2"
ARTIFACT_KIND: Final = "DIAGNOSTIC_NON_AUTHORITATIVE"
CURRENT_STATUS: Final = "EXACT_ERRATA_RESOLVED_DUAL_GOLDEN_VERIFICATION_REQUIRED"
CURRENT_CHILD_STATE: Final = "NOT_RUN"
M3_GATES_SATISFIED: Final = 14
M3_GATES_TOTAL: Final = 24

GATE24_NAME: Final = (
    "M3_EXECUTION_MANIFEST_ROOT_NON_NULL_AND_15_OUTPUT_ROOTS_NULL"
)
RUN_OUTPUT_SLOT_NAMES: Final = (
    "canonical_program_archive_root_or_null",
    "program_chunk_manifest_root_or_null",
    "bucket_accounting_root_or_null",
    "outside_program_output_archive_root_or_null",
    "outside_output_chunk_manifest_root_or_null",
    "outside_match_set_root_or_null",
    "outside_role_evaluation_receipt_root_or_null",
    "null_program_output_archive_root_or_null",
    "null_output_chunk_manifest_root_or_null",
    "null_match_set_root_or_null",
    "null_role_evaluation_receipt_root_or_null",
    "python_enumeration_receipt_root_or_null",
    "rust_enumeration_receipt_root_or_null",
    "dual_replay_agreement_root_or_null",
    "final_state_record_root_or_null",
)

EXTERNAL_GENESIS_START_GUARD_FIELDS: Final = (
    "errata_document_in_commit_A",
    "python_errata_vectors_pass",
    "rust_errata_vectors_pass",
    "python_rust_canonical_bytes_equal",
    "python_rust_error_codes_equal",
    "actor_trust_genesis_schema_frozen",
    "append_only_id_registry_schema_frozen",
    "parent_audit_bundle_schema_frozen",
    "bridge_statement_and_execution_v2_schema_frozen",
    "secrets_absent_from_repository",
)

FAIL_M25_EXACT_ERRATA_REQUIRED: Final = "FAIL_M25_EXACT_ERRATA_REQUIRED"
FAIL_SECRET_STATE_PATH_INVALID: Final = "FAIL_SECRET_STATE_PATH_INVALID"
FAIL_SECRET_STATE_INSIDE_REPOSITORY: Final = "FAIL_SECRET_STATE_INSIDE_REPOSITORY"
FAIL_SECRET_STATE_PERMISSIONS: Final = "FAIL_SECRET_STATE_PERMISSIONS"
FAIL_SECRET_FILE_PERMISSIONS: Final = "FAIL_SECRET_FILE_PERMISSIONS"
FAIL_SECRET_FILE_SIZE: Final = "FAIL_SECRET_FILE_SIZE"
FAIL_SECRET_PIPE_RUNTIME: Final = "FAIL_SECRET_PIPE_RUNTIME"
FAIL_SPLIT_SEED_ALREADY_INSTANTIATED: Final = (
    "FAIL_SPLIT_SEED_ALREADY_INSTANTIATED"
)
FAIL_SPLIT_SEED_PENDING_EXTERNAL_RECOVERY_REQUIRED: Final = (
    "FAIL_SPLIT_SEED_PENDING_EXTERNAL_RECOVERY_REQUIRED"
)
FAIL_ACTOR_KEY_ID_COLLISION: Final = "FAIL_ACTOR_KEY_ID_COLLISION"
FAIL_PUBLIC_BUNDLE_SECRET_FIELD: Final = "FAIL_PUBLIC_BUNDLE_SECRET_FIELD"
FAIL_PROCESS_NONZERO_EXIT: Final = "FAIL_PROCESS_NONZERO_EXIT"
FAIL_PUBLICATION_COMMIT_CONTAINS_IMPLEMENTATION_CHANGE: Final = (
    "FAIL_PUBLICATION_COMMIT_CONTAINS_IMPLEMENTATION_CHANGE"
)


class ExternalGenesisPreflightError(RuntimeError):
    """Stable fail-closed error for non-authoritative external preparation."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise ExternalGenesisPreflightError(code, detail)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_type_strict_equal(left: object, right: object) -> bool:
    """Compare diagnostic JSON without bool/int or int/float coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_type_strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_type_strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


@dataclass(frozen=True)
class ErrataOption:
    """One mutually exclusive normative choice for an exact decision."""

    option_id: str
    decision: str
    impact: str

    def to_dict(self) -> dict[str, str]:
        return {
            "option_id": self.option_id,
            "decision": self.decision,
            "impact": self.impact,
        }


@dataclass(frozen=True)
class ResolvedErrataPrerequisite:
    """One owner-resolved deterministic decision required before dual replay."""

    decision_id: str
    title: str
    evidence: tuple[str, ...]
    options: tuple[ErrataOption, ...]
    selected_option_id: str
    required_machine_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        option_ids = tuple(option.option_id for option in self.options)
        if len(option_ids) < 2 or len(option_ids) != len(set(option_ids)):
            raise ValueError("errata decision options must be unique and nontrivial")
        if self.selected_option_id not in option_ids:
            raise ValueError("selected option must be one of the frozen choices")

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "title": self.title,
            "evidence": list(self.evidence),
            "options": [option.to_dict() for option in self.options],
            "selected_option_id": self.selected_option_id,
            "required_machine_fields": list(self.required_machine_fields),
            "resolved": True,
        }


EXACT_ERRATA_PREREQUISITES: Final = (
    ResolvedErrataPrerequisite(
        decision_id="E1_M3_RUN_GENESIS_SLOT_CARDINALITY",
        title="M3RunGenesis lists 15 output slots while Gate 24 says 16",
        evidence=(
            "The M3RunGenesisV1 array names exactly 15 run-produced root slots.",
            "Section 14.2 requires all 16 run-produced output slots to be null.",
            "The existing implementation and regression test also enumerate 15.",
        ),
        options=(
            ErrataOption(
                "E1_A_EXACTLY_15",
                "Declare the listed 15 slots authoritative and correct the prose count.",
                "No wire field is added; existing listed order remains stable.",
            ),
            ErrataOption(
                "E1_B_DEFINE_16TH_SLOT",
                "Name, type, position, and semantics of a sixteenth slot.",
                "Changes M3RunGenesis bytes and requires new vectors/schema versioning.",
            ),
        ),
        selected_option_id="E1_A_EXACTLY_15",
        required_machine_fields=(
            "run_output_slot_count",
            "ordered_run_output_slot_names",
            "m3_run_genesis_schema_id",
        ),
    ),
    ResolvedErrataPrerequisite(
        decision_id="E2_BRIDGE_TOPOLOGY_ORDER",
        title="Bridge signatures precede the execution root they sign",
        evidence=(
            "The bridge signature preimage contains execution_manifest_candidate_root.",
            "The published DAG places bridge attestations at step 25 and the execution manifest at step 30.",
            "M3ExecutionManifestV1 itself contains run_id and roots produced in steps 26-29.",
        ),
        options=(
            ErrataOption(
                "E2_A_REORDER_EXECUTION_BEFORE_SIGNATURES",
                "Generate/register run_id, build the execution candidate, then collect bridge signatures.",
                "Preserves the stronger two-root signature message and removes the impossible dependency order.",
            ),
            ErrataOption(
                "E2_B_REMOVE_EXECUTION_ROOT_FROM_SIGNATURE",
                "Sign only diagnostic_formal_bridge_root.",
                "Weakens execution-specific replay protection and changes the signature preimage.",
            ),
        ),
        selected_option_id="E2_A_REORDER_EXECUTION_BEFORE_SIGNATURES",
        required_machine_fields=(
            "ordered_root_dag_steps",
            "run_id_registration_step",
            "bridge_signature_preimage_fields",
        ),
    ),
    ResolvedErrataPrerequisite(
        decision_id="E3_BRIDGE_ENVELOPE_AND_IDENTITY_BINDING",
        title="The 3-of-3 bridge statement has no unique envelope identity",
        evidence=(
            "SignedManifestEnvelopeV1 carries one enclosed root but the bridge signature covers two roots.",
            "The text does not choose one three-signature envelope versus three one-signature envelopes.",
            "Neither M3ExecutionManifestV1 nor M3RunGenesisV1 binds the bridge attestation bundle root.",
        ),
        options=(
            ErrataOption(
                "E3_A_STATEMENT_PLUS_THREE_ENVELOPES",
                "Add a bridge-statement object containing both roots, use three purpose-specific one-signature envelopes, and bind their bundle into run identity.",
                "Makes every signature and key purpose independently replayable.",
            ),
            ErrataOption(
                "E3_B_EXTEND_EXISTING_ENVELOPE",
                "Extend SignedManifestEnvelopeV1 to carry both roots and exactly three signatures.",
                "Requires envelope schema/version changes and a single epoch policy.",
            ),
        ),
        selected_option_id="E3_A_STATEMENT_PLUS_THREE_ENVELOPES",
        required_machine_fields=(
            "bridge_statement_tag",
            "bridge_statement_schema_id",
            "bridge_statement_hash_domain",
            "bridge_envelope_cardinality",
            "bridge_attestation_bundle_binding_location",
        ),
    ),
    ResolvedErrataPrerequisite(
        decision_id="E4_MISSING_HASH_DOMAINS",
        title="Required envelope and agreement roots lack hash domains",
        evidence=(
            "SignedManifestEnvelopeV1 is referenced by signed_envelope_root but has no root formula/domain.",
            "M3DualReplayAgreementV1 has a schema but no ContentHash domain.",
            "A raw SHA-256 guess would violate the project's domain-separated identity policy.",
        ),
        options=(
            ErrataOption(
                "E4_A_DOMAIN_SEPARATED_CONTENT_HASH",
                "Assign explicit ContentHash domains to both formal objects.",
                "Matches the existing formal identity convention.",
            ),
            ErrataOption(
                "E4_B_UNDOMAINED_SHA256",
                "Use SHA-256 over canonical CBOR without a domain.",
                "Creates a policy exception and must be explicitly frozen.",
            ),
        ),
        selected_option_id="E4_A_DOMAIN_SEPARATED_CONTENT_HASH",
        required_machine_fields=(
            "signed_manifest_envelope_root_formula",
            "signed_manifest_envelope_hash_domain",
            "m3_dual_replay_agreement_root_formula",
            "m3_dual_replay_agreement_hash_domain",
        ),
    ),
    ResolvedErrataPrerequisite(
        decision_id="E5_ACTOR_KEY_TRUST_AND_PURPOSE",
        title="Actor key roots and signer purposes are not anchored",
        evidence=(
            "ActorKeyManifestV1 roots are not transitively bound by execution/genesis identity.",
            "The pinned custodian and bridge attester trust-anchor source is unspecified.",
            "It is unclear whether the custodian manifest signer is the purpose-1 bridge key.",
            "SignedManifestEnvelopeV1 names custodian_key_epoch even for auditor/attester use.",
        ),
        options=(
            ErrataOption(
                "E5_A_PURPOSE1_IS_CUSTODIAN_IDENTITY",
                "Treat purpose 1 as the same custodian identity for domain-separated manifest and bridge signatures; bind all actor-key roots in a trust-genesis bundle.",
                "Uses one custodian key without cross-numeric-purpose reuse.",
            ),
            ErrataOption(
                "E5_B_SEPARATE_CUSTODY_SIGNER_PURPOSE",
                "Add a distinct custodian-manifest-signer purpose/key and bind it beside the bridge key.",
                "Adds a fifth M2.5 actor key and new purpose registry entry.",
            ),
        ),
        selected_option_id="E5_A_PURPOSE1_IS_CUSTODIAN_IDENTITY",
        required_machine_fields=(
            "actor_key_root_binding_object",
            "pinned_genesis_trust_anchor",
            "custodian_manifest_signer_purpose_id",
            "noncustodian_envelope_epoch_semantics",
        ),
    ),
    ResolvedErrataPrerequisite(
        decision_id="E6_LEDGER_CALCULATOR_ACTOR_BOUNDARY",
        title="FD-3 split replay conflicts with a genesis-only hidden ledger",
        evidence=(
            "Gate 16 requires one genesis record and no access/reveal event.",
            "The custodian must provide the raw seed to Python and Rust calculators via FD 3.",
            "The hidden-artifact registry classifies raw-seed access as event type 2.",
        ),
        options=(
            ErrataOption(
                "E6_A_CALCULATORS_INSIDE_CUSTODIAN_BOUNDARY",
                "Define the two calculators as constrained custodian subprocesses rather than independent ledger actors.",
                "Preserves genesis-only Gate 16 while retaining implementation diversity.",
            ),
            ErrataOption(
                "E6_B_RECORD_CALCULATOR_ACCESS",
                "Append two authorized access events and change Gate 16/head predicates.",
                "Changes ledger roots, gate semantics, and execution inputs.",
            ),
        ),
        selected_option_id="E6_A_CALCULATORS_INSIDE_CUSTODIAN_BOUNDARY",
        required_machine_fields=(
            "calculator_actor_boundary",
            "fd3_access_event_required",
            "gate16_required_ledger_count",
            "gate16_required_head_rule",
        ),
    ),
    ResolvedErrataPrerequisite(
        decision_id="E7_PARENT_ABSENCE_AUDIT_WIRE",
        title="Parent-history and two legacy sources lack one auditable wire",
        evidence=(
            "AuditedPathBlobRecordV1 tag 0x3210 is absent from the tag registry.",
            "audited_path_set_root and history records have no complete row schemas/order/root rules.",
            "The described history root has no field in ParentManifestAbsenceAttestationV1.",
            "The attestation has one legacy source digest while odd and sink currently have distinct legacy source IDs.",
        ),
        options=(
            ErrataOption(
                "E7_A_VERSIONED_AUDIT_BUNDLE",
                "Define path, history, and legacy-source rows plus a versioned audit-bundle root referenced by a revised absence attestation.",
                "Binds the complete audit without overloading existing root meanings.",
            ),
            ErrataOption(
                "E7_B_COMPOSITE_EXISTING_ROOTS",
                "Define audited_source_tree_root/audited_path_set_root as composite trees that include history and both source IDs.",
                "Avoids a new top-level object but requires exact heterogeneous row tagging.",
            ),
        ),
        selected_option_id="E7_A_VERSIONED_AUDIT_BUNDLE",
        required_machine_fields=(
            "audited_path_blob_tag_registry_entry",
            "audited_path_set_row_schema",
            "audited_history_row_schema",
            "audit_tree_ordering",
            "audit_root_binding_field",
            "legacy_parent_payload_source_ids",
        ),
    ),
    ResolvedErrataPrerequisite(
        decision_id="E8_ROLE_AND_INITIAL_STATE_ENUMS",
        title="Target role and M3RunGenesis initial-state bytes are ambiguous",
        evidence=(
            "M3StateId.NOT_RUN is 0 while ChildInitialStateId.NOT_RUN is 1.",
            "M3RunGenesisV1.initial_state_id does not name which registry applies.",
            "role_id appears in split, target, role-binding, and receipts without a TargetRoleId registry.",
            "ArtifactRoleId describes artifact classes and cannot safely be assumed to be target roles.",
        ),
        options=(
            ErrataOption(
                "E8_A_M3STATE0_NEW_TARGET_ROLE_ENUM",
                "Bind M3RunGenesis.initial_state_id to M3StateId and add TargetRoleId {OUTSIDE=1,NULL=2}.",
                "Separates execution roles from artifact-role IDs.",
            ),
            ErrataOption(
                "E8_B_CHILDSTATE1_ARTIFACT_ROLE_REUSE",
                "Use ChildInitialStateId and reuse selected ArtifactRoleId values for target roles.",
                "Avoids an enum but couples unrelated namespaces.",
            ),
        ),
        selected_option_id="E8_A_M3STATE0_NEW_TARGET_ROLE_ENUM",
        required_machine_fields=(
            "m3_run_genesis_initial_state_registry",
            "m3_run_genesis_initial_state_value",
            "target_role_enum_registry",
            "role_id_field_registry_map",
        ),
    ),
    ResolvedErrataPrerequisite(
        decision_id="E9_ROOT_PREIMAGES_INSTANCE_IDS_AND_PATH_ALIAS",
        title="Several named roots and machine-ID instances still lack exact preimages",
        evidence=(
            "HiddenArtifactScope has prose/YAML but no formal schema/domain.",
            "Canonical AST/CBOR profile, Phase-2B contract, and MDL table roots lack a unique preimage object.",
            "Implementation source/dependency roots and StateMachineContract legal-transition rows are incomplete.",
            "The inherited M3RunState example conflicts with the common [version, tag, schema_id] prefix.",
            "InputSignatureSpec.static_role_metadata has no nested payload schema.",
            "TraversalContract and BucketAccountingContract field IDs/invariants have no registries or nested schemas.",
            "TargetSpecFormal.claim_level_id and MismatchRecord.mismatch_kind_id have no enum registries.",
            "SplitContract.assignment_ordering_rule_id and fallback_split_policy_id have no numeric registry.",
            "Many profile/order/fallback *_id_digest values lack one exact machine-ID catalog.",
            "The literal repository path 'Hegel Machine/...' contains a space forbidden by IdDigestV1 syntax.",
        ),
        options=(
            ErrataOption(
                "E9_A_FORMAL_PREIMAGE_AND_ALIAS_REGISTRY",
                "Add formal preimage schemas plus one machine-ID/path-alias registry; bind a legal ASCII alias to raw repository path bytes.",
                "Preserves IdDigestV1 and makes every named root replayable.",
            ),
            ErrataOption(
                "E9_B_PATH_BYTES_AND_DOCUMENT_ROOTS",
                "Use raw path byte strings for path fields and explicitly reuse normative-document roots for named profile roots.",
                "Requires field/schema changes and explicit role-to-document mapping.",
            ),
        ),
        selected_option_id="E9_A_FORMAL_PREIMAGE_AND_ALIAS_REGISTRY",
        required_machine_fields=(
            "formal_root_preimage_registry",
            "instance_machine_id_catalog",
            "repository_path_alias_rule",
            "amendment_document_path_alias",
            "source_and_dependency_root_row_schemas",
            "state_machine_nested_row_schemas",
            "m3_run_state_exact_prefix",
            "input_signature_static_role_metadata_schema",
            "target_claim_level_enum_registry",
            "mismatch_kind_enum_registry",
            "split_contract_numeric_rule_registry",
            "traversal_and_bucket_field_id_registries",
        ),
    ),
    ResolvedErrataPrerequisite(
        decision_id="E10_OPAQUE_ID_REGISTRY_EVIDENCE",
        title="Fresh run/ledger IDs have no replayable registry evidence",
        evidence=(
            "AppendOnlyOpaqueIdRegistryV1 is named but has no file/wire/root definition.",
            "Gate 24 requires the run ID to be fresh and registered.",
            "No execution/genesis field binds evidence for the project-wide duplicate scope.",
        ),
        options=(
            ErrataOption(
                "E10_A_FORMAL_APPEND_ONLY_REGISTRY_ROOT",
                "Define registry rows/root and bind the current registry root into genesis qualification.",
                "Makes freshness independently replayable.",
            ),
            ErrataOption(
                "E10_B_OPERATIONAL_OEXCL_REGISTRY",
                "Freeze an external one-file-per-ID O_EXCL registry and a verifier scan over published identities.",
                "Provides operational safety but needs an exact evidence receipt for Gate 24.",
            ),
        ),
        selected_option_id="E10_A_FORMAL_APPEND_ONLY_REGISTRY_ROOT",
        required_machine_fields=(
            "opaque_id_registry_record_schema",
            "opaque_id_registry_ordering",
            "opaque_id_registry_root_formula",
            "opaque_id_registry_binding_location",
            "duplicate_scope_verification_rule",
        ),
    ),
    ResolvedErrataPrerequisite(
        decision_id="E11_NULL_WITNESS_BINDING_FIELD",
        title="The sink witness is assigned to a nonexistent role-binding field",
        evidence=(
            "Section 4.6 names DslRoleBindingManifestV1.required_witness_ast_hash.",
            "DslRoleBindingManifestV1 has no such field.",
            "TargetSpecFormalV1 and TargetBundleV1 already contain witness-hash slots.",
        ),
        options=(
            ErrataOption(
                "E11_A_USE_TARGET_SPEC_AND_BUNDLE_FIELDS",
                "Correct the prose to bind the witness in the two existing target fields.",
                "No wire change; role binding reaches it through semantic_spec_formal_root.",
            ),
            ErrataOption(
                "E11_B_ADD_ROLE_BINDING_FIELD",
                "Add required_witness_ast_hash to a new DslRoleBinding schema version.",
                "Changes role-binding roots and downstream execution identity.",
            ),
        ),
        selected_option_id="E11_A_USE_TARGET_SPEC_AND_BUNDLE_FIELDS",
        required_machine_fields=(
            "authoritative_witness_binding_fields",
            "dsl_role_binding_schema_change_required",
        ),
    ),
    ResolvedErrataPrerequisite(
        decision_id="E12_CUSTODIAN_ENVELOPE_COVERAGE",
        title="The topology and signature guard disagree on signed custodian objects",
        evidence=(
            "The DAG requires final CustodianBindingManifestV1 plus a signature envelope.",
            "The inherited exact one-signature guard names only seed commitment, seed continuity, and ledger genesis.",
            "Envelope-root binding locations for all custodian signatures remain incomplete.",
        ),
        options=(
            ErrataOption(
                "E12_A_SIGN_FOUR_CUSTODIAN_OBJECTS",
                "Require one pinned custodian signature for tags 0x3103, 0x3105, 0x3106, and 0x3108 and bind all envelope roots.",
                "Matches the published topology and makes final custody replayable.",
            ),
            ErrataOption(
                "E12_B_REMOVE_FINAL_BINDING_SIGNATURE",
                "Keep signatures only on the inherited three objects and correct the topology.",
                "Leaves final custodian binding authenticated only transitively/self-consistently.",
            ),
        ),
        selected_option_id="E12_A_SIGN_FOUR_CUSTODIAN_OBJECTS",
        required_machine_fields=(
            "custodian_signed_object_tags",
            "custodian_signature_domain_by_tag",
            "custodian_envelope_root_binding_locations",
            "required_signature_count_by_tag",
        ),
    ),
)

EXACT_ERRATA_BY_ID: Final = MappingProxyType(
    {
        prerequisite.decision_id: prerequisite
        for prerequisite in EXACT_ERRATA_PREREQUISITES
    }
)


@dataclass(frozen=True)
class DualGoldenVerification:
    """The ten exact, side-effect-free preconditions from errata section 3."""

    errata_document_in_commit_A: bool
    python_errata_vectors_pass: bool
    rust_errata_vectors_pass: bool
    python_rust_canonical_bytes_equal: bool
    python_rust_error_codes_equal: bool
    actor_trust_genesis_schema_frozen: bool
    append_only_id_registry_schema_frozen: bool
    parent_audit_bundle_schema_frozen: bool
    bridge_statement_and_execution_v2_schema_frozen: bool
    secrets_absent_from_repository: bool

    def __post_init__(self) -> None:
        for field_name in EXTERNAL_GENESIS_START_GUARD_FIELDS:
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool")

    @classmethod
    def unverified(cls) -> "DualGoldenVerification":
        return cls(*(False for _ in EXTERNAL_GENESIS_START_GUARD_FIELDS))

    def to_dict(self) -> dict[str, bool]:
        return {
            field_name: getattr(self, field_name)
            for field_name in EXTERNAL_GENESIS_START_GUARD_FIELDS
        }

    @property
    def all_required_checks_pass(self) -> bool:
        return all(self.to_dict().values())


@dataclass(frozen=True)
class ExternalGenesisStartAuthorization:
    """Pure authorization result; possession performs and proves no side effect."""

    machine_freeze_id: str
    child_dsl_id: str
    external_genesis_start_allowed: bool
    authorization_is_side_effect_free: bool
    m3_gates_satisfied: int
    m3_gates_total: int
    child_state: str
    gate24_qualified: bool
    m3_entry_allowed: bool
    m3_run_started: bool
    phase3_m3_start_authorized: bool
    external_object_repository_commit_rule: str
    publication_commit_may_substitute: bool


def validate_dual_golden_verification(
    verification: DualGoldenVerification | Mapping[str, object],
) -> DualGoldenVerification:
    """Validate the exact ten-field guard input without granting authority."""

    if isinstance(verification, DualGoldenVerification):
        return verification
    if not isinstance(verification, Mapping):
        raise TypeError("dual-golden verification must be a mapping or frozen record")
    if set(verification) != set(EXTERNAL_GENESIS_START_GUARD_FIELDS):
        _fail(
            FAIL_M25_EXACT_ERRATA_REQUIRED,
            "dual-golden external-start guard field-set mismatch",
        )
    values: list[bool] = []
    for field_name in EXTERNAL_GENESIS_START_GUARD_FIELDS:
        value = verification[field_name]
        if type(value) is not bool:
            _fail(
                FAIL_M25_EXACT_ERRATA_REQUIRED,
                f"dual-golden guard field {field_name} must be bool",
            )
        values.append(value)
    return DualGoldenVerification(*values)


def external_genesis_start_guard_report(
    verification: DualGoldenVerification | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate the ten prerequisites without CSPRNG, marker, signature, or root."""

    snapshot = validate_dual_golden_verification(
        DualGoldenVerification.unverified() if verification is None else verification
    )
    checks = snapshot.to_dict()
    passed = sum(value is True for value in checks.values())
    return {
        "required_check_count": len(EXTERNAL_GENESIS_START_GUARD_FIELDS),
        "passed_check_count": passed,
        "checks": checks,
        "all_required_checks_pass": snapshot.all_required_checks_pass,
        "external_genesis_start_allowed": snapshot.all_required_checks_pass,
        "guard_evaluation_is_side_effect_free": True,
        "commit_A_binding_required": True,
        "commit_A_check_semantics": (
            "The committed normative bundle contains the base amendment, errata "
            "resolution, and implementation addendum and binds the deterministic "
            "implementation basis commit."
        ),
        "gate_effect": "NONE",
        "child_state_effect": "NONE",
    }


def external_genesis_preflight_report() -> dict[str, object]:
    """Return current v2 readiness: errata resolved, dual evidence not yet bound."""

    guard = external_genesis_start_guard_report()
    payload: dict[str, object] = {
        "artifact": ARTIFACT_NAME,
        "schema_version": REPORT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "child_dsl_id": CHILD_DSL_ID,
        "status": CURRENT_STATUS,
        "errata_resolution_document_sha256": _sha256_file(
            ERRATA_RESOLUTION_DOCUMENT
        ),
        "implementation_addendum_document_sha256": _sha256_file(
            IMPLEMENTATION_ADDENDUM_DOCUMENT
        ),
        "child_state": CURRENT_CHILD_STATE,
        "m3_gates_satisfied": M3_GATES_SATISFIED,
        "m3_gates_total": M3_GATES_TOTAL,
        "m3_entry_allowed": False,
        "m3_entry_qualified": False,
        "m3_run_started": False,
        "external_genesis_start_allowed": guard[
            "external_genesis_start_allowed"
        ],
        "exact_errata_resolved": True,
        "unresolved_specification_blockers": [],
        "resolved_errata_prerequisite_count": len(EXACT_ERRATA_PREREQUISITES),
        "resolved_errata_prerequisites": [
            prerequisite.to_dict()
            for prerequisite in EXACT_ERRATA_PREREQUISITES
        ],
        "external_genesis_start_guard": guard,
        "commit_A_binding_semantics": {
            "normative_document_bundle_roles": [
                "BASE_AMENDMENT",
                "ERRATA_RESOLUTION",
                "IMPLEMENTATION_CLOSURE_ADDENDUM",
            ],
            "repository_commit_role": "DETERMINISTIC_IMPLEMENTATION_BASIS_COMMIT_A",
            "publication_commit_may_substitute": False,
        },
        "gate24_contract": {
            "gate_name": GATE24_NAME,
            "current_status": "NOT_EVALUATED_EXTERNAL_INPUTS_ABSENT",
            "ordered_run_output_slot_names": list(RUN_OUTPUT_SLOT_NAMES),
            "pass_predicate": {
                "m3_execution_manifest_v2_root_non_null": True,
                "m3_run_genesis_v1_root_non_null": True,
                "m3_run_genesis_initial_state": "M3StateId.NOT_RUN = 0",
                "run_output_slot_count": len(RUN_OUTPUT_SLOT_NAMES),
                "all_run_output_slots_null": True,
                "run_id_registered_in_bound_opaque_id_snapshot": True,
                "bridge_envelope_count": 3,
                "bridge_signer_purposes_exactly": [1, 2, 3],
            },
            "current_evidence": {
                "m3_execution_manifest_v2_root_non_null": False,
                "m3_run_genesis_v1_root_non_null": False,
                "run_id_registered_in_bound_opaque_id_snapshot": False,
                "bridge_envelope_count": 0,
                "bridge_signer_purposes": [],
            },
            "gate24_passed": False,
            "qualification_effect_if_passed": {
                "m3_entry_qualified": True,
                "m3_entry_allowed": True,
                "m3_run_started": False,
                "child_state": "NOT_RUN",
            },
        },
        "run_output_slots": {name: None for name in RUN_OUTPUT_SLOT_NAMES},
        "phase3_m3_start_contract": {
            "action_id": "phase3-m3-start",
            "implementation_status": "SEMANTICS_FROZEN_NOT_INVOKED",
            "requires_complete_24_of_24_replay": True,
            "requires_bound_opaque_id_snapshot": True,
            "only_transition": (
                "NOT_RUN/NONE -> RUNNING/CANONICAL_ENUMERATION"
            ),
            "transition_index": 0,
            "previous_state_record_root": None,
            "transition_reason": "ENTRY_GATES_24_OF_24",
            "triggering_receipt_root": None,
            "start_record_created": False,
        },
        "authority_side_effects": {
            "os_csprng_called": False,
            "instantiation_marker_created": False,
            "real_seed_generated": False,
            "real_private_key_generated": False,
            "real_signature_generated": False,
            "formal_root_claimed": False,
            "gate_15_24_advanced": False,
        },
        "allowed_current_scope": [
            "deterministic_schema_and_enum_implementation",
            "dual_errata_golden_qualification",
            "public_typed_row_and_root_replay",
            "synthetic_split_and_signature_fault_injection",
            "read_only_storage_fd3_marker_policy_validation",
        ],
        "claim_boundary": (
            "E1--E12 are resolved deterministic prerequisites, but the committed "
            "dual-golden evidence and external actors are not yet bound. This v2 "
            "diagnostic performs no CSPRNG call, marker creation, seed/key "
            "generation, signature, authoritative formal-root generation, M3 "
            "gate advancement, Gate-24 qualification, or state transition."
        ),
    }
    payload["diagnostic_report_id"] = stable_hash(
        payload,
        prefix="phase3_m25_external_preflight_",
    )
    return payload


def validate_external_genesis_preflight_report(report: Mapping[str, object]) -> None:
    """Reject stale v1 data, unresolved prerequisites, or authority escalation."""

    if not isinstance(report, Mapping):
        raise TypeError("external genesis preflight report must be a mapping")
    expected = external_genesis_preflight_report()
    if not _json_type_strict_equal(dict(report), expected):
        _fail(
            FAIL_M25_EXACT_ERRATA_REQUIRED,
            "external genesis v2 preflight differs from current resolved state",
        )


def assert_external_genesis_start_allowed(
    verification: DualGoldenVerification | Mapping[str, object] | None = None,
) -> ExternalGenesisStartAuthorization:
    """Return pure authorization only after all ten committed checks pass."""

    snapshot = validate_dual_golden_verification(
        DualGoldenVerification.unverified() if verification is None else verification
    )
    failed = [
        field_name
        for field_name, passed in snapshot.to_dict().items()
        if passed is not True
    ]
    if failed:
        _fail(
            FAIL_M25_EXACT_ERRATA_REQUIRED,
            "external genesis is blocked before CSPRNG/marker by dual-golden "
            "guard fields: " + ",".join(failed),
        )
    return ExternalGenesisStartAuthorization(
        machine_freeze_id=MACHINE_FREEZE_ID,
        child_dsl_id=CHILD_DSL_ID,
        external_genesis_start_allowed=True,
        authorization_is_side_effect_free=True,
        m3_gates_satisfied=M3_GATES_SATISFIED,
        m3_gates_total=M3_GATES_TOTAL,
        child_state=CURRENT_CHILD_STATE,
        gate24_qualified=False,
        m3_entry_allowed=False,
        m3_run_started=False,
        phase3_m3_start_authorized=False,
        external_object_repository_commit_rule="USE_DETERMINISTIC_BASIS_COMMIT_A",
        publication_commit_may_substitute=False,
    )


def _resolved_non_symlink(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        _fail(FAIL_SECRET_STATE_PATH_INVALID, f"{label} may not be a symlink")
    try:
        return path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        _fail(FAIL_SECRET_STATE_PATH_INVALID, f"{label} is unavailable: {exc}")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_secret_state_directory(
    secret_state_directory: Path,
    *,
    repository_root: Path,
) -> Path:
    """Read-only validation of the minimum external ``0700`` directory policy."""

    if not isinstance(secret_state_directory, Path) or not isinstance(
        repository_root, Path
    ):
        raise TypeError("secret-state and repository paths must be pathlib.Path")
    secret = _resolved_non_symlink(secret_state_directory, label="secret state directory")
    repository = _resolved_non_symlink(repository_root, label="repository root")
    if not secret.is_dir():
        _fail(FAIL_SECRET_STATE_PATH_INVALID, "secret state path is not a directory")
    if secret == repository or _is_within(secret, repository):
        _fail(
            FAIL_SECRET_STATE_INSIDE_REPOSITORY,
            "secret state directory must be outside the repository",
        )
    mode = stat.S_IMODE(secret.stat().st_mode)
    if mode != 0o700:
        _fail(
            FAIL_SECRET_STATE_PERMISSIONS,
            f"secret state directory mode must be 0700, got {mode:04o}",
        )
    return secret


def validate_secret_file(
    secret_file: Path,
    *,
    secret_state_directory: Path,
    expected_size: int | None = None,
) -> Path:
    """Read-only validation of one external regular ``0600`` secret file."""

    if not isinstance(secret_file, Path) or not isinstance(secret_state_directory, Path):
        raise TypeError("secret file and state directory must be pathlib.Path")
    if expected_size is not None and (
        type(expected_size) is not int or expected_size < 0
    ):
        raise TypeError("expected_size must be a nonnegative integer or None")
    state = _resolved_non_symlink(secret_state_directory, label="secret state directory")
    path = _resolved_non_symlink(secret_file, label="secret file")
    if not state.is_dir():
        _fail(FAIL_SECRET_STATE_PATH_INVALID, "secret state path is not a directory")
    state_mode = stat.S_IMODE(state.stat().st_mode)
    if state_mode != 0o700:
        _fail(
            FAIL_SECRET_STATE_PERMISSIONS,
            f"secret state directory mode must be 0700, got {state_mode:04o}",
        )
    if not path.is_file() or not _is_within(path, state):
        _fail(
            FAIL_SECRET_STATE_PATH_INVALID,
            "secret file must be a regular file inside the secret state directory",
        )
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        _fail(
            FAIL_SECRET_FILE_PERMISSIONS,
            f"secret file mode must be 0600, got {mode:04o}",
        )
    if expected_size is not None and path.stat().st_size != expected_size:
        _fail(
            FAIL_SECRET_FILE_SIZE,
            f"secret file must contain exactly {expected_size} bytes",
        )
    return path


def validate_secret_fd_number(fd: object) -> int:
    """Require the frozen inherited anonymous-pipe descriptor number 3."""

    if type(fd) is not int or fd != 3:
        _fail(FAIL_SECRET_PIPE_RUNTIME, "authoritative secret pipe must be FD 3")
    return fd


def validate_secret_fd_payload(payload: object) -> bytes:
    """Validate an already-read synthetic FD payload without retaining it."""

    if type(payload) is not bytes or len(payload) != 32:
        _fail(
            FAIL_SECRET_PIPE_RUNTIME,
            "split calculator must read exactly 32 bytes then EOF from FD 3",
        )
    return payload


@dataclass(frozen=True)
class MarkerSnapshot:
    """In-memory view of a marker; this module never creates or rewrites it."""

    state: str
    split_version_digest: bytes
    seed_commitment_manifest_root: bytes | None
    custodian_key_id: bytes
    created_at_unix_seconds: int


def validate_marker_snapshot(snapshot: MarkerSnapshot) -> MarkerSnapshot:
    """Validate PENDING/COMPLETE invariants without touching persistent state."""

    if not isinstance(snapshot, MarkerSnapshot):
        raise TypeError("marker snapshot must be MarkerSnapshot")
    if snapshot.state not in {"PENDING", "COMPLETE"}:
        _fail(FAIL_SECRET_STATE_PATH_INVALID, "marker state must be PENDING or COMPLETE")
    if type(snapshot.split_version_digest) is not bytes or len(
        snapshot.split_version_digest
    ) != 32:
        _fail(FAIL_SECRET_STATE_PATH_INVALID, "split version digest must be 32 bytes")
    if type(snapshot.custodian_key_id) is not bytes or len(snapshot.custodian_key_id) != 16:
        _fail(FAIL_SECRET_STATE_PATH_INVALID, "custodian key ID must be 16 bytes")
    if (
        type(snapshot.created_at_unix_seconds) is not int
        or snapshot.created_at_unix_seconds < 0
    ):
        _fail(FAIL_SECRET_STATE_PATH_INVALID, "marker timestamp must be nonnegative")
    root = snapshot.seed_commitment_manifest_root
    if snapshot.state == "PENDING" and root is not None:
        _fail(FAIL_SECRET_STATE_PATH_INVALID, "PENDING marker must not contain a root")
    if snapshot.state == "COMPLETE" and (type(root) is not bytes or len(root) != 32):
        _fail(FAIL_SECRET_STATE_PATH_INVALID, "COMPLETE marker requires a 32-byte root")
    return snapshot


def assert_seed_instantiation_marker_absent(*, marker_exists: bool) -> None:
    """Pure second-invocation guard used before a future atomic-create call."""

    if type(marker_exists) is not bool:
        raise TypeError("marker_exists must be bool")
    if marker_exists:
        _fail(
            FAIL_SPLIT_SEED_ALREADY_INSTANTIATED,
            "an existing marker prohibits a second CSPRNG invocation",
        )


def assert_marker_does_not_require_external_recovery(
    snapshot: MarkerSnapshot,
) -> None:
    """A PENDING crash state may never trigger an automatic redraw."""

    validate_marker_snapshot(snapshot)
    if snapshot.state == "PENDING":
        _fail(
            FAIL_SPLIT_SEED_PENDING_EXTERNAL_RECOVERY_REQUIRED,
            "PENDING marker requires external recovery; automatic redraw is forbidden",
        )


def validate_distinct_actor_key_ids(
    *,
    custodian_key_id: bytes,
    python_attester_key_id: bytes,
    rust_attester_key_id: bytes,
    auditor_key_id: bytes,
) -> tuple[bytes, bytes, bytes, bytes]:
    """Require four exact, pairwise-distinct 16-byte actor key IDs."""

    values = (
        custodian_key_id,
        python_attester_key_id,
        rust_attester_key_id,
        auditor_key_id,
    )
    if any(type(value) is not bytes or len(value) != 16 for value in values):
        _fail(FAIL_ACTOR_KEY_ID_COLLISION, "every actor key ID must be 16 bytes")
    if len(set(values)) != len(values):
        _fail(FAIL_ACTOR_KEY_ID_COLLISION, "actor key IDs must be pairwise distinct")
    return values


_FORBIDDEN_PUBLIC_FIELD_NAMES: Final = frozenset(
    {
        "raw_private_key",
        "private_key",
        "private_key_seed",
        "raw_split_seed",
        "split_master_seed",
        "master_seed_hex",
        "derived_role_key",
        "k_role",
        "assignment_rows",
        "validation_membership",
        "sealed_membership",
        "sealed_prediction_membership",
        "pre_final_match_set",
        "pre_final_output_archive",
    }
)


def assert_public_payload_contains_no_secret_fields(payload: object) -> None:
    """Recursively reject explicitly forbidden secret-bearing public fields."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str):
                _fail(
                    FAIL_PUBLIC_BUNDLE_SECRET_FIELD,
                    "diagnostic public mappings require string field names",
                )
            if key.lower() in _FORBIDDEN_PUBLIC_FIELD_NAMES:
                _fail(
                    FAIL_PUBLIC_BUNDLE_SECRET_FIELD,
                    f"forbidden public field {key!r}",
                )
            assert_public_payload_contains_no_secret_fields(value)
    elif isinstance(payload, (tuple, list)):
        for value in payload:
            assert_public_payload_contains_no_secret_fields(value)


def validate_calculator_process_result(
    *,
    exit_code: object,
    public_payload: object,
) -> None:
    """Require clean exit and reject known secret-bearing public field names.

    This syntactic lint is not evidence that an arbitrary value is non-secret;
    a future trusted calculator output still needs an exact allowlisted
    executable digest plus signed external-genesis evidence.
    """

    if type(exit_code) is not int or exit_code != 0:
        _fail(FAIL_PROCESS_NONZERO_EXIT, "split calculator must exit with code 0")
    assert_public_payload_contains_no_secret_fields(public_payload)


def validate_commit_b_changed_paths(
    changed_paths: Sequence[str],
    *,
    allowed_public_prefixes: Sequence[str],
    executable_prefixes: Sequence[str],
) -> tuple[str, ...]:
    """Pure allowlist guard for the Commit-A to Commit-B publication diff."""

    if isinstance(changed_paths, (str, bytes)) or isinstance(
        allowed_public_prefixes, (str, bytes)
    ) or isinstance(executable_prefixes, (str, bytes)):
        raise TypeError("path collections must be non-string sequences")
    normalized = tuple(changed_paths)
    if any(not isinstance(path, str) or not path for path in normalized):
        raise TypeError("changed paths must be nonempty strings")
    for path in normalized:
        if any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in executable_prefixes):
            _fail(
                FAIL_PUBLICATION_COMMIT_CONTAINS_IMPLEMENTATION_CHANGE,
                f"publication changes executable path {path!r}",
            )
        if not any(
            path == prefix or path.startswith(prefix.rstrip("/") + "/")
            for prefix in allowed_public_prefixes
        ):
            _fail(
                FAIL_PUBLICATION_COMMIT_CONTAINS_IMPLEMENTATION_CHANGE,
                f"publication path {path!r} is outside the public allowlist",
            )
    return normalized


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_NAME",
    "CHILD_DSL_ID",
    "CURRENT_CHILD_STATE",
    "CURRENT_STATUS",
    "DualGoldenVerification",
    "ERRATA_RESOLUTION_DOCUMENT",
    "EXACT_ERRATA_PREREQUISITES",
    "EXACT_ERRATA_BY_ID",
    "EXTERNAL_GENESIS_START_GUARD_FIELDS",
    "ExternalGenesisStartAuthorization",
    "ExternalGenesisPreflightError",
    "FAIL_ACTOR_KEY_ID_COLLISION",
    "FAIL_M25_EXACT_ERRATA_REQUIRED",
    "FAIL_PROCESS_NONZERO_EXIT",
    "FAIL_PUBLICATION_COMMIT_CONTAINS_IMPLEMENTATION_CHANGE",
    "FAIL_PUBLIC_BUNDLE_SECRET_FIELD",
    "FAIL_SECRET_FILE_PERMISSIONS",
    "FAIL_SECRET_FILE_SIZE",
    "FAIL_SECRET_PIPE_RUNTIME",
    "FAIL_SECRET_STATE_INSIDE_REPOSITORY",
    "FAIL_SECRET_STATE_PATH_INVALID",
    "FAIL_SECRET_STATE_PERMISSIONS",
    "FAIL_SPLIT_SEED_ALREADY_INSTANTIATED",
    "FAIL_SPLIT_SEED_PENDING_EXTERNAL_RECOVERY_REQUIRED",
    "GATE24_NAME",
    "IMPLEMENTATION_ADDENDUM_DOCUMENT",
    "M3_GATES_SATISFIED",
    "M3_GATES_TOTAL",
    "MACHINE_FREEZE_ID",
    "MarkerSnapshot",
    "REPORT_SCHEMA",
    "RUN_OUTPUT_SLOT_NAMES",
    "assert_external_genesis_start_allowed",
    "assert_marker_does_not_require_external_recovery",
    "assert_public_payload_contains_no_secret_fields",
    "assert_seed_instantiation_marker_absent",
    "external_genesis_preflight_report",
    "external_genesis_start_guard_report",
    "validate_calculator_process_result",
    "validate_commit_b_changed_paths",
    "validate_distinct_actor_key_ids",
    "validate_dual_golden_verification",
    "validate_external_genesis_preflight_report",
    "validate_marker_snapshot",
    "validate_secret_fd_number",
    "validate_secret_fd_payload",
    "validate_secret_file",
    "validate_secret_state_directory",
]
