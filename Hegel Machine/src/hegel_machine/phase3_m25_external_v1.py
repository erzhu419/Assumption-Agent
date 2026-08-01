"""Fail-closed external-genesis preflight for Phase-3A M2.5.

This module is deliberately incapable of generating randomness, creating an
Ed25519 key, writing an instantiation marker, signing a root, or advancing an
M3 gate.  It records the exact v1.1.2 errata that still change formal bytes,
root identity, actor authority, or state.  The guard
:func:`assert_external_genesis_start_allowed` therefore fails before a future
external workflow may call an OS CSPRNG or create its ``O_EXCL`` marker.

The storage, FD-3, marker, actor-ID, and publication helpers are read-only or
pure validators.  They support synthetic fault-injection tests without
weakening the external-actor boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Final, Mapping, NoReturn, Sequence

from .hashing import stable_hash


MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
CHILD_DSL_ID: Final = "hegel-old-dsl-v1.1.0"
ARTIFACT_NAME: Final = "phase3_m25_external_preflight_v1"
ARTIFACT_KIND: Final = "DIAGNOSTIC_NON_AUTHORITATIVE"
CURRENT_STATUS: Final = "EXACT_ERRATA_REQUIRED_EXTERNAL_GENESIS_BLOCKED"
CURRENT_CHILD_STATE: Final = "NOT_RUN"
M3_GATES_SATISFIED: Final = 14
M3_GATES_TOTAL: Final = 24

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
    """One mutually exclusive normative choice for an exact blocker."""

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
class ExactErrataBlocker:
    """One unresolved choice that changes bytes, roots, authority, or state."""

    blocker_id: str
    title: str
    evidence: tuple[str, ...]
    options: tuple[ErrataOption, ...]
    recommended_option_id: str
    required_machine_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        option_ids = tuple(option.option_id for option in self.options)
        if len(option_ids) < 2 or len(option_ids) != len(set(option_ids)):
            raise ValueError("errata blocker options must be unique and nontrivial")
        if self.recommended_option_id not in option_ids:
            raise ValueError("recommended option must be one of the frozen choices")

    def to_dict(self) -> dict[str, object]:
        return {
            "blocker_id": self.blocker_id,
            "title": self.title,
            "evidence": list(self.evidence),
            "options": [option.to_dict() for option in self.options],
            "recommended_option_id": self.recommended_option_id,
            "required_machine_fields": list(self.required_machine_fields),
            "resolved": False,
        }


EXACT_ERRATA_BLOCKERS: Final = (
    ExactErrataBlocker(
        blocker_id="E1_M3_RUN_GENESIS_SLOT_CARDINALITY",
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
        recommended_option_id="E1_A_EXACTLY_15",
        required_machine_fields=(
            "run_output_slot_count",
            "ordered_run_output_slot_names",
            "m3_run_genesis_schema_id",
        ),
    ),
    ExactErrataBlocker(
        blocker_id="E2_BRIDGE_TOPOLOGY_ORDER",
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
        recommended_option_id="E2_A_REORDER_EXECUTION_BEFORE_SIGNATURES",
        required_machine_fields=(
            "ordered_root_dag_steps",
            "run_id_registration_step",
            "bridge_signature_preimage_fields",
        ),
    ),
    ExactErrataBlocker(
        blocker_id="E3_BRIDGE_ENVELOPE_AND_IDENTITY_BINDING",
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
        recommended_option_id="E3_A_STATEMENT_PLUS_THREE_ENVELOPES",
        required_machine_fields=(
            "bridge_statement_tag",
            "bridge_statement_schema_id",
            "bridge_statement_hash_domain",
            "bridge_envelope_cardinality",
            "bridge_attestation_bundle_binding_location",
        ),
    ),
    ExactErrataBlocker(
        blocker_id="E4_MISSING_HASH_DOMAINS",
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
        recommended_option_id="E4_A_DOMAIN_SEPARATED_CONTENT_HASH",
        required_machine_fields=(
            "signed_manifest_envelope_root_formula",
            "signed_manifest_envelope_hash_domain",
            "m3_dual_replay_agreement_root_formula",
            "m3_dual_replay_agreement_hash_domain",
        ),
    ),
    ExactErrataBlocker(
        blocker_id="E5_ACTOR_KEY_TRUST_AND_PURPOSE",
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
        recommended_option_id="E5_A_PURPOSE1_IS_CUSTODIAN_IDENTITY",
        required_machine_fields=(
            "actor_key_root_binding_object",
            "pinned_genesis_trust_anchor",
            "custodian_manifest_signer_purpose_id",
            "noncustodian_envelope_epoch_semantics",
        ),
    ),
    ExactErrataBlocker(
        blocker_id="E6_LEDGER_CALCULATOR_ACTOR_BOUNDARY",
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
        recommended_option_id="E6_A_CALCULATORS_INSIDE_CUSTODIAN_BOUNDARY",
        required_machine_fields=(
            "calculator_actor_boundary",
            "fd3_access_event_required",
            "gate16_required_ledger_count",
            "gate16_required_head_rule",
        ),
    ),
    ExactErrataBlocker(
        blocker_id="E7_PARENT_ABSENCE_AUDIT_WIRE",
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
        recommended_option_id="E7_A_VERSIONED_AUDIT_BUNDLE",
        required_machine_fields=(
            "audited_path_blob_tag_registry_entry",
            "audited_path_set_row_schema",
            "audited_history_row_schema",
            "audit_tree_ordering",
            "audit_root_binding_field",
            "legacy_parent_payload_source_ids",
        ),
    ),
    ExactErrataBlocker(
        blocker_id="E8_ROLE_AND_INITIAL_STATE_ENUMS",
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
        recommended_option_id="E8_A_M3STATE0_NEW_TARGET_ROLE_ENUM",
        required_machine_fields=(
            "m3_run_genesis_initial_state_registry",
            "m3_run_genesis_initial_state_value",
            "target_role_enum_registry",
            "role_id_field_registry_map",
        ),
    ),
    ExactErrataBlocker(
        blocker_id="E9_ROOT_PREIMAGES_INSTANCE_IDS_AND_PATH_ALIAS",
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
        recommended_option_id="E9_A_FORMAL_PREIMAGE_AND_ALIAS_REGISTRY",
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
    ExactErrataBlocker(
        blocker_id="E10_OPAQUE_ID_REGISTRY_EVIDENCE",
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
        recommended_option_id="E10_A_FORMAL_APPEND_ONLY_REGISTRY_ROOT",
        required_machine_fields=(
            "opaque_id_registry_record_schema",
            "opaque_id_registry_ordering",
            "opaque_id_registry_root_formula",
            "opaque_id_registry_binding_location",
            "duplicate_scope_verification_rule",
        ),
    ),
    ExactErrataBlocker(
        blocker_id="E11_NULL_WITNESS_BINDING_FIELD",
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
        recommended_option_id="E11_A_USE_TARGET_SPEC_AND_BUNDLE_FIELDS",
        required_machine_fields=(
            "authoritative_witness_binding_fields",
            "dsl_role_binding_schema_change_required",
        ),
    ),
    ExactErrataBlocker(
        blocker_id="E12_CUSTODIAN_ENVELOPE_COVERAGE",
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
        recommended_option_id="E12_A_SIGN_FOUR_CUSTODIAN_OBJECTS",
        required_machine_fields=(
            "custodian_signed_object_tags",
            "custodian_signature_domain_by_tag",
            "custodian_envelope_root_binding_locations",
            "required_signature_count_by_tag",
        ),
    ),
)

EXACT_ERRATA_BY_ID: Final = MappingProxyType(
    {blocker.blocker_id: blocker for blocker in EXACT_ERRATA_BLOCKERS}
)


def external_genesis_preflight_report() -> dict[str, object]:
    """Return a deterministic diagnostic report that cannot authorize genesis."""

    payload: dict[str, object] = {
        "artifact": ARTIFACT_NAME,
        "artifact_kind": ARTIFACT_KIND,
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "child_dsl_id": CHILD_DSL_ID,
        "status": CURRENT_STATUS,
        "child_state": CURRENT_CHILD_STATE,
        "m3_gates_satisfied": M3_GATES_SATISFIED,
        "m3_gates_total": M3_GATES_TOTAL,
        "m3_entry_allowed": False,
        "m3_entry_qualified": False,
        "m3_run_started": False,
        "external_genesis_start_allowed": False,
        "exact_errata_required": True,
        "exact_errata_blocker_count": len(EXACT_ERRATA_BLOCKERS),
        "exact_errata_blockers": [
            blocker.to_dict() for blocker in EXACT_ERRATA_BLOCKERS
        ],
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
            "public_typed_row_and_root_replay",
            "synthetic_split_and_signature_fault_injection",
            "read_only_storage_fd3_marker_policy_validation",
        ],
        "claim_boundary": (
            "This is a diagnostic exact-errata preflight. It performs no CSPRNG "
            "call, marker creation, seed/key generation, signature, authoritative "
            "formal-root generation, M3 gate advancement, or state transition."
        ),
    }
    payload["diagnostic_report_id"] = stable_hash(
        payload,
        prefix="phase3_m25_external_preflight_",
    )
    return payload


def validate_external_genesis_preflight_report(report: Mapping[str, object]) -> None:
    """Reject any tampering that weakens the mandatory pre-CSPRNG stop."""

    if not isinstance(report, Mapping):
        raise TypeError("external genesis preflight report must be a mapping")
    expected = external_genesis_preflight_report()
    if not _json_type_strict_equal(dict(report), expected):
        _fail(
            FAIL_M25_EXACT_ERRATA_REQUIRED,
            "external genesis preflight differs from the frozen blocked report",
        )


def assert_external_genesis_start_allowed(
    report: Mapping[str, object] | None = None,
) -> NoReturn:
    """Fail before any CSPRNG call or marker creation can be attempted.

    There is intentionally no override flag.  A future normative amendment
    must replace this function/module version after all E1--E12 decisions are
    frozen and independently implemented.
    """

    candidate = external_genesis_preflight_report() if report is None else report
    validate_external_genesis_preflight_report(candidate)
    unresolved = ",".join(blocker.blocker_id for blocker in EXACT_ERRATA_BLOCKERS)
    _fail(
        FAIL_M25_EXACT_ERRATA_REQUIRED,
        f"external genesis is blocked before CSPRNG/marker by {unresolved}",
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
    an eventual authoritative calculator output needs an exact allowlisted
    schema after E3/E5/E6/E9 are resolved.
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
    "EXACT_ERRATA_BLOCKERS",
    "EXACT_ERRATA_BY_ID",
    "ExactErrataBlocker",
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
    "M3_GATES_SATISFIED",
    "M3_GATES_TOTAL",
    "MACHINE_FREEZE_ID",
    "MarkerSnapshot",
    "assert_external_genesis_start_allowed",
    "assert_marker_does_not_require_external_recovery",
    "assert_public_payload_contains_no_secret_fields",
    "assert_seed_instantiation_marker_absent",
    "external_genesis_preflight_report",
    "validate_calculator_process_result",
    "validate_commit_b_changed_paths",
    "validate_distinct_actor_key_ids",
    "validate_external_genesis_preflight_report",
    "validate_marker_snapshot",
    "validate_secret_fd_number",
    "validate_secret_fd_payload",
    "validate_secret_file",
    "validate_secret_state_directory",
]
