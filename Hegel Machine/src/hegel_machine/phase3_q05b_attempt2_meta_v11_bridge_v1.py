"""Pure exact-wire bridge for one authorized Q0.5b actual attempt-2.

The module performs no filesystem mutation, Docker/Cargo invocation, network
access, or Q0.5b execution.  It validates one real Meta V11 report by calling
the frozen public thirty-object verifier, constructs the destination-scoped
admission report, and constructs content-addressed ledger/link records.  The
separate runner owns live Git/source checks, no-replace ledger I/O, and the
single call to the unchanged Q0.5b ``run_actual_v1`` entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.machinery
import json
import math
import re
from pathlib import Path
import stat
import sys
from typing import Any, Final, Mapping, NoReturn


CONTRACT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-meta-v11-bridge-contract/1"
)
CONTRACT_ID: Final = "hegel-phase3a-q05b-attempt2-meta-v11-bridge-v1"
INPUT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-meta-v11-bridge-input/1"
)
REPORT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-meta-v11-bridge-report/1"
)
EXECUTION_LINK_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-meta-v11-bridge-execution-link/1"
)
FAILURE_LINK_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-meta-v11-bridge-failure-link/1"
)
ERROR_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-meta-v11-bridge-error/1"
)
CONSUMED_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-meta-v11-bridge-consumed/1"
)
ADOPTION_RECEIPT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-destination-adoption-receipt/1"
)

FROZEN_CONTRACT_DIGEST: Final = (
    "sha256:e36ecb243a7f36d24c5a29f4fd6beb913c195848350f845ce615d8b0304535ed"
)
META_V11_CONTRACT_DIGEST: Final = (
    "sha256:9a52bb7ea3f4ce0f0cb16c5a5a296d284a85f4bfbdb2ee671e8c865bb2d3d493"
)
META_V11_QUALIFIED: Final = "QUALIFIED_FRESH_POST_TRANSITION_EVALUATOR_EPOCH"
META_V11_VERIFIED_QUALIFIED: Final = (
    "VERIFIED_QUALIFIED_FRESH_POST_TRANSITION_EVALUATOR_EPOCH"
)
BRIDGE_REFUSED: Final = "META_V11_REPORT_NOT_QUALIFIED_BRIDGE_REFUSED"
BRIDGE_ADMITTED: Final = (
    "META_V11_QUALIFIED_BRIDGE_ADMITTED_ATTEMPT2_PENDING_CONSUMPTION"
)
MISSING_REAL_V11_REPORT: Final = "MISSING_REAL_V11_REPORT"
ALREADY_CONSUMED: Final = "ATTEMPT2_AUTHORIZATION_ALREADY_CONSUMED"
ATTEMPT2_FAILED: Final = "ATTEMPT2_FAILED_CLOSED"
ATTEMPT2_LINKED: Final = "ATTEMPT2_QUALIFIED_LINKED"
OWNER_AUTHORIZATION_DIGEST: Final = (
    "sha256:9623deef9716248e19ec2eb38a4be6adb8bab2d0332666add76ac916d268e722"
)
GENESIS_DIGEST: Final = "sha256:" + "0" * 64

DOMAINS: Final = {
    "adoption_receipt": b"HEGEL/Q05B/ATTEMPT2/ADOPTION_RECEIPT/V1\x00",
    "audit_event": b"HEGEL/Q05B/ATTEMPT2/AUDIT_EVENT/V1\x00",
    "contract": b"HEGEL/Q05B/ATTEMPT2/META_V11_BRIDGE/CONTRACT/V1\x00",
    "execution_link": b"HEGEL/Q05B/ATTEMPT2/EXECUTION_LINK/V1\x00",
    "failure_link": b"HEGEL/Q05B/ATTEMPT2/FAILURE_LINK/V1\x00",
    "input": b"HEGEL/Q05B/ATTEMPT2/INPUT/V1\x00",
    "ledger_consumed": b"HEGEL/Q05B/ATTEMPT2/LEDGER_CONSUMED/V1\x00",
    "owner_authorization": b"HEGEL/Q05B/ATTEMPT2/OWNER_AUTHORIZATION/V1\x00",
    "report": b"HEGEL/Q05B/ATTEMPT2/REPORT/V1\x00",
}

EXACT_FIVE_PATHS: Final = [
    "Hegel Machine/src/hegel_machine/phase3_q05b_attempt2_meta_v11_bridge_v1.py",
    "Hegel Machine/config/phase3_q05b_attempt2_meta_v11_bridge_v1.json",
    "Hegel Machine/tools/phase3_q05b_attempt2_meta_v11_bridge_v1.py",
    "Hegel Machine/tests/test_phase3_q05b_attempt2_meta_v11_bridge_v1.py",
    "Hegel Machine/docs/Hegel_Machine_Phase3A_Q05b_Attempt2_Meta_V11_Bridge_v1.md",
]
LEDGER_ROOT: Final = (
    "/home/erzhu419/.local/state/hegel-machine/q05b-attempt2-ledger-v1"
)
LEDGER_NAMES: Final = {
    "bridge": "attempt-2.bridge-report.json",
    "consumed": "attempt-2.consumed.json",
    "execution": "attempt-2.execution-link.json",
    "failure": "attempt-2.failure-link.json",
}
PROJECT_ROOT: Final = (
    "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine"
)
Q05B_ARTIFACT_PATH: Final = PROJECT_ROOT + "/artifacts/phase3_q05b_dual_qualification_v1.json"
CARGO_CACHE_SOURCE: Final = "/home/erzhu419/.local/state/hegel-machine/rust-cargo-cache"
RUN_ACTUAL_ENTRYPOINT: Final = "tools/phase3_q05b_dual_qualification_v1.py::run_actual_v1"
META_SOURCE_ONLY_PYCACHE_PREFIX: Final = (
    "/dev/null/hegel-q05b-attempt2-meta-v11-source-only"
)
META_SOURCE_BINDING_ROOT: Final = (
    "/home/erzhu419/.local/state/kg-op/meta-prior-theory-operations-worktree/"
    "v1/SC-OLH-KG"
)
META_PERFORMANCE_ROOT: Final = META_SOURCE_BINDING_ROOT + "/performance"

OWNER_AUTHORIZATION_BODY: Final = {
    "schema_version": "hegel-phase3a-q05b-attempt2-owner-authorization/1",
    "authority_role": "PROJECT_OWNER",
    "selected_antecedent_text": (
        "下一步若要进入 Hegel Q0.5b actual，需要你另行明确授权 attempt-2，并先补 "
        "digest-bound external authority/adoption bridge"
    ),
    "confirmation_text": "确认授权",
    "confirmation_semantics": "EXACT_CONFIRMATION_OF_SELECTED_ANTECEDENT",
    "authorized_attempt_ordinal": 2,
    "maximum_actual_attempts": 1,
    "authorized_actions": [
        "CONSTRUCT_DIGEST_BOUND_META_V11_TO_HEGEL_EXTERNAL_AUTHORITY_ADOPTION_BRIDGE",
        "EXECUTE_HEGEL_Q05B_ACTUAL_ATTEMPT_2_ONLY_AFTER_BRIDGE_AND_ALL_EXISTING_RUNTIME_GATES",
    ],
    "forbidden_actions": [
        "ATTEMPT_3_OR_RETRY",
        "NETWORK_ACCESS",
        "Q1_EXECUTION",
        "ACTIVE_TRANSITION",
        "UNBOUND_META_ADOPTION",
        "CURRENT_POINTER_WRITE_OUTSIDE_EXACT_BRIDGE_RECEIPT",
    ],
    "authorization_is_execution_evidence": False,
    "authorization_is_external_signature": False,
}
OWNER_AUTHORIZATION: Final = {
    **OWNER_AUTHORIZATION_BODY,
    "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
}

META_EXACT_FIVE_RAW_SHA256: Final = {
    "performance/shadow_interval_multi_q_post_transition_qualification.py": (
        "d9ef2a166abb78f04cf41a5a4f4ae48a7fa3227817cd42dbc11272dc68bdeb36"
    ),
    "performance/manifests/shadow_interval_multi_q_post_transition_qualification_v1.json": (
        "34c63a1ee3825543e3db612c9621af9af6441c6f58685423307164f05d4ce554"
    ),
    "runners/run_shadow_interval_multi_q_post_transition_qualification.py": (
        "29b14485d2fa7d268af0723ff7b36bd6bd90f1ecab53e0c1bc6f3624181416eb"
    ),
    "tests/test_shadow_interval_multi_q_post_transition_qualification.py": (
        "2046e500a37d06bab00bbaa351a6ce942dc109faf9ab9d7165f83097a8a9f45d"
    ),
    "docs/shadow_interval_multi_q_post_transition_qualification_v1.md": (
        "660488c35fc443759d2c3387030d132ee43ec71adfb3e2906fc6f1becc968d6c"
    ),
}
META_SOURCE_BINDING: Final = {
    "repo_root": (
        "/home/erzhu419/.local/state/kg-op/meta-prior-theory-operations-worktree/"
        "v1/SC-OLH-KG"
    ),
    "git_commit": "fc9e781ce29ca0fef0f37ed97d07d5037adfdfff",
    "git_tree": "932ea7acbea5e8fbf36ddbd11c648611e1fcdae0",
    "git_clean": True,
    "v11_contract_digest": META_V11_CONTRACT_DIGEST,
    "v11_public_verifier": (
        "verify_shadow_interval_multi_q_post_transition_qualification"
    ),
    "exact_five_raw_sha256": META_EXACT_FIVE_RAW_SHA256,
}

ATTEMPT_HISTORY_BINDING: Final = {
    "attempt1_ordinal": 1,
    "attempt1_source_commit": "07d91847ceff1dc1692b05084b17718409155392",
    "attempt1_consumed": True,
    "attempt1_result": "FAILED_CLOSED_AT_STAGE_07",
    "attempt1_last_completed_stage": 6,
    "attempt1_failed_stage": 7,
    "attempt1_exit_code": 1,
    "attempt1_artifact_absent": True,
    "attempt1_q05b_predicate_count": 0,
    "attempt1_q1_state": "NOT_RUN",
    "attempt1_failure_manifest_digest": (
        "805aca208885717710061ac8fa5bd4f4605c8da6fcfde60b7bce22c1915bd0bc"
    ),
    "attempt1_errata_path": (
        "Hegel Machine/docs/"
        "Hegel_Machine_Phase3A_Q05b_Actual_Attempt_1_Failure_Errata_v1.md"
    ),
    "attempt1_errata_raw_sha256": (
        "03e299ae6d0e6354c9afc259137a290009ff1b52765674ba310387204bcd0894"
    ),
    "attempt1_errata_git_blob_oid": "6ba3adb22c77af28e64bc56bf7a63180d75d1609",
    "repair_commit": "d55c017fb5ae16a83ff57758cdb3e9b57c29afb9",
}

HEGEL_UNCHANGED_RAW_SHA256: Final = {
    "config/phase3_q05b_dual_isolation_v1.json": (
        "d3a1189cce5a40c5be76f0c5ff1c4e375781a803c11258361fc131a70a7c7242"
    ),
    "config/phase3_q05b_node3_dual_projection_qualification_v1.json": (
        "29fb2c0d8a6bd657212d1799c4e7a751bf42d3937ff865223e593e46f96d14ab"
    ),
    "docs/Hegel_Machine_Phase3A_Q05b_Actual_Attempt_1_Failure_Errata_v1.md": (
        "03e299ae6d0e6354c9afc259137a290009ff1b52765674ba310387204bcd0894"
    ),
    "src/hegel_machine/phase3_q05b_actual_admission_v1.py": (
        "d1de0335dee62dbc737e5bd64410de392632648aa92aeabdb4fa078551e9b216"
    ),
    "src/hegel_machine/phase3_q05b_actual_artifact_v1.py": (
        "75dc2a784280cd59e5bde201b9d9aef78130f6713be59c833864297ab9b36837"
    ),
    "tools/phase3_q05b_dual_qualification_v1.py": (
        "af406bba2e4d11f8f59c1d7f91d3f84c887d1e90d10cc1749baf2a8610a7c5d2"
    ),
}

NONCLAIMS: Final = [
    "no_fabricated_or_synthetic_meta_v11_report",
    "missing_real_v11_report_produces_no_bridge_report",
    "nonqualified_meta_v11_produces_no_adoption_or_attempt_admission",
    "owner_confirmation_is_not_a_cryptographic_signature",
    "owner_confirmation_is_not_execution_evidence",
    "destination_scoped_input_adoption_only",
    "no_meta_global_adoption",
    "no_meta_promotion",
    "no_meta_current_pointer_write",
    "no_unbounded_h_t_to_h_t_plus_1_acceptance",
    "no_hegel_q1_execution",
    "no_active_transition",
    "no_network_access",
    "no_attempt3_retry_or_resume_after_consumption",
    "bridge_preflight_performs_no_write_docker_cargo_or_actual",
    "ledger_consumption_is_not_q05b_qualification",
    "q05b_success_remains_non_q1_qualification_only",
    "no_prior_meta_slice_or_existing_q05b_file_modification",
    "no_operations_research_baseline_claim_or_paper_change",
    "report_digest_is_not_a_signature",
    "input_artifacts_require_independent_raw_file_metadata",
    "exact_five_additive_slice_only",
    "pure_builders_and_standalone_validators_are_untrusted_structural_constructors",
    "only_full_33_object_verifier_plus_runner_live_gates_authorize_publication_or_actual",
]

AUTHORITY_CAPABILITIES: Final = {
    "scope": "EXACT_META_V11_QUALIFIED_CHILD_TO_HEGEL_Q05B_ATTEMPT2_ONLY",
    "meta_v11_exact_public_replay": True,
    "destination_scoped_adoption_eligibility": True,
    "destination_scoped_adoption": True,
    "hegel_q05b_attempt2_admission": True,
    "one_attempt_ledger_consumption": True,
    "existing_q05b_actual_execution": True,
    "execution_linkage": True,
    "full_33_object_verifier_plus_live_runner_gates_required": True,
    "pure_structural_constructors_authorize_execution": False,
    "meta_global_adoption": False,
    "meta_promotion": False,
    "meta_current_pointer_write": False,
    "hegel_q1_execution": False,
    "active_transition": False,
    "network_access": False,
    "attempt3_or_retry": False,
}

V11_POSITIONAL_NAMES: Final = (
    "competition_input", "competition_contract", "competition_report",
    "transition_contract", "transition_report", "qualification_input",
    "qualification_contract", "qualification_report", "review_contract",
    "review_report", "probe_input", "probe_contract", "probe_report",
    "restriction_input", "restriction_contract", "restriction_report",
    "adjudication_input", "adjudication_contract", "adjudication_report",
    "adapter_input", "adapter_contract", "adapter_report",
    "interval_competition_input", "interval_competition_contract",
    "interval_competition_report", "interval_transition_contract",
    "interval_transition_report", "post_transition_qualification_input",
    "post_transition_qualification_contract",
    "post_transition_qualification_report",
)
BRIDGE_INPUT_NAMES: Final = V11_POSITIONAL_NAMES + (
    "attempt2_bridge_input", "attempt2_bridge_contract"
)
EXPECTED_DIGEST_NAMES: Final = (
    "expected_competition_contract_digest",
    "expected_competition_report_digest",
    "expected_transition_contract_digest", "expected_transition_report_digest",
    "expected_qualification_input_digest",
    "expected_qualification_contract_digest",
    "expected_qualification_report_digest", "expected_review_contract_digest",
    "expected_review_report_digest", "expected_probe_input_digest",
    "expected_probe_contract_digest", "expected_probe_report_digest",
    "expected_restriction_input_digest", "expected_restriction_contract_digest",
    "expected_restriction_report_digest", "expected_adjudication_input_digest",
    "expected_adjudication_contract_digest",
    "expected_adjudication_report_digest", "expected_adapter_input_digest",
    "expected_adapter_contract_digest", "expected_adapter_report_digest",
    "expected_interval_competition_input_digest",
    "expected_interval_competition_contract_digest",
    "expected_interval_competition_report_digest",
    "expected_interval_transition_contract_digest",
    "expected_interval_transition_report_digest",
    "expected_post_transition_qualification_input_digest",
    "expected_post_transition_qualification_contract_digest",
    "expected_post_transition_qualification_report_digest",
)

CONTRACT_FIELDS: Final = frozenset({
    "schema_version", "contract_id", "input_schema_version",
    "report_schema_version", "execution_link_schema_version",
    "failure_link_schema_version", "error_schema_version",
    "canonicalization_policy", "hash_domain_registry", "exact_five_paths",
    "owner_authorization_policy", "meta_source_binding",
    "meta_v11_replay_policy", "attempt_history_binding", "hegel_source_policy",
    "durable_ledger_policy", "actual_execution_policy", "execution_link_policy",
    "disposition_registry", "authority_boundary", "nonclaims",
})
INPUT_FIELDS: Final = frozenset({
    "schema_version", "owner_authorization", "meta_source_binding",
    "meta_v11_expected_digests", "attempt_history_binding",
    "hegel_source_binding", "durable_ledger_binding", "execution_request",
})
REPORT_FIELDS: Final = frozenset({
    "schema_version", "contract_id", "contract_digest", "bridge_input_digest",
    "disposition", "owner_authorization_binding", "meta_source_binding",
    "meta_v11_replay_receipt", "meta_v11_qualification_binding",
    "external_authority_decision", "destination_adoption_receipt",
    "attempt_history_binding", "hegel_destination_binding",
    "attempt2_admission", "durable_ledger_plan", "authority_boundary",
    "nonclaims", "input_artifacts", "audit_events", "audit_head",
    "report_digest",
})
ARTIFACT_ROW_FIELDS: Final = frozenset({"path", "bytes", "sha256"})
V11_RECEIPT_FIELDS: Final = frozenset({
    "status", "disposition", "report_digest", "contract_digest",
    "qualification_input_digest", "source_transition_report_digest",
    "qualification_id", "parent_theory_state_digest", "child_theory_state_digest",
    "operation_kind", "transition_kind", "holdout_evaluated",
    "stress_confirmation_evaluated", "qualified", "adoption_eligibility",
    "adoption_status", "promotion_status", "current_status",
})
META_SOURCE_ONLY_RECEIPT_FIELDS: Final = frozenset({
    "policy", "preloaded_performance_modules_absent",
    "dont_write_bytecode_active", "pycache_prefix", "namespace_package_path",
    "loaded_source_module_origins", "in_tree_bytecode_loaded",
    "newly_loaded_performance_modules_removed_after_replay",
    "interpreter_import_state_restored",
    "exact_meta_root_temporarily_inserted_in_sys_path",
    "standard_import_machinery_only", "source_file_loader_only_for_project",
    "path_hooks_and_importer_cache_restored",
    "sys_path_and_meta_path_restored",
    "live_git_commit_tree_clean_raw_hash_checks_performed",
})


class Phase3Q05BAttempt2MetaV11BridgeValidationError(ValueError):
    """Stable fail-closed validation error for the additive bridge."""

    def __init__(self, detail: str, *, code: str = "INVALID_ATTEMPT2_META_V11_BRIDGE") -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class MissingRealV11ReportError(Phase3Q05BAttempt2MetaV11BridgeValidationError):
    """Raised before import/replay when the required real report is missing."""

    def __init__(self) -> None:
        super().__init__(
            "required real post-transition qualification report is absent",
            code=MISSING_REAL_V11_REPORT,
        )


def _fail(detail: str) -> NoReturn:
    raise Phase3Q05BAttempt2MetaV11BridgeValidationError(detail)


def canonical_json_bytes_v1(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        _fail(f"canonical JSON differs: {error}")


def _copy(value: object) -> Any:
    try:
        return json.loads(canonical_json_bytes_v1(value).decode("ascii"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"value is not canonical finite JSON: {error}")


def _domain_digest(domain: bytes, value: object) -> str:
    return "sha256:" + hashlib.sha256(domain + canonical_json_bytes_v1(value)).hexdigest()


def _raw_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str] | set[str], label: str) -> None:
    if set(value) != set(expected):
        _fail(
            f"{label} keys differ; missing={sorted(set(expected) - set(value))}, "
            f"extra={sorted(set(value) - set(expected), key=str)}"
        )


def _text(value: object, label: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        _fail(f"{label} must be bounded non-empty text")
    return value


def _digest(value: object, label: str) -> str:
    text = _text(value, label, maximum=71)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        _fail(f"{label} must be sha256:<64lowerhex>")
    return text


def _hex(value: object, size: int, label: str) -> str:
    text = _text(value, label, maximum=size)
    if len(text) != size or re.fullmatch(r"[0-9a-f]+", text) is None:
        _fail(f"{label} must be {size} lowercase hexadecimal characters")
    return text


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a non-negative integer")
    return value


def _bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be boolean")
    return value


def _validate_owner_authorization(value: object) -> dict[str, Any]:
    authorization = _copy(_mapping(value, "owner_authorization"))
    _exact_keys(authorization, frozenset(OWNER_AUTHORIZATION), "owner_authorization")
    if authorization != OWNER_AUTHORIZATION:
        _fail("owner authorization differs from the exact contextual confirmation")
    body = {key: authorization[key] for key in OWNER_AUTHORIZATION_BODY}
    if _domain_digest(DOMAINS["owner_authorization"], body) != OWNER_AUTHORIZATION_DIGEST:
        _fail("owner authorization domain digest differs")
    return authorization


def validate_phase3_q05b_attempt2_meta_v11_bridge_contract_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _copy(_mapping(value, "attempt2_bridge_contract"))
    _exact_keys(contract, CONTRACT_FIELDS, "attempt2_bridge_contract")
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("contract_id") != CONTRACT_ID
        or contract.get("input_schema_version") != INPUT_SCHEMA_VERSION
        or contract.get("report_schema_version") != REPORT_SCHEMA_VERSION
        or contract.get("execution_link_schema_version") != EXECUTION_LINK_SCHEMA_VERSION
        or contract.get("failure_link_schema_version") != FAILURE_LINK_SCHEMA_VERSION
        or contract.get("error_schema_version") != ERROR_SCHEMA_VERSION
        or contract.get("exact_five_paths") != EXACT_FIVE_PATHS
        or contract.get("owner_authorization_policy") != OWNER_AUTHORIZATION
        or contract.get("meta_source_binding") != META_SOURCE_BINDING
        or contract.get("attempt_history_binding") != ATTEMPT_HISTORY_BINDING
        or contract.get("authority_boundary") != AUTHORITY_CAPABILITIES
        or contract.get("nonclaims") != NONCLAIMS
    ):
        _fail("attempt2 bridge contract differs from the frozen contract")
    registry = _mapping(contract.get("hash_domain_registry"), "hash_domain_registry")
    if registry != {key: value.decode("ascii") for key, value in DOMAINS.items()}:
        _fail("hash domain registry differs")
    contract_digest = _domain_digest(DOMAINS["contract"], contract)
    if contract_digest != FROZEN_CONTRACT_DIGEST:
        _fail("attempt2 bridge contract digest differs from frozen V1")
    return contract


HEGEL_SOURCE_FIELDS: Final = frozenset({
    "project_root", "source_commit", "source_tree", "clean", "baseline_commit",
    "baseline_tree", "repair_is_ancestor", "exact_five_diff_paths",
    "exact_five_raw_sha256", "existing_actual_tool_raw_sha256",
    "existing_admission_core_raw_sha256", "existing_artifact_core_raw_sha256",
    "isolation_config_raw_sha256", "qualification_config_raw_sha256",
    "existing_unchanged_raw_sha256",
})
DURABLE_LEDGER_BINDING_FIELDS: Final = frozenset({"root", "expected_initial_state"})
EXECUTION_REQUEST_FIELDS: Final = frozenset({
    "mode", "attempt_ordinal", "maximum_actual_invocations", "source_commit",
    "artifact_path", "cargo_cache_source", "run_actual_entrypoint",
})


def _validate_sha_map(value: object, keys: set[str], label: str) -> dict[str, str]:
    mapping = _copy(_mapping(value, label))
    _exact_keys(mapping, keys, label)
    for key, item in mapping.items():
        _hex(item, 64, f"{label}.{key}")
    return mapping


def _validate_hegel_source_binding(
    value: object, contract: Mapping[str, Any]
) -> dict[str, Any]:
    binding = _copy(_mapping(value, "hegel_source_binding"))
    _exact_keys(binding, HEGEL_SOURCE_FIELDS, "hegel_source_binding")
    policy = _mapping(contract["hegel_source_policy"], "hegel_source_policy")
    if (
        binding.get("project_root") != PROJECT_ROOT
        or binding.get("baseline_commit") != policy.get("baseline_commit")
        or binding.get("baseline_tree") != policy.get("baseline_tree")
        or binding.get("clean") is not True
        or binding.get("repair_is_ancestor") is not True
        or binding.get("exact_five_diff_paths") != EXACT_FIVE_PATHS
    ):
        _fail("Hegel source binding differs from the frozen direct-child policy")
    _hex(binding.get("source_commit"), 40, "hegel_source_binding.source_commit")
    _hex(binding.get("source_tree"), 40, "hegel_source_binding.source_tree")
    exact_five = _validate_sha_map(
        binding.get("exact_five_raw_sha256"), set(EXACT_FIVE_PATHS),
        "hegel_source_binding.exact_five_raw_sha256",
    )
    if len(set(exact_five.values())) != len(exact_five):
        _fail("exact-five source file hashes must be distinct")
    unchanged = _validate_sha_map(
        binding.get("existing_unchanged_raw_sha256"),
        set(HEGEL_UNCHANGED_RAW_SHA256),
        "hegel_source_binding.existing_unchanged_raw_sha256",
    )
    if unchanged != HEGEL_UNCHANGED_RAW_SHA256:
        _fail("Hegel unchanged source path-to-hash binding differs")
    if policy.get("existing_unchanged_raw_sha256") != HEGEL_UNCHANGED_RAW_SHA256:
        _fail("contract unchanged source path-to-hash binding differs")
    for key in (
        "existing_actual_tool_raw_sha256", "existing_admission_core_raw_sha256",
        "existing_artifact_core_raw_sha256", "isolation_config_raw_sha256",
        "qualification_config_raw_sha256",
    ):
        expected = policy.get(key)
        if binding.get(key) != expected:
            _fail(f"hegel_source_binding.{key} differs from unchanged source")
        _hex(binding[key], 64, f"hegel_source_binding.{key}")
    return binding


def _validate_input(
    value: object, contract: Mapping[str, Any]
) -> dict[str, Any]:
    bridge_input = _copy(_mapping(value, "attempt2_bridge_input"))
    _exact_keys(bridge_input, INPUT_FIELDS, "attempt2_bridge_input")
    if bridge_input.get("schema_version") != INPUT_SCHEMA_VERSION:
        _fail("attempt2 bridge input schema differs")
    _validate_owner_authorization(bridge_input.get("owner_authorization"))
    if bridge_input.get("meta_source_binding") != META_SOURCE_BINDING:
        _fail("Meta source binding differs from frozen V11 commit bytes")
    expected_digests = _copy(
        _mapping(
            bridge_input.get("meta_v11_expected_digests"),
            "meta_v11_expected_digests",
        )
    )
    _exact_keys(
        expected_digests, set(EXPECTED_DIGEST_NAMES), "meta_v11_expected_digests"
    )
    for key in EXPECTED_DIGEST_NAMES:
        _digest(expected_digests[key], f"meta_v11_expected_digests.{key}")
    if bridge_input.get("attempt_history_binding") != ATTEMPT_HISTORY_BINDING:
        _fail("attempt-1 failure/repair history binding differs")
    hegel = _validate_hegel_source_binding(
        bridge_input.get("hegel_source_binding"), contract
    )
    ledger = _copy(
        _mapping(bridge_input.get("durable_ledger_binding"), "durable_ledger_binding")
    )
    _exact_keys(ledger, DURABLE_LEDGER_BINDING_FIELDS, "durable_ledger_binding")
    if ledger != {
        "root": LEDGER_ROOT,
        "expected_initial_state": "OPEN_OR_EXACT_BRIDGE_STAGED_NOT_CONSUMED",
    }:
        _fail("durable ledger binding differs")
    request = _copy(
        _mapping(bridge_input.get("execution_request"), "execution_request")
    )
    _exact_keys(request, EXECUTION_REQUEST_FIELDS, "execution_request")
    if request != {
        "mode": "HEGEL_Q05B_ACTUAL_ATTEMPT2",
        "attempt_ordinal": 2,
        "maximum_actual_invocations": 1,
        "source_commit": hegel["source_commit"],
        "artifact_path": Q05B_ARTIFACT_PATH,
        "cargo_cache_source": CARGO_CACHE_SOURCE,
        "run_actual_entrypoint": RUN_ACTUAL_ENTRYPOINT,
    }:
        _fail("attempt-2 execution request differs")
    return bridge_input


def _validate_input_artifacts(value: object) -> dict[str, dict[str, Any]]:
    artifacts = _copy(_mapping(value, "input_artifacts"))
    _exact_keys(artifacts, set(BRIDGE_INPUT_NAMES), "input_artifacts")
    identities: set[tuple[str, int, str]] = set()
    for label in BRIDGE_INPUT_NAMES:
        row = _mapping(artifacts[label], f"input_artifacts.{label}")
        _exact_keys(row, ARTIFACT_ROW_FIELDS, f"input_artifacts.{label}")
        path = _text(row.get("path"), f"input_artifacts.{label}.path", maximum=8192)
        if not path.startswith("/"):
            _fail(f"input_artifacts.{label}.path must be absolute")
        size = _nonnegative_int(row.get("bytes"), f"input_artifacts.{label}.bytes")
        sha = _hex(row.get("sha256"), 64, f"input_artifacts.{label}.sha256")
        identity = (path, size, sha)
        if identity in identities:
            _fail("independent input artifact rows must be distinct")
        identities.add(identity)
    return artifacts


def _v11_artifact_maps(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    competition = {
        "contract_json": artifacts["competition_contract"],
        "evidence_json": artifacts["competition_input"],
    }
    transition = {
        "competition_contract_json": artifacts["competition_contract"],
        "competition_input_json": artifacts["competition_input"],
        "competition_report_json": artifacts["competition_report"],
        "transition_contract_json": artifacts["transition_contract"],
    }
    qualification = {
        "competition_contract_json": artifacts["competition_contract"],
        "competition_input_json": artifacts["competition_input"],
        "competition_report_json": artifacts["competition_report"],
        "qualification_contract_json": artifacts["qualification_contract"],
        "qualification_input_json": artifacts["qualification_input"],
        "transition_contract_json": artifacts["transition_contract"],
        "transition_report_json": artifacts["transition_report"],
    }
    review = {
        **qualification,
        "qualification_report_json": artifacts["qualification_report"],
        "review_contract_json": artifacts["review_contract"],
    }
    probe = {
        **review,
        "probe_contract_json": artifacts["probe_contract"],
        "probe_input_json": artifacts["probe_input"],
        "review_report_json": artifacts["review_report"],
    }
    restriction = {
        **probe,
        "probe_report_json": artifacts["probe_report"],
        "restriction_contract_json": artifacts["restriction_contract"],
        "restriction_input_json": artifacts["restriction_input"],
    }
    adjudication = {
        **restriction,
        "restriction_report_json": artifacts["restriction_report"],
        "adjudication_contract_json": artifacts["adjudication_contract"],
        "adjudication_input_json": artifacts["adjudication_input"],
    }
    adapter = {
        **adjudication,
        "adjudication_report_json": artifacts["adjudication_report"],
        "adapter_contract_json": artifacts["adapter_contract"],
        "adapter_input_json": artifacts["adapter_input"],
    }
    interval_competition = {
        **adapter,
        "adapter_report_json": artifacts["adapter_report"],
        "interval_competition_contract_json": artifacts[
            "interval_competition_contract"
        ],
        "interval_competition_input_json": artifacts["interval_competition_input"],
    }
    interval_transition = {
        **interval_competition,
        "interval_competition_report_json": artifacts[
            "interval_competition_report"
        ],
        "interval_transition_contract_json": artifacts[
            "interval_transition_contract"
        ],
    }
    post_transition = {
        **interval_transition,
        "interval_transition_report_json": artifacts["interval_transition_report"],
        "post_transition_qualification_contract_json": artifacts[
            "post_transition_qualification_contract"
        ],
        "post_transition_qualification_input_json": artifacts[
            "post_transition_qualification_input"
        ],
    }
    return {
        "expected_competition_input_artifacts": competition,
        "expected_transition_input_artifacts": transition,
        "expected_qualification_input_artifacts": qualification,
        "expected_review_input_artifacts": review,
        "expected_probe_input_artifacts": probe,
        "expected_restriction_input_artifacts": restriction,
        "expected_adjudication_input_artifacts": adjudication,
        "expected_adapter_input_artifacts": adapter,
        "expected_interval_competition_input_artifacts": interval_competition,
        "expected_interval_transition_input_artifacts": interval_transition,
        "expected_post_transition_qualification_input_artifacts": post_transition,
    }


def _public_v11_replay(
    positional: tuple[object, ...],
    expected_digests: Mapping[str, str],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(positional) != 30:
        _fail("Meta V11 public verifier requires exactly thirty positional objects")
    report = positional[-1]
    if not isinstance(report, Mapping) or not report:
        raise MissingRealV11ReportError()
    preloaded = sorted(
        name for name in sys.modules
        if name == "performance" or name.startswith("performance.")
    )
    if preloaded:
        _fail("preloaded performance modules forbid source-only Meta V11 replay")
    previous_dont_write = sys.dont_write_bytecode
    previous_pycache_prefix = sys.pycache_prefix
    previous_sys_path = list(sys.path)
    previous_meta_path = list(sys.meta_path)
    previous_path_hooks = list(sys.path_hooks)
    previous_path_importer_cache = sys.path_importer_cache
    module = None
    receipt: dict[str, Any] | None = None
    loaded_origins: dict[str, str] = {}
    try:
        sys.dont_write_bytecode = True
        sys.pycache_prefix = META_SOURCE_ONLY_PYCACHE_PREFIX
        sys.path[:] = [META_SOURCE_BINDING["repo_root"]] + [
            item for item in previous_sys_path
            if item != META_SOURCE_BINDING["repo_root"]
        ]
        sys.meta_path[:] = [
            importlib.machinery.BuiltinImporter,
            importlib.machinery.FrozenImporter,
            importlib.machinery.PathFinder,
        ]
        sys.path_hooks[:] = [
            importlib.machinery.FileFinder.path_hook(
                (
                    importlib.machinery.SourceFileLoader,
                    importlib.machinery.SOURCE_SUFFIXES,
                ),
                (
                    importlib.machinery.ExtensionFileLoader,
                    importlib.machinery.EXTENSION_SUFFIXES,
                ),
            )
        ]
        sys.path_importer_cache = {}
        try:
            namespace = importlib.import_module("performance")
        except (ImportError, AttributeError) as error:
            _fail(f"frozen Meta performance namespace is unavailable: {error}")
        namespace_paths = sorted(
            str(Path(path).resolve())
            for path in list(getattr(namespace, "__path__", ()))
        )
        if namespace_paths != [META_PERFORMANCE_ROOT]:
            _fail("Meta performance namespace path differs from frozen root")
        try:
            module = importlib.import_module(
                "performance.shadow_interval_multi_q_post_transition_qualification"
            )
        except (ImportError, AttributeError) as error:
            _fail(f"frozen Meta V11 public verifier is unavailable: {error}")
        for name, loaded in sorted(sys.modules.items()):
            if not name.startswith("performance."):
                continue
            origin = getattr(loaded, "__file__", None)
            if type(origin) is not str:
                _fail(f"loaded Meta project module {name} lacks a source origin")
            resolved = Path(origin).resolve()
            expected = Path(META_SOURCE_BINDING["repo_root"]) / (
                name.replace(".", "/") + ".py"
            )
            if (
                resolved != expected
                or resolved.suffix != ".py"
                or "__pycache__" in resolved.parts
                or not isinstance(
                    getattr(loaded, "__loader__", None),
                    importlib.machinery.SourceFileLoader,
                )
            ):
                _fail(f"loaded Meta project module {name} is not exact-root source")
            loaded_origins[name] = str(resolved)
        if (
            getattr(module, "FROZEN_CONTRACT_DIGEST", None)
            != META_V11_CONTRACT_DIGEST
            or getattr(module, "QUALIFIED", None) != META_V11_QUALIFIED
        ):
            _fail("imported Meta V11 constants differ from the frozen source")
        verifier = getattr(
            module,
            "verify_shadow_interval_multi_q_post_transition_qualification",
            None,
        )
        if (
            not callable(verifier)
            or getattr(verifier, "__module__", None) != module.__name__
        ):
            _fail("Meta V11 public verifier callable identity differs")
        kwargs: dict[str, Any] = dict(expected_digests)
        kwargs.update(_v11_artifact_maps(artifacts))
        try:
            receipt = verifier(*positional, **kwargs)
        except Exception as error:
            if (
                error.__class__.__name__
                != "ShadowIntervalMultiQPostTransitionQualificationValidationError"
                or error.__class__.__module__ != module.__name__
            ):
                raise
            _fail(f"Meta V11 exact public replay rejected: {error}")
        for name, loaded in sorted(sys.modules.items()):
            if not name.startswith("performance.") or name in loaded_origins:
                continue
            origin = getattr(loaded, "__file__", None)
            expected = Path(META_SOURCE_BINDING_ROOT) / (
                name.replace(".", "/") + ".py"
            )
            if (
                type(origin) is not str
                or Path(origin).resolve() != expected
                or not isinstance(
                    getattr(loaded, "__loader__", None),
                    importlib.machinery.SourceFileLoader,
                )
            ):
                _fail(f"lazily loaded Meta project module {name} is not exact source")
            loaded_origins[name] = str(expected)
    finally:
        for name in tuple(sys.modules):
            if name == "performance" or name.startswith("performance."):
                sys.modules.pop(name, None)
        sys.dont_write_bytecode = previous_dont_write
        sys.pycache_prefix = previous_pycache_prefix
        sys.path[:] = previous_sys_path
        sys.meta_path[:] = previous_meta_path
        sys.path_hooks[:] = previous_path_hooks
        sys.path_importer_cache = previous_path_importer_cache
    if receipt is None:
        _fail("Meta V11 public replay produced no receipt")
    receipt = _copy(_mapping(receipt, "meta_v11_replay_receipt"))
    _exact_keys(receipt, V11_RECEIPT_FIELDS, "meta_v11_replay_receipt")
    for key in (
        "report_digest", "contract_digest", "qualification_input_digest",
        "source_transition_report_digest", "parent_theory_state_digest",
        "child_theory_state_digest",
    ):
        _digest(receipt.get(key), f"meta_v11_replay_receipt.{key}")
    for key in ("holdout_evaluated", "stress_confirmation_evaluated", "qualified"):
        _bool(receipt.get(key), f"meta_v11_replay_receipt.{key}")
    if (
        receipt["report_digest"]
        != expected_digests["expected_post_transition_qualification_report_digest"]
        or receipt["contract_digest"] != META_V11_CONTRACT_DIGEST
        or receipt["qualification_input_digest"]
        != expected_digests["expected_post_transition_qualification_input_digest"]
        or receipt["source_transition_report_digest"]
        != expected_digests["expected_interval_transition_report_digest"]
    ):
        _fail("Meta V11 replay receipt digest binding differs")
    source_only_receipt = {
        "policy": "SOURCE_ONLY_IMPORT_WITH_EXTERNAL_UNAVAILABLE_PYCACHE_PREFIX",
        "preloaded_performance_modules_absent": True,
        "dont_write_bytecode_active": True,
        "pycache_prefix": META_SOURCE_ONLY_PYCACHE_PREFIX,
        "namespace_package_path": META_PERFORMANCE_ROOT,
        "loaded_source_module_origins": loaded_origins,
        "in_tree_bytecode_loaded": False,
        "newly_loaded_performance_modules_removed_after_replay": True,
        "interpreter_import_state_restored": True,
        "exact_meta_root_temporarily_inserted_in_sys_path": True,
        "standard_import_machinery_only": True,
        "source_file_loader_only_for_project": True,
        "path_hooks_and_importer_cache_restored": True,
        "sys_path_and_meta_path_restored": True,
        "live_git_commit_tree_clean_raw_hash_checks_performed": False,
    }
    return receipt, source_only_receipt


def _audit_events(qualified: bool) -> tuple[list[dict[str, Any]], str]:
    kinds = [
        "OWNER_AUTHORIZATION_REPLAYED",
        "META_V11_PUBLIC_30_OBJECT_REPLAYED",
        (
            "DESTINATION_SCOPED_ADOPTION_DECIDED"
            if qualified
            else "META_V11_BRIDGE_REFUSED"
        ),
        "ATTEMPT2_ADMISSION_RESOLVED",
    ]
    events: list[dict[str, Any]] = []
    previous = GENESIS_DIGEST
    for sequence, event_type in enumerate(kinds):
        body = {
            "sequence": sequence,
            "event_type": event_type,
            "previous_event_digest": previous,
        }
        event = {
            **body,
            "event_digest": _domain_digest(DOMAINS["audit_event"], body),
        }
        events.append(event)
        previous = event["event_digest"]
    return events, previous


def _owner_authorization_binding() -> dict[str, Any]:
    return {
        "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "authority_role": "PROJECT_OWNER",
        "confirmation_semantics": "EXACT_CONFIRMATION_OF_SELECTED_ANTECEDENT",
        "authorized_attempt_ordinal": 2,
        "maximum_actual_attempts": 1,
        "authorization_replayed": True,
        "authorization_is_execution_evidence": False,
        "authorization_is_external_signature": False,
    }


def _qualification_binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": receipt["status"],
        "disposition": receipt["disposition"],
        "qualification_id": receipt["qualification_id"],
        "qualification_input_digest": receipt["qualification_input_digest"],
        "source_transition_report_digest": receipt[
            "source_transition_report_digest"
        ],
        "parent_theory_state_digest": receipt["parent_theory_state_digest"],
        "child_theory_state_digest": receipt["child_theory_state_digest"],
        "operation_kind": receipt["operation_kind"],
        "transition_kind": receipt["transition_kind"],
        "holdout_evaluated": receipt["holdout_evaluated"],
        "stress_confirmation_evaluated": receipt[
            "stress_confirmation_evaluated"
        ],
        "qualified": receipt["qualified"],
        "source_adoption_eligibility": receipt["adoption_eligibility"],
        "source_adoption_status": receipt["adoption_status"],
        "source_promotion_status": receipt["promotion_status"],
        "source_current_status": receipt["current_status"],
    }


def _qualified_receipt_gate(receipt: Mapping[str, Any]) -> bool:
    if (
        receipt["adoption_eligibility"]
        != "NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED"
        or receipt["adoption_status"] != "NOT_ADOPTED_SHADOW_ONLY"
        or receipt["promotion_status"] != "NOT_PROMOTED"
        or receipt["current_status"] != "NOT_CURRENT"
    ):
        _fail("Meta V11 source lifecycle surface differs")
    qualified = receipt["disposition"] == META_V11_QUALIFIED
    if receipt["qualified"] is not qualified:
        _fail("Meta V11 qualified boolean and disposition differ")
    if receipt["status"] != "VERIFIED_" + receipt["disposition"]:
        _fail("Meta V11 verifier status and disposition differ")
    if qualified and (
        receipt["status"] != META_V11_VERIFIED_QUALIFIED
        or receipt["holdout_evaluated"] is not True
        or receipt["stress_confirmation_evaluated"] is not True
    ):
        _fail("qualified Meta V11 receipt lacks fresh holdout/stress confirmation")
    return qualified


def _destination_adoption_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    for key in (
        "report_digest", "qualification_id", "parent_theory_state_digest",
        "child_theory_state_digest", "operation_kind", "transition_kind",
    ):
        _text(receipt[key], f"qualified Meta V11 {key}")
    body = {
        "schema_version": ADOPTION_RECEIPT_SCHEMA_VERSION,
        "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "meta_v11_contract_digest": META_V11_CONTRACT_DIGEST,
        "meta_v11_report_digest": receipt["report_digest"],
        "qualification_id": receipt["qualification_id"],
        "parent_theory_state_digest": receipt["parent_theory_state_digest"],
        "child_theory_state_digest": receipt["child_theory_state_digest"],
        "operation_kind": receipt["operation_kind"],
        "transition_kind": receipt["transition_kind"],
        "source_adoption_eligibility": receipt["adoption_eligibility"],
        "source_adoption_status": receipt["adoption_status"],
        "decision": (
            "ADOPT_EXACT_QUALIFIED_META_V11_CHILD_AS_HEGEL_Q05B_ATTEMPT2_INPUT_ONLY"
        ),
        "adoption_scope": "HEGEL_Q05B_ATTEMPT2_ONLY",
        "destination_input_eligible": True,
        "destination_input_adopted": True,
        "meta_global_adoption": False,
        "meta_current_pointer_written": False,
    }
    return {
        **body,
        "receipt_digest": _domain_digest(DOMAINS["adoption_receipt"], body),
    }


def _report_authority(qualified: bool) -> dict[str, Any]:
    return {
        **AUTHORITY_CAPABILITIES,
        "meta_v11_replay_performed": True,
        "meta_v11_qualified": qualified,
        "owner_authority_recorded": qualified,
        "destination_scoped_adoption_decided": qualified,
        "attempt2_admitted": qualified,
        "ledger_consumed": False,
        "actual_called": False,
        "execution_link_created": False,
        "failure_link_created": False,
    }


def _build_bridge_report(
    bridge_input: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    receipt: Mapping[str, Any],
    source_only_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    qualified = _qualified_receipt_gate(receipt)
    adoption = _destination_adoption_receipt(receipt) if qualified else None
    source_commit = bridge_input["hegel_source_binding"]["source_commit"]
    external_decision = None
    admission = None
    ledger_plan = None
    if qualified:
        external_decision = {
            "authority_kind": "PROJECT_OWNER_CONTEXTUAL_CONFIRMATION",
            "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
            "decision": (
                "AUTHORIZE_EXACT_QUALIFIED_META_V11_CHILD_AS_HEGEL_Q05B_ATTEMPT2_INPUT_ONLY"
            ),
            "decision_scope": "HEGEL_Q05B_ATTEMPT2_ONLY",
            "decision_performed": True,
            "external_signature_verified": False,
            "execution_evidence_claimed": False,
        }
        admission = {
            "admitted": True,
            "attempt_ordinal": 2,
            "maximum_actual_invocations": 1,
            "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
            "destination_adoption_receipt_digest": adoption["receipt_digest"],
            "hegel_source_commit": source_commit,
            "artifact_path": Q05B_ARTIFACT_PATH,
            "cargo_cache_source": CARGO_CACHE_SOURCE,
            "run_actual_entrypoint": RUN_ACTUAL_ENTRYPOINT,
            "ledger_consumed": False,
            "actual_called": False,
        }
        ledger_plan = {
            "root": LEDGER_ROOT,
            "initial_state": "OPEN_OR_EXACT_BRIDGE_STAGED_NOT_CONSUMED",
            "bridge_record_name": LEDGER_NAMES["bridge"],
            "consume_record_name": LEDGER_NAMES["consumed"],
            "success_record_name": LEDGER_NAMES["execution"],
            "failure_record_name": LEDGER_NAMES["failure"],
            "no_replace": True,
            "consume_before_actual": True,
            "retry_after_consume_allowed": False,
        }
    events, head = _audit_events(qualified)
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_digest": FROZEN_CONTRACT_DIGEST,
        "bridge_input_digest": _domain_digest(DOMAINS["input"], bridge_input),
        "disposition": BRIDGE_ADMITTED if qualified else BRIDGE_REFUSED,
        "owner_authorization_binding": _owner_authorization_binding(),
        "meta_source_binding": {
            "declared_binding": META_SOURCE_BINDING,
            "source_only_import_receipt": _copy(source_only_receipt),
            "live_git_commit_tree_clean_raw_hash_checks_performed": False,
        },
        "meta_v11_replay_receipt": _copy(receipt),
        "meta_v11_qualification_binding": _qualification_binding(receipt),
        "external_authority_decision": external_decision,
        "destination_adoption_receipt": adoption,
        "attempt_history_binding": ATTEMPT_HISTORY_BINDING,
        "hegel_destination_binding": {
            "declared_source_binding": bridge_input["hegel_source_binding"],
            "execution_request": bridge_input["execution_request"],
            "live_source_checks_performed": False,
            "actual_execution_performed": False,
        },
        "attempt2_admission": admission,
        "durable_ledger_plan": ledger_plan,
        "authority_boundary": _report_authority(qualified),
        "nonclaims": NONCLAIMS,
        "input_artifacts": _copy(artifacts),
        "audit_events": events,
        "audit_head": head,
    }
    return {
        **body,
        "report_digest": _domain_digest(DOMAINS["report"], body),
    }


@dataclass(frozen=True)
class Phase3Q05BAttempt2MetaV11BridgeResult:
    report: dict[str, object]

    @property
    def disposition(self) -> str:
        return str(self.report["disposition"])

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    @property
    def admitted(self) -> bool:
        return self.disposition == BRIDGE_ADMITTED

    def to_dict(self) -> dict[str, object]:
        return _copy(self.report)


def build_phase3_q05b_attempt2_meta_v11_bridge_v1(
    competition_input: Mapping[str, Any],
    competition_contract: Mapping[str, Any],
    competition_report: Mapping[str, Any],
    transition_contract: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    qualification_input: Mapping[str, Any],
    qualification_contract: Mapping[str, Any],
    qualification_report: Mapping[str, Any],
    review_contract: Mapping[str, Any],
    review_report: Mapping[str, Any],
    probe_input: Mapping[str, Any],
    probe_contract: Mapping[str, Any],
    probe_report: Mapping[str, Any],
    restriction_input: Mapping[str, Any],
    restriction_contract: Mapping[str, Any],
    restriction_report: Mapping[str, Any],
    adjudication_input: Mapping[str, Any],
    adjudication_contract: Mapping[str, Any],
    adjudication_report: Mapping[str, Any],
    adapter_input: Mapping[str, Any],
    adapter_contract: Mapping[str, Any],
    adapter_report: Mapping[str, Any],
    interval_competition_input: Mapping[str, Any],
    interval_competition_contract: Mapping[str, Any],
    interval_competition_report: Mapping[str, Any],
    interval_transition_contract: Mapping[str, Any],
    interval_transition_report: Mapping[str, Any],
    post_transition_qualification_input: Mapping[str, Any],
    post_transition_qualification_contract: Mapping[str, Any],
    post_transition_qualification_report: Mapping[str, Any],
    attempt2_bridge_input: Mapping[str, Any],
    attempt2_bridge_contract: Mapping[str, Any],
    *,
    input_artifacts: Mapping[str, Any] | None = None,
) -> Phase3Q05BAttempt2MetaV11BridgeResult:
    """Build one pure bridge report; never executes or writes the attempt."""

    if not isinstance(post_transition_qualification_report, Mapping) or not post_transition_qualification_report:
        raise MissingRealV11ReportError()
    contract = validate_phase3_q05b_attempt2_meta_v11_bridge_contract_v1(
        attempt2_bridge_contract
    )
    bridge_input = _validate_input(attempt2_bridge_input, contract)
    if input_artifacts is None:
        _fail("independent input_artifacts are required")
    artifacts = _validate_input_artifacts(input_artifacts)
    positional = (
        competition_input, competition_contract, competition_report,
        transition_contract, transition_report, qualification_input,
        qualification_contract, qualification_report, review_contract,
        review_report, probe_input, probe_contract, probe_report,
        restriction_input, restriction_contract, restriction_report,
        adjudication_input, adjudication_contract, adjudication_report,
        adapter_input, adapter_contract, adapter_report,
        interval_competition_input, interval_competition_contract,
        interval_competition_report, interval_transition_contract,
        interval_transition_report, post_transition_qualification_input,
        post_transition_qualification_contract,
        post_transition_qualification_report,
    )
    receipt, source_only_receipt = _public_v11_replay(
        positional,
        bridge_input["meta_v11_expected_digests"],
        artifacts,
    )
    report = _build_bridge_report(
        bridge_input, artifacts, receipt, source_only_receipt
    )
    return Phase3Q05BAttempt2MetaV11BridgeResult(report)


def validate_phase3_q05b_attempt2_meta_v11_bridge_report_v1(
    value: Mapping[str, Any],
    bridge_contract: Mapping[str, Any],
    *,
    expected_input_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = validate_phase3_q05b_attempt2_meta_v11_bridge_contract_v1(
        bridge_contract
    )
    report = _copy(_mapping(value, "attempt2_bridge_report"))
    _exact_keys(report, REPORT_FIELDS, "attempt2_bridge_report")
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("contract_id") != CONTRACT_ID
        or report.get("contract_digest") != FROZEN_CONTRACT_DIGEST
        or report.get("owner_authorization_binding")
        != _owner_authorization_binding()
        or report.get("attempt_history_binding") != ATTEMPT_HISTORY_BINDING
        or report.get("nonclaims") != NONCLAIMS
    ):
        _fail("attempt2 bridge report fixed surfaces differ")
    _digest(report.get("bridge_input_digest"), "bridge_input_digest")
    _digest(report.get("report_digest"), "report_digest")
    body = {key: report[key] for key in report if key != "report_digest"}
    if _domain_digest(DOMAINS["report"], body) != report["report_digest"]:
        _fail("attempt2 bridge report self digest differs")
    artifacts = _validate_input_artifacts(report.get("input_artifacts"))
    if expected_input_artifacts is not None:
        expected = _validate_input_artifacts(expected_input_artifacts)
        if artifacts != expected:
            _fail("attempt2 bridge report input_artifacts differ")
    source = _mapping(report.get("meta_source_binding"), "meta_source_binding")
    _exact_keys(
        source,
        {
            "declared_binding", "source_only_import_receipt",
            "live_git_commit_tree_clean_raw_hash_checks_performed",
        },
        "meta_source_binding",
    )
    if (
        source.get("declared_binding") != META_SOURCE_BINDING
        or source.get("live_git_commit_tree_clean_raw_hash_checks_performed")
        is not False
    ):
        _fail("bridge report Meta source claim boundary differs")
    source_receipt = _mapping(
        source.get("source_only_import_receipt"), "source_only_import_receipt"
    )
    _exact_keys(
        source_receipt,
        META_SOURCE_ONLY_RECEIPT_FIELDS,
        "source_only_import_receipt",
    )
    source_origins = _mapping(
        source_receipt.get("loaded_source_module_origins"),
        "loaded_source_module_origins",
    )
    if (
        source_receipt.get("policy")
        != "SOURCE_ONLY_IMPORT_WITH_EXTERNAL_UNAVAILABLE_PYCACHE_PREFIX"
        or
        source_receipt.get("preloaded_performance_modules_absent") is not True
        or source_receipt.get("dont_write_bytecode_active") is not True
        or source_receipt.get("pycache_prefix")
        != META_SOURCE_ONLY_PYCACHE_PREFIX
        or source_receipt.get("namespace_package_path") != META_PERFORMANCE_ROOT
        or source_receipt.get("in_tree_bytecode_loaded") is not False
        or source_receipt.get("source_file_loader_only_for_project") is not True
        or source_receipt.get("standard_import_machinery_only") is not True
        or source_receipt.get("interpreter_import_state_restored") is not True
        or source_receipt.get("path_hooks_and_importer_cache_restored") is not True
        or source_receipt.get("newly_loaded_performance_modules_removed_after_replay")
        is not True
        or source_receipt.get("exact_meta_root_temporarily_inserted_in_sys_path")
        is not True
        or source_receipt.get("sys_path_and_meta_path_restored") is not True
        or source_receipt.get("live_git_commit_tree_clean_raw_hash_checks_performed")
        is not False
    ):
        _fail("bridge report source-only replay receipt differs")
    required_v11_name = (
        "performance.shadow_interval_multi_q_post_transition_qualification"
    )
    if required_v11_name not in source_origins:
        _fail("source-only receipt omits the exact Meta V11 module")
    for name, path in source_origins.items():
        if type(name) is not str or not name.startswith("performance."):
            _fail("source-only receipt module name differs")
        expected_origin = Path(META_SOURCE_BINDING_ROOT) / (
            name.replace(".", "/") + ".py"
        )
        if path != str(expected_origin):
            _fail(f"source-only receipt origin differs for {name}")
    replay = _copy(
        _mapping(report.get("meta_v11_replay_receipt"), "meta_v11_replay_receipt")
    )
    _exact_keys(replay, V11_RECEIPT_FIELDS, "meta_v11_replay_receipt")
    qualified = _qualified_receipt_gate(replay)
    if report.get("meta_v11_qualification_binding") != _qualification_binding(replay):
        _fail("Meta V11 qualification binding differs")
    destination = _mapping(
        report.get("hegel_destination_binding"), "hegel_destination_binding"
    )
    _exact_keys(
        destination,
        {
            "declared_source_binding", "execution_request",
            "live_source_checks_performed", "actual_execution_performed",
        },
        "hegel_destination_binding",
    )
    declared_hegel = _validate_hegel_source_binding(
        destination.get("declared_source_binding"), contract
    )
    expected_request = {
        "mode": "HEGEL_Q05B_ACTUAL_ATTEMPT2",
        "attempt_ordinal": 2,
        "maximum_actual_invocations": 1,
        "source_commit": declared_hegel["source_commit"],
        "artifact_path": Q05B_ARTIFACT_PATH,
        "cargo_cache_source": CARGO_CACHE_SOURCE,
        "run_actual_entrypoint": RUN_ACTUAL_ENTRYPOINT,
    }
    if (
        destination.get("execution_request") != expected_request
        or destination.get("live_source_checks_performed") is not False
        or destination.get("actual_execution_performed") is not False
    ):
        _fail("Hegel destination claim boundary differs")
    expected_disposition = BRIDGE_ADMITTED if qualified else BRIDGE_REFUSED
    if report.get("disposition") != expected_disposition:
        _fail("bridge disposition differs from verified Meta V11 qualification")
    if qualified:
        adoption = _mapping(
            report.get("destination_adoption_receipt"),
            "destination_adoption_receipt",
        )
        expected_adoption = _destination_adoption_receipt(replay)
        expected_external = {
            "authority_kind": "PROJECT_OWNER_CONTEXTUAL_CONFIRMATION",
            "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
            "decision": (
                "AUTHORIZE_EXACT_QUALIFIED_META_V11_CHILD_AS_HEGEL_Q05B_ATTEMPT2_INPUT_ONLY"
            ),
            "decision_scope": "HEGEL_Q05B_ATTEMPT2_ONLY",
            "decision_performed": True,
            "external_signature_verified": False,
            "execution_evidence_claimed": False,
        }
        expected_admission = {
            "admitted": True,
            "attempt_ordinal": 2,
            "maximum_actual_invocations": 1,
            "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
            "destination_adoption_receipt_digest": expected_adoption[
                "receipt_digest"
            ],
            "hegel_source_commit": declared_hegel["source_commit"],
            "artifact_path": Q05B_ARTIFACT_PATH,
            "cargo_cache_source": CARGO_CACHE_SOURCE,
            "run_actual_entrypoint": RUN_ACTUAL_ENTRYPOINT,
            "ledger_consumed": False,
            "actual_called": False,
        }
        expected_plan = {
            "root": LEDGER_ROOT,
            "initial_state": "OPEN_OR_EXACT_BRIDGE_STAGED_NOT_CONSUMED",
            "bridge_record_name": LEDGER_NAMES["bridge"],
            "consume_record_name": LEDGER_NAMES["consumed"],
            "success_record_name": LEDGER_NAMES["execution"],
            "failure_record_name": LEDGER_NAMES["failure"],
            "no_replace": True,
            "consume_before_actual": True,
            "retry_after_consume_allowed": False,
        }
        if (
            adoption != expected_adoption
            or report.get("external_authority_decision") != expected_external
            or report.get("attempt2_admission") != expected_admission
            or report.get("durable_ledger_plan") != expected_plan
            or report.get("authority_boundary") != _report_authority(True)
        ):
            _fail("qualified bridge authority/adoption/admission surface differs")
    else:
        for key in (
            "external_authority_decision", "destination_adoption_receipt",
            "attempt2_admission", "durable_ledger_plan",
        ):
            if report.get(key) is not None:
                _fail("refused bridge must not expose adoption or admission")
        if report.get("authority_boundary") != _report_authority(False):
            _fail("refused bridge authority boundary differs")
    expected_events, expected_head = _audit_events(qualified)
    if (
        report.get("audit_events") != expected_events
        or report.get("audit_head") != expected_head
    ):
        _fail("bridge report exact audit semantics differ")
    return report


def verify_phase3_q05b_attempt2_meta_v11_bridge_v1(
    competition_input: Mapping[str, Any],
    competition_contract: Mapping[str, Any],
    competition_report: Mapping[str, Any],
    transition_contract: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    qualification_input: Mapping[str, Any],
    qualification_contract: Mapping[str, Any],
    qualification_report: Mapping[str, Any],
    review_contract: Mapping[str, Any],
    review_report: Mapping[str, Any],
    probe_input: Mapping[str, Any],
    probe_contract: Mapping[str, Any],
    probe_report: Mapping[str, Any],
    restriction_input: Mapping[str, Any],
    restriction_contract: Mapping[str, Any],
    restriction_report: Mapping[str, Any],
    adjudication_input: Mapping[str, Any],
    adjudication_contract: Mapping[str, Any],
    adjudication_report: Mapping[str, Any],
    adapter_input: Mapping[str, Any],
    adapter_contract: Mapping[str, Any],
    adapter_report: Mapping[str, Any],
    interval_competition_input: Mapping[str, Any],
    interval_competition_contract: Mapping[str, Any],
    interval_competition_report: Mapping[str, Any],
    interval_transition_contract: Mapping[str, Any],
    interval_transition_report: Mapping[str, Any],
    post_transition_qualification_input: Mapping[str, Any],
    post_transition_qualification_contract: Mapping[str, Any],
    post_transition_qualification_report: Mapping[str, Any],
    attempt2_bridge_input: Mapping[str, Any],
    attempt2_bridge_contract: Mapping[str, Any],
    attempt2_bridge_report: Mapping[str, Any],
    *,
    expected_attempt2_bridge_report_digest: str,
    expected_input_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay and byte-compare the exact 32-object bridge construction."""

    if not isinstance(post_transition_qualification_report, Mapping) or not post_transition_qualification_report:
        raise MissingRealV11ReportError()
    expected_digest = _digest(
        expected_attempt2_bridge_report_digest,
        "expected_attempt2_bridge_report_digest",
    )
    supplied = validate_phase3_q05b_attempt2_meta_v11_bridge_report_v1(
        attempt2_bridge_report,
        attempt2_bridge_contract,
        expected_input_artifacts=expected_input_artifacts,
    )
    fresh = build_phase3_q05b_attempt2_meta_v11_bridge_v1(
        competition_input, competition_contract, competition_report,
        transition_contract, transition_report, qualification_input,
        qualification_contract, qualification_report, review_contract,
        review_report, probe_input, probe_contract, probe_report,
        restriction_input, restriction_contract, restriction_report,
        adjudication_input, adjudication_contract, adjudication_report,
        adapter_input, adapter_contract, adapter_report,
        interval_competition_input, interval_competition_contract,
        interval_competition_report, interval_transition_contract,
        interval_transition_report, post_transition_qualification_input,
        post_transition_qualification_contract,
        post_transition_qualification_report, attempt2_bridge_input,
        attempt2_bridge_contract, input_artifacts=expected_input_artifacts,
    ).to_dict()
    if fresh["report_digest"] != expected_digest:
        _fail("replayed bridge report digest differs from independent expectation")
    if canonical_json_bytes_v1(supplied) != canonical_json_bytes_v1(fresh):
        _fail("supplied bridge report differs from exact replay")
    return {
        "status": "VERIFIED_" + fresh["disposition"],
        "disposition": fresh["disposition"],
        "report_digest": fresh["report_digest"],
        "contract_digest": fresh["contract_digest"],
        "bridge_input_digest": fresh["bridge_input_digest"],
        "meta_v11_report_digest": fresh["meta_v11_replay_receipt"][
            "report_digest"
        ],
        "destination_adoption_receipt_digest": (
            None
            if fresh["destination_adoption_receipt"] is None
            else fresh["destination_adoption_receipt"]["receipt_digest"]
        ),
        "admitted": fresh["disposition"] == BRIDGE_ADMITTED,
    }


CONSUMED_FIELDS: Final = frozenset({
    "schema_version", "status", "attempt_ordinal", "authorization_digest",
    "bridge_contract_digest", "bridge_report_digest",
    "destination_adoption_receipt_digest", "meta_v11_report_digest",
    "hegel_source_commit", "q05b_artifact_path", "previous_ledger_state",
    "consumed_before_actual", "retry_allowed", "record_digest",
})
CONSUMED_FILE_EVIDENCE_FIELDS: Final = frozenset({
    "path", "device", "inode", "nlink", "mode", "size", "sha256",
})
ACTUAL_SOURCE_ONLY_RECEIPT_FIELDS: Final = frozenset({
    "policy", "preloaded_hegel_machine_modules",
    "preloaded_modules_within_allowlist", "dont_write_bytecode_active",
    "pycache_prefix", "standard_import_machinery_only",
    "source_file_loader_only_for_project", "actual_tool_source_origin",
    "actual_tool_raw_sha256", "unchanged_source_file_evidence",
    "run_actual_callable_identity_verified", "loaded_hegel_source_module_origins",
    "in_tree_bytecode_loaded", "path_hooks_and_importer_cache_restored",
    "interpreter_import_state_restored",
    "live_source_commit_tree_clean_raw_hash_checks_performed",
    "unchanged_source_path_hash_checks_performed",
})
SOURCE_FILE_EVIDENCE_FIELDS: Final = frozenset({
    "path", "device", "inode", "nlink", "mode", "size", "sha256",
})
META_LIVE_SOURCE_RECEIPT_FIELDS: Final = frozenset({
    "repo_root", "git_commit", "git_tree", "git_clean",
    "v11_contract_digest", "exact_five_file_evidence",
    "commit_tree_clean_checks_performed", "exact_five_file_checks_performed",
    "v11_contract_digest_check_performed",
})
HEGEL_LIVE_SOURCE_RECEIPT_FIELDS: Final = frozenset({
    "project_root", "source_commit", "source_tree", "clean",
    "baseline_commit", "baseline_tree", "source_parent_commit",
    "source_commit_count_after_baseline", "source_direct_child_of_baseline",
    "attempt1_source_commit", "repair_commit", "repair_is_ancestor",
    "exact_five_diff_paths", "exact_five_file_evidence",
    "commit_tree_clean_checks_performed",
    "direct_child_and_commit_count_checks_performed",
    "attempt1_repair_lineage_checks_performed",
    "exact_five_diff_check_performed", "exact_five_file_checks_performed",
    "no_deletions_renames_or_other_edits",
})
EXECUTION_LINK_FIELDS: Final = frozenset({
    "schema_version", "status", "attempt_ordinal", "authorization_digest",
    "bridge_contract_digest", "bridge_report_digest",
    "destination_adoption_receipt_digest", "meta_v11_report_digest",
    "hegel_source_commit", "consumed_record_digest", "consumed_file_evidence",
    "run_actual_result_digest", "q05b_attempt_id",
    "q05b_admission_boundary_root", "q05b_actual_admission_evidence_root",
    "q05b_artifact_path", "q05b_artifact_raw_sha256",
    "q05b_artifact_canonical_sha256", "q05b_artifact_set_root",
    "q05b_candidate_receipt_root", "q05b_final_receipt_root",
    "q05b_qualification_count", "q05b_qualification_mask", "q1_state",
    "q1_gate_count", "q1_gate_mask", "q1_formal_output_roots",
    "artifact_written", "authority_boundary", "nonclaims", "audit_events",
    "audit_head", "execution_link_digest",
})
FAILURE_LINK_FIELDS: Final = frozenset({
    "schema_version", "status", "attempt_ordinal", "authorization_digest",
    "bridge_contract_digest", "bridge_report_digest",
    "destination_adoption_receipt_digest", "meta_v11_report_digest",
    "hegel_source_commit", "consumed_record_digest", "consumed_file_evidence",
    "failure_code", "failure_detail", "failure_detail_sha256",
    "artifact_written", "artifact_path_status", "actual_returned",
    "retry_allowed", "authority_boundary", "nonclaims", "audit_events",
    "audit_head", "failure_link_digest",
})


def build_missing_real_v11_report_error_v1() -> dict[str, Any]:
    """Return the exact canonical stderr object for the missing-report route."""

    return build_attempt2_supervisor_error_v1(
        MISSING_REAL_V11_REPORT,
        "required real post-transition qualification report is absent",
        bridge_report_created=False,
        ledger_created=False,
        ledger_consumed=False,
        actual_called=False,
        docker_or_cargo_called=False,
        q05b_artifact_written=False,
    )


def build_attempt2_supervisor_error_v1(
    status: str,
    detail: str,
    *,
    bridge_report_created: bool,
    ledger_created: bool,
    ledger_consumed: bool,
    actual_called: bool,
    docker_or_cargo_called: bool,
    q05b_artifact_written: bool,
) -> dict[str, Any]:
    code = _text(status, "supervisor error status", maximum=256)
    message = _text(detail, "supervisor error detail", maximum=4096)
    bridge = _bool(bridge_report_created, "bridge_report_created")
    ledger = _bool(ledger_created, "ledger_created")
    consumed = _bool(ledger_consumed, "ledger_consumed")
    called = _bool(actual_called, "actual_called")
    external = _bool(docker_or_cargo_called, "docker_or_cargo_called")
    artifact = _bool(q05b_artifact_written, "q05b_artifact_written")
    if consumed and (not bridge or not ledger):
        _fail("ledger consumption requires bridge report and ledger creation")
    if called and not consumed:
        _fail("actual call requires prior ledger consumption")
    if (external or artifact) and not called:
        _fail("external tool/artifact facts require actual call boundary")
    if code == MISSING_REAL_V11_REPORT and (
        message != "required real post-transition qualification report is absent"
        or any((bridge, ledger, consumed, called, external, artifact))
    ):
        _fail("missing real V11 report error facts must all be false")
    return {
        "schema_version": ERROR_SCHEMA_VERSION,
        "status": code,
        "detail": message,
        "bridge_report_created": bridge,
        "ledger_created": ledger,
        "ledger_consumed": consumed,
        "actual_called": called,
        "docker_or_cargo_called": external,
        "q05b_artifact_written": artifact,
    }


def validate_attempt2_supervisor_error_v1(value: object) -> dict[str, Any]:
    error = _copy(_mapping(value, "attempt2 supervisor error"))
    expected_keys = {
        "schema_version", "status", "detail", "bridge_report_created",
        "ledger_created", "ledger_consumed", "actual_called",
        "docker_or_cargo_called", "q05b_artifact_written",
    }
    _exact_keys(error, expected_keys, "attempt2 supervisor error")
    if error.get("schema_version") != ERROR_SCHEMA_VERSION:
        _fail("attempt2 supervisor error schema differs")
    rebuilt = build_attempt2_supervisor_error_v1(
        error["status"],
        error["detail"],
        bridge_report_created=error["bridge_report_created"],
        ledger_created=error["ledger_created"],
        ledger_consumed=error["ledger_consumed"],
        actual_called=error["actual_called"],
        docker_or_cargo_called=error["docker_or_cargo_called"],
        q05b_artifact_written=error["q05b_artifact_written"],
    )
    if error != rebuilt:
        _fail("attempt2 supervisor error wire differs")
    return error


def validate_missing_real_v11_report_error_v1(value: object) -> dict[str, Any]:
    expected = build_missing_real_v11_report_error_v1()
    error = _copy(_mapping(value, "missing V11 report error"))
    _exact_keys(error, set(expected), "missing V11 report error")
    if error != expected:
        _fail("missing V11 report error wire differs")
    return error


def build_attempt2_consumed_record_v1(
    bridge_report: Mapping[str, Any],
    bridge_contract: Mapping[str, Any],
) -> dict[str, Any]:
    report = validate_phase3_q05b_attempt2_meta_v11_bridge_report_v1(
        bridge_report, bridge_contract
    )
    if report["disposition"] != BRIDGE_ADMITTED:
        _fail("only an admitted bridge may consume attempt-2 authorization")
    body = {
        "schema_version": CONSUMED_SCHEMA_VERSION,
        "status": "ATTEMPT2_AUTHORIZATION_CONSUMED_BEFORE_ACTUAL",
        "attempt_ordinal": 2,
        "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "bridge_contract_digest": FROZEN_CONTRACT_DIGEST,
        "bridge_report_digest": report["report_digest"],
        "destination_adoption_receipt_digest": report[
            "destination_adoption_receipt"
        ]["receipt_digest"],
        "meta_v11_report_digest": report["meta_v11_replay_receipt"][
            "report_digest"
        ],
        "hegel_source_commit": report["attempt2_admission"][
            "hegel_source_commit"
        ],
        "q05b_artifact_path": Q05B_ARTIFACT_PATH,
        "previous_ledger_state": "BRIDGE_STAGED_NOT_CONSUMED",
        "consumed_before_actual": True,
        "retry_allowed": False,
    }
    return {
        **body,
        "record_digest": _domain_digest(DOMAINS["ledger_consumed"], body),
    }


def validate_attempt2_consumed_record_v1(
    value: Mapping[str, Any],
    bridge_report: Mapping[str, Any],
    bridge_contract: Mapping[str, Any],
) -> dict[str, Any]:
    consumed = _copy(_mapping(value, "attempt2 consumed record"))
    _exact_keys(consumed, CONSUMED_FIELDS, "attempt2 consumed record")
    expected = build_attempt2_consumed_record_v1(bridge_report, bridge_contract)
    if consumed != expected:
        _fail("attempt2 consumed record differs from exact admitted bridge")
    return consumed


def validate_consumed_file_evidence_v1(
    value: Mapping[str, Any], consumed_record: Mapping[str, Any]
) -> dict[str, Any]:
    evidence = _copy(_mapping(value, "consumed_file_evidence"))
    _exact_keys(evidence, CONSUMED_FILE_EVIDENCE_FIELDS, "consumed_file_evidence")
    for key in ("device", "inode", "nlink", "mode", "size"):
        _nonnegative_int(evidence.get(key), f"consumed_file_evidence.{key}")
    consumed = _copy(_mapping(consumed_record, "consumed_record"))
    _exact_keys(consumed, CONSUMED_FIELDS, "consumed_record")
    payload = canonical_json_bytes_v1(consumed)
    if (
        evidence.get("path") != LEDGER_ROOT + "/" + LEDGER_NAMES["consumed"]
        or evidence.get("device") == 0
        or evidence.get("inode") == 0
        or evidence.get("nlink") != 1
        or evidence.get("mode") != 0o444
        or evidence.get("size") != len(payload)
        or evidence.get("sha256") != _raw_sha(payload)
    ):
        _fail("consumed file evidence differs from exact immutable 0444 bytes")
    return evidence


def validate_hegel_actual_source_only_import_receipt_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _copy(_mapping(value, "Hegel actual source-only import receipt"))
    _exact_keys(
        receipt,
        ACTUAL_SOURCE_ONLY_RECEIPT_FIELDS,
        "Hegel actual source-only import receipt",
    )
    required_preloaded = [
        "hegel_machine",
        "hegel_machine.phase3_q05b_attempt2_meta_v11_bridge_v1",
    ]
    preloaded = receipt.get("preloaded_hegel_machine_modules")
    origins = _mapping(
        receipt.get("loaded_hegel_source_module_origins"),
        "loaded_hegel_source_module_origins",
    )
    if (
        preloaded != required_preloaded
        or receipt.get("policy")
        != "SOURCE_ONLY_HEGEL_ACTUAL_IMPORT_AND_CALL"
        or receipt.get("preloaded_modules_within_allowlist") is not True
        or receipt.get("dont_write_bytecode_active") is not True
        or receipt.get("pycache_prefix")
        != "/dev/null/hegel-q05b-attempt2-hegel-actual-source-only"
        or receipt.get("standard_import_machinery_only") is not True
        or receipt.get("source_file_loader_only_for_project") is not True
        or receipt.get("actual_tool_source_origin")
        != PROJECT_ROOT + "/tools/phase3_q05b_dual_qualification_v1.py"
        or receipt.get("actual_tool_raw_sha256")
        != HEGEL_UNCHANGED_RAW_SHA256[
            "tools/phase3_q05b_dual_qualification_v1.py"
        ]
        or receipt.get("run_actual_callable_identity_verified") is not True
        or receipt.get("in_tree_bytecode_loaded") is not False
        or receipt.get("path_hooks_and_importer_cache_restored") is not True
        or receipt.get("interpreter_import_state_restored") is not True
        or receipt.get("live_source_commit_tree_clean_raw_hash_checks_performed")
        is not True
        or receipt.get("unchanged_source_path_hash_checks_performed") is not True
    ):
        _fail("Hegel actual source-only import receipt differs")
    for name, path in origins.items():
        if type(name) is not str or not name.startswith("hegel_machine."):
            _fail("loaded Hegel module name differs")
        expected = Path(PROJECT_ROOT) / "src" / (
            name.replace(".", "/") + ".py"
        )
        path_text = _text(path, f"loaded origin {name}", maximum=8192)
        if path_text != str(expected) or ".." in Path(path_text).parts:
            _fail(f"loaded Hegel module {name} origin differs")
    required_origins = {
        "hegel_machine.phase3_q05b_attempt2_meta_v11_bridge_v1",
        "hegel_machine.phase3_q05b_actual_admission_v1",
    }
    if not required_origins.issubset(origins):
        _fail("Hegel actual source receipt omits required loaded source modules")
    evidence_map = _copy(
        _mapping(
            receipt.get("unchanged_source_file_evidence"),
            "unchanged_source_file_evidence",
        )
    )
    _exact_keys(
        evidence_map,
        set(HEGEL_UNCHANGED_RAW_SHA256),
        "unchanged_source_file_evidence",
    )
    identities: set[tuple[int, int]] = set()
    for relative, expected_sha in HEGEL_UNCHANGED_RAW_SHA256.items():
        row = _mapping(
            evidence_map[relative], f"unchanged_source_file_evidence.{relative}"
        )
        _exact_keys(
            row,
            SOURCE_FILE_EVIDENCE_FIELDS,
            f"unchanged_source_file_evidence.{relative}",
        )
        for key in ("device", "inode", "nlink", "mode", "size"):
            _nonnegative_int(
                row.get(key), f"unchanged_source_file_evidence.{relative}.{key}"
            )
        identity = (row["device"], row["inode"])
        if (
            row.get("path") != PROJECT_ROOT + "/" + relative
            or row.get("device") == 0
            or row.get("inode") == 0
            or row.get("nlink") != 1
            or not stat.S_ISREG(row.get("mode"))
            or row.get("sha256") != expected_sha
            or identity in identities
        ):
            _fail(f"unchanged source file evidence differs for {relative}")
        identities.add(identity)
    return receipt


def validate_meta_live_source_validation_receipt_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _copy(_mapping(value, "Meta live source validation receipt"))
    _exact_keys(
        receipt,
        META_LIVE_SOURCE_RECEIPT_FIELDS,
        "Meta live source validation receipt",
    )
    if (
        receipt.get("repo_root") != META_SOURCE_BINDING_ROOT
        or receipt.get("git_commit") != META_SOURCE_BINDING["git_commit"]
        or receipt.get("git_tree") != META_SOURCE_BINDING["git_tree"]
        or receipt.get("git_clean") is not True
        or receipt.get("v11_contract_digest") != META_V11_CONTRACT_DIGEST
        or receipt.get("commit_tree_clean_checks_performed") is not True
        or receipt.get("exact_five_file_checks_performed") is not True
        or receipt.get("v11_contract_digest_check_performed") is not True
    ):
        _fail("Meta live source identity receipt differs")
    evidence_map = _copy(
        _mapping(
            receipt.get("exact_five_file_evidence"),
            "Meta exact-five file evidence",
        )
    )
    _exact_keys(
        evidence_map,
        set(META_EXACT_FIVE_RAW_SHA256),
        "Meta exact-five file evidence",
    )
    identities: set[tuple[int, int]] = set()
    for relative, expected_sha in META_EXACT_FIVE_RAW_SHA256.items():
        row = _mapping(
            evidence_map[relative], f"Meta exact-five file evidence.{relative}"
        )
        _exact_keys(
            row,
            SOURCE_FILE_EVIDENCE_FIELDS,
            f"Meta exact-five file evidence.{relative}",
        )
        for key in ("device", "inode", "nlink", "mode", "size"):
            _nonnegative_int(row.get(key), f"Meta file evidence {relative}.{key}")
        identity = (row["device"], row["inode"])
        if (
            row.get("path") != META_SOURCE_BINDING_ROOT + "/" + relative
            or row.get("device") == 0
            or row.get("inode") == 0
            or row.get("nlink") != 1
            or not stat.S_ISREG(row.get("mode"))
            or row.get("size") == 0
            or row.get("sha256") != expected_sha
            or identity in identities
        ):
            _fail(f"Meta exact-five ordinary file evidence differs for {relative}")
        identities.add(identity)
    return receipt


def validate_hegel_live_source_validation_receipt_v1(
    value: Mapping[str, Any],
    expected_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _copy(_mapping(expected_source_binding, "expected Hegel source binding"))
    receipt = _copy(_mapping(value, "Hegel live source validation receipt"))
    _exact_keys(
        receipt,
        HEGEL_LIVE_SOURCE_RECEIPT_FIELDS,
        "Hegel live source validation receipt",
    )
    if (
        receipt.get("project_root") != PROJECT_ROOT
        or receipt.get("source_commit") != binding.get("source_commit")
        or receipt.get("source_tree") != binding.get("source_tree")
        or receipt.get("clean") is not True
        or receipt.get("baseline_commit") != binding.get("baseline_commit")
        or receipt.get("baseline_tree") != binding.get("baseline_tree")
        or receipt.get("source_parent_commit") != binding.get("baseline_commit")
        or receipt.get("source_commit_count_after_baseline") != 1
        or receipt.get("source_direct_child_of_baseline") is not True
        or receipt.get("attempt1_source_commit")
        != ATTEMPT_HISTORY_BINDING["attempt1_source_commit"]
        or receipt.get("repair_commit")
        != ATTEMPT_HISTORY_BINDING["repair_commit"]
        or receipt.get("repair_is_ancestor") is not True
        or receipt.get("exact_five_diff_paths") != EXACT_FIVE_PATHS
        or receipt.get("commit_tree_clean_checks_performed") is not True
        or receipt.get("direct_child_and_commit_count_checks_performed") is not True
        or receipt.get("attempt1_repair_lineage_checks_performed") is not True
        or receipt.get("exact_five_diff_check_performed") is not True
        or receipt.get("exact_five_file_checks_performed") is not True
        or receipt.get("no_deletions_renames_or_other_edits") is not True
    ):
        _fail("Hegel live source/lineage identity receipt differs")
    expected_hashes = _copy(
        _mapping(binding.get("exact_five_raw_sha256"), "Hegel exact-five hashes")
    )
    _exact_keys(expected_hashes, set(EXACT_FIVE_PATHS), "Hegel exact-five hashes")
    evidence_map = _copy(
        _mapping(
            receipt.get("exact_five_file_evidence"),
            "Hegel exact-five file evidence",
        )
    )
    _exact_keys(
        evidence_map, set(EXACT_FIVE_PATHS), "Hegel exact-five file evidence"
    )
    identities: set[tuple[int, int]] = set()
    for relative in EXACT_FIVE_PATHS:
        row = _mapping(
            evidence_map[relative], f"Hegel exact-five file evidence.{relative}"
        )
        _exact_keys(
            row,
            SOURCE_FILE_EVIDENCE_FIELDS,
            f"Hegel exact-five file evidence.{relative}",
        )
        for key in ("device", "inode", "nlink", "mode", "size"):
            _nonnegative_int(row.get(key), f"Hegel file evidence {relative}.{key}")
        expected_path = str(Path(PROJECT_ROOT).parent / relative)
        identity = (row["device"], row["inode"])
        if (
            row.get("path") != expected_path
            or row.get("device") == 0
            or row.get("inode") == 0
            or row.get("nlink") != 1
            or not stat.S_ISREG(row.get("mode"))
            or row.get("size") == 0
            or row.get("sha256") != expected_hashes[relative]
            or identity in identities
        ):
            _fail(f"Hegel exact-five ordinary file evidence differs for {relative}")
        identities.add(identity)
    return receipt


def _record_audit_events(kinds: list[str]) -> tuple[list[dict[str, Any]], str]:
    events: list[dict[str, Any]] = []
    previous = GENESIS_DIGEST
    for sequence, event_type in enumerate(kinds):
        body = {
            "sequence": sequence,
            "event_type": event_type,
            "previous_event_digest": previous,
        }
        event = {
            **body,
            "event_digest": _domain_digest(DOMAINS["audit_event"], body),
        }
        events.append(event)
        previous = event["event_digest"]
    return events, previous


def _artifact_source_only_replay(
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if type(payload) is not bytes or not payload:
        _fail("q05b_artifact must be non-empty exact bytes")
    allowed = {
        "hegel_machine",
        "hegel_machine.phase3_q05b_attempt2_meta_v11_bridge_v1",
    }
    preloaded = sorted(
        name for name in sys.modules
        if name == "hegel_machine" or name.startswith("hegel_machine.")
    )
    if not set(preloaded).issubset(allowed):
        _fail("preloaded Hegel modules forbid source-only artifact replay")
    previous_dont_write = sys.dont_write_bytecode
    previous_pycache_prefix = sys.pycache_prefix
    previous_sys_path = list(sys.path)
    previous_meta_path = list(sys.meta_path)
    previous_path_hooks = list(sys.path_hooks)
    previous_path_importer_cache = sys.path_importer_cache
    loaded_origins: dict[str, str] = {}
    replayed: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    source_root = Path(PROJECT_ROOT) / "src"
    try:
        sys.dont_write_bytecode = True
        sys.pycache_prefix = (
            "/dev/null/hegel-q05b-attempt2-hegel-artifact-source-only"
        )
        sys.path[:] = [str(source_root)] + [
            item for item in previous_sys_path if item != str(source_root)
        ]
        sys.meta_path[:] = [
            importlib.machinery.BuiltinImporter,
            importlib.machinery.FrozenImporter,
            importlib.machinery.PathFinder,
        ]
        sys.path_hooks[:] = [
            importlib.machinery.FileFinder.path_hook(
                (
                    importlib.machinery.SourceFileLoader,
                    importlib.machinery.SOURCE_SUFFIXES,
                ),
                (
                    importlib.machinery.ExtensionFileLoader,
                    importlib.machinery.EXTENSION_SUFFIXES,
                ),
            )
        ]
        sys.path_importer_cache = {}
        package = sys.modules.get("hegel_machine")
        package_paths = sorted(
            str(Path(path).resolve())
            for path in list(getattr(package, "__path__", ()))
        )
        if package_paths != [str(source_root / "hegel_machine")]:
            _fail("Hegel package path differs before artifact source replay")
        module = importlib.import_module(
            "hegel_machine.phase3_q05b_actual_artifact_v1"
        )
        decoder = getattr(module, "decode_and_replay_actual_artifact_v1", None)
        summarizer = getattr(module, "actual_artifact_summary_v1", None)
        if (
            not callable(decoder)
            or not callable(summarizer)
            or getattr(decoder, "__module__", None) != module.__name__
            or getattr(summarizer, "__module__", None) != module.__name__
        ):
            _fail("Q0.5b artifact public replay callable identity differs")
        try:
            replayed = decoder(payload)
            summary = summarizer(replayed)
        except Exception as error:
            if error.__class__.__module__ != module.__name__:
                raise
            _fail(f"Q0.5b artifact strict public replay rejected: {error}")
        for name, loaded in sorted(sys.modules.items()):
            if not name.startswith("hegel_machine.") or name in allowed:
                continue
            origin = getattr(loaded, "__file__", None)
            if type(origin) is not str:
                _fail(f"loaded Hegel artifact dependency {name} lacks source origin")
            resolved = Path(origin).resolve()
            try:
                resolved.relative_to(source_root / "hegel_machine")
            except ValueError:
                _fail(f"loaded Hegel artifact dependency {name} lies outside root")
            if (
                resolved.suffix != ".py"
                or "__pycache__" in resolved.parts
                or not isinstance(
                    getattr(loaded, "__loader__", None),
                    importlib.machinery.SourceFileLoader,
                )
            ):
                _fail(f"loaded Hegel artifact dependency {name} is not .py source")
            loaded_origins[name] = str(resolved)
        artifact_origin = Path(getattr(module, "__file__", "")).resolve()
        if (
            _raw_sha(artifact_origin.read_bytes())
            != HEGEL_UNCHANGED_RAW_SHA256[
                "src/hegel_machine/phase3_q05b_actual_artifact_v1.py"
            ]
        ):
            _fail("Q0.5b artifact replay source raw hash differs")
    finally:
        for name in tuple(sys.modules):
            if (
                (name == "hegel_machine" or name.startswith("hegel_machine."))
                and name not in preloaded
            ):
                sys.modules.pop(name, None)
        sys.dont_write_bytecode = previous_dont_write
        sys.pycache_prefix = previous_pycache_prefix
        sys.path[:] = previous_sys_path
        sys.meta_path[:] = previous_meta_path
        sys.path_hooks[:] = previous_path_hooks
        sys.path_importer_cache = previous_path_importer_cache
    if replayed is None or summary is None:
        _fail("Q0.5b artifact public replay produced no result")
    receipt = {
        "policy": "SOURCE_ONLY_Q05B_ARTIFACT_PUBLIC_REPLAY",
        "preloaded_hegel_machine_modules": preloaded,
        "preloaded_modules_within_allowlist": True,
        "dont_write_bytecode_active": True,
        "pycache_prefix": (
            "/dev/null/hegel-q05b-attempt2-hegel-artifact-source-only"
        ),
        "standard_import_machinery_only": True,
        "source_file_loader_only_for_project": True,
        "loaded_hegel_source_module_origins": loaded_origins,
        "decode_and_replay_callable_identity_verified": True,
        "actual_artifact_summary_callable_identity_verified": True,
        "in_tree_bytecode_loaded": False,
        "path_hooks_and_importer_cache_restored": True,
        "interpreter_import_state_restored": True,
    }
    return _copy(replayed), _copy(summary), receipt


def _outcome_authority(
    *,
    success: bool,
    actual_source_receipt: Mapping[str, Any],
    artifact_replay_receipt: Mapping[str, Any] | None,
    unchanged_binding: Mapping[str, str],
    meta_live_source_receipt: Mapping[str, Any],
    hegel_live_source_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **AUTHORITY_CAPABILITIES,
        "meta_v11_replay_performed": True,
        "meta_v11_qualified": True,
        "owner_authority_recorded": True,
        "destination_scoped_adoption_decided": True,
        "attempt2_admitted": True,
        "ledger_consumed": True,
        "actual_called": True,
        "execution_link_created": success,
        "failure_link_created": not success,
        "hegel_live_source_checks_performed": True,
        "hegel_unchanged_source_binding": _copy(unchanged_binding),
        "meta_live_source_validation_receipt": _copy(meta_live_source_receipt),
        "hegel_live_source_validation_receipt": _copy(hegel_live_source_receipt),
        "hegel_actual_source_only_import_receipt": _copy(actual_source_receipt),
        "q05b_artifact_source_only_replay_receipt": _copy(
            artifact_replay_receipt
        ),
    }


RUN_ACTUAL_RESULT_FIELDS: Final = frozenset({
    "artifact_path", "artifact_sha256", "artifact_set_root",
    "candidate_receipt_root", "final_receipt_root", "qualification_count",
    "qualification_mask", "q1_state", "q1_gate_count", "q1_gate_mask",
    "q1_formal_output_roots",
    "actual_admission_prepublication_live_replay_root",
    "actual_admission_pre_artifact_live_replay_root",
    "actual_admission_stage10_live_replay_root",
    "actual_admission_postpublication_live_replay_root",
    "actual_final_delivery_identity", "actual_final_delivery_root",
    "publisher_handle_close_status", "stage_roots",
})


def build_attempt2_execution_link_v1(
    bridge_report: Mapping[str, Any],
    consumed_record: Mapping[str, Any],
    run_actual_result: Mapping[str, Any],
    q05b_artifact: bytes,
    bridge_contract: Mapping[str, Any],
    *,
    consumed_file_evidence: Mapping[str, Any],
    hegel_actual_source_only_import_receipt: Mapping[str, Any],
    meta_live_source_validation_receipt: Mapping[str, Any],
    hegel_live_source_validation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    report = validate_phase3_q05b_attempt2_meta_v11_bridge_report_v1(
        bridge_report, bridge_contract
    )
    consumed = validate_attempt2_consumed_record_v1(
        consumed_record, report, bridge_contract
    )
    file_evidence = validate_consumed_file_evidence_v1(
        consumed_file_evidence, consumed
    )
    actual_source = validate_hegel_actual_source_only_import_receipt_v1(
        hegel_actual_source_only_import_receipt
    )
    meta_live_source = validate_meta_live_source_validation_receipt_v1(
        meta_live_source_validation_receipt
    )
    declared_hegel_source = report["hegel_destination_binding"][
        "declared_source_binding"
    ]
    hegel_live_source = validate_hegel_live_source_validation_receipt_v1(
        hegel_live_source_validation_receipt, declared_hegel_source
    )
    result = _copy(_mapping(run_actual_result, "run_actual_result"))
    _exact_keys(result, RUN_ACTUAL_RESULT_FIELDS, "run_actual_result")
    artifact, summary, artifact_source = _artifact_source_only_replay(q05b_artifact)
    derived = _mapping(artifact.get("derived"), "q05b artifact derived")
    sections = _mapping(artifact.get("sections"), "q05b artifact sections")
    actual_admission = _mapping(
        sections.get("actual_admission"), "q05b actual admission section"
    )
    issue = _mapping(
        actual_admission.get("issue_record"), "q05b actual admission issue record"
    )
    actual_root = _hex(
        actual_admission.get("actual_admission_evidence_root"),
        64,
        "q05b actual admission evidence root",
    )
    attempt_id = _text(issue.get("attempt_id"), "q05b attempt id", maximum=512)
    boundary_root = _hex(
        issue.get("boundary_root"), 64, "q05b admission boundary root"
    )
    raw_sha = _raw_sha(q05b_artifact)
    for key in (
        "artifact_set_root", "candidate_receipt_root", "final_receipt_root",
    ):
        _hex(summary.get(key), 64, f"q05b summary {key}")
    canonical_sha = _hex(
        summary.get("canonical_sha256"), 64, "q05b canonical sha256"
    )
    final_identity = _mapping(
        result.get("actual_final_delivery_identity"),
        "run_actual_result.actual_final_delivery_identity",
    )
    if (
        artifact.get("status") != "Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN"
        or derived.get("qualification_count") != 20
        or derived.get("qualification_mask") != 0xFFFFF
        or result.get("artifact_path") != Q05B_ARTIFACT_PATH
        or result.get("artifact_sha256") != raw_sha
        or raw_sha != canonical_sha
        or result.get("artifact_set_root") != summary["artifact_set_root"]
        or result.get("candidate_receipt_root")
        != summary["candidate_receipt_root"]
        or result.get("final_receipt_root") != summary["final_receipt_root"]
        or result.get("qualification_count") != 20
        or result.get("qualification_mask") != 0xFFFFF
        or result.get("q1_state") != "NOT_RUN"
        or result.get("q1_gate_count") != 0
        or result.get("q1_gate_mask") != 0
        or result.get("q1_formal_output_roots") != [None] * 8
        or final_identity.get("source_commit") != consumed["hegel_source_commit"]
        or final_identity.get("artifact_path") != Q05B_ARTIFACT_PATH
        or final_identity.get("actual_admission_evidence_root") != actual_root
        or final_identity.get("artifact_set_root") != summary["artifact_set_root"]
        or final_identity.get("candidate_receipt_root")
        != summary["candidate_receipt_root"]
        or final_identity.get("final_receipt_root") != summary["final_receipt_root"]
        or final_identity.get("payload_sha256") != raw_sha
        or result.get("publisher_handle_close_status") != "CLOSED"
    ):
        _fail("run_actual result differs from strict Q0.5b artifact replay")
    unchanged = report["hegel_destination_binding"]["declared_source_binding"][
        "existing_unchanged_raw_sha256"
    ]
    authority = _outcome_authority(
        success=True,
        actual_source_receipt=actual_source,
        artifact_replay_receipt=artifact_source,
        unchanged_binding=unchanged,
        meta_live_source_receipt=meta_live_source,
        hegel_live_source_receipt=hegel_live_source,
    )
    events, head = _record_audit_events([
        "ATTEMPT2_AUTHORIZATION_CONSUMED",
        "HEGEL_Q05B_ACTUAL_RETURNED",
        "Q05B_ARTIFACT_PUBLIC_STRICT_REPLAYED",
        "ATTEMPT2_EXECUTION_LINK_CREATED",
    ])
    body = {
        "schema_version": EXECUTION_LINK_SCHEMA_VERSION,
        "status": ATTEMPT2_LINKED,
        "attempt_ordinal": 2,
        "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "bridge_contract_digest": FROZEN_CONTRACT_DIGEST,
        "bridge_report_digest": report["report_digest"],
        "destination_adoption_receipt_digest": report[
            "destination_adoption_receipt"
        ]["receipt_digest"],
        "meta_v11_report_digest": report["meta_v11_replay_receipt"][
            "report_digest"
        ],
        "hegel_source_commit": consumed["hegel_source_commit"],
        "consumed_record_digest": consumed["record_digest"],
        "consumed_file_evidence": file_evidence,
        "run_actual_result_digest": "sha256:" + _raw_sha(
            canonical_json_bytes_v1(result)
        ),
        "q05b_attempt_id": attempt_id,
        "q05b_admission_boundary_root": boundary_root,
        "q05b_actual_admission_evidence_root": actual_root,
        "q05b_artifact_path": Q05B_ARTIFACT_PATH,
        "q05b_artifact_raw_sha256": raw_sha,
        "q05b_artifact_canonical_sha256": canonical_sha,
        "q05b_artifact_set_root": summary["artifact_set_root"],
        "q05b_candidate_receipt_root": summary["candidate_receipt_root"],
        "q05b_final_receipt_root": summary["final_receipt_root"],
        "q05b_qualification_count": 20,
        "q05b_qualification_mask": 0xFFFFF,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_output_roots": [None] * 8,
        "artifact_written": True,
        "authority_boundary": authority,
        "nonclaims": NONCLAIMS,
        "audit_events": events,
        "audit_head": head,
    }
    return {
        **body,
        "execution_link_digest": _domain_digest(DOMAINS["execution_link"], body),
    }


def build_attempt2_failure_link_v1(
    bridge_report: Mapping[str, Any],
    consumed_record: Mapping[str, Any],
    failure_code: str,
    failure_detail: str,
    artifact_written: bool,
    bridge_contract: Mapping[str, Any],
    *,
    consumed_file_evidence: Mapping[str, Any],
    hegel_actual_source_only_import_receipt: Mapping[str, Any],
    meta_live_source_validation_receipt: Mapping[str, Any],
    hegel_live_source_validation_receipt: Mapping[str, Any],
    actual_returned: bool,
) -> dict[str, Any]:
    report = validate_phase3_q05b_attempt2_meta_v11_bridge_report_v1(
        bridge_report, bridge_contract
    )
    consumed = validate_attempt2_consumed_record_v1(
        consumed_record, report, bridge_contract
    )
    file_evidence = validate_consumed_file_evidence_v1(
        consumed_file_evidence, consumed
    )
    actual_source = validate_hegel_actual_source_only_import_receipt_v1(
        hegel_actual_source_only_import_receipt
    )
    meta_live_source = validate_meta_live_source_validation_receipt_v1(
        meta_live_source_validation_receipt
    )
    declared_hegel_source = report["hegel_destination_binding"][
        "declared_source_binding"
    ]
    hegel_live_source = validate_hegel_live_source_validation_receipt_v1(
        hegel_live_source_validation_receipt, declared_hegel_source
    )
    code = _text(failure_code, "failure_code", maximum=256)
    detail = _text(failure_detail, "failure_detail", maximum=4096)
    written = _bool(artifact_written, "artifact_written")
    returned = _bool(actual_returned, "actual_returned")
    unchanged = report["hegel_destination_binding"]["declared_source_binding"][
        "existing_unchanged_raw_sha256"
    ]
    authority = _outcome_authority(
        success=False,
        actual_source_receipt=actual_source,
        artifact_replay_receipt=None,
        unchanged_binding=unchanged,
        meta_live_source_receipt=meta_live_source,
        hegel_live_source_receipt=hegel_live_source,
    )
    events, head = _record_audit_events([
        "ATTEMPT2_AUTHORIZATION_CONSUMED",
        (
            "HEGEL_Q05B_POSTRETURN_VALIDATION_FAILED_CLOSED"
            if returned
            else "HEGEL_Q05B_ACTUAL_FAILED_CLOSED"
        ),
        "ATTEMPT2_FAILURE_LINK_CREATED",
    ])
    body = {
        "schema_version": FAILURE_LINK_SCHEMA_VERSION,
        "status": ATTEMPT2_FAILED,
        "attempt_ordinal": 2,
        "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "bridge_contract_digest": FROZEN_CONTRACT_DIGEST,
        "bridge_report_digest": report["report_digest"],
        "destination_adoption_receipt_digest": report[
            "destination_adoption_receipt"
        ]["receipt_digest"],
        "meta_v11_report_digest": report["meta_v11_replay_receipt"][
            "report_digest"
        ],
        "hegel_source_commit": consumed["hegel_source_commit"],
        "consumed_record_digest": consumed["record_digest"],
        "consumed_file_evidence": file_evidence,
        "failure_code": code,
        "failure_detail": detail,
        "failure_detail_sha256": _raw_sha(detail.encode("utf-8")),
        "artifact_written": written,
        "artifact_path_status": (
            "RESIDUAL_PRESENT_FAIL_CLOSED" if written else "ABSENT"
        ),
        "actual_returned": returned,
        "retry_allowed": False,
        "authority_boundary": authority,
        "nonclaims": NONCLAIMS,
        "audit_events": events,
        "audit_head": head,
    }
    return {
        **body,
        "failure_link_digest": _domain_digest(DOMAINS["failure_link"], body),
    }


def validate_attempt2_execution_link_v1(
    value: Mapping[str, Any],
    bridge_report: Mapping[str, Any],
    consumed_record: Mapping[str, Any],
    run_actual_result: Mapping[str, Any],
    q05b_artifact: bytes,
    bridge_contract: Mapping[str, Any],
    *,
    consumed_file_evidence: Mapping[str, Any],
    hegel_actual_source_only_import_receipt: Mapping[str, Any],
    meta_live_source_validation_receipt: Mapping[str, Any],
    hegel_live_source_validation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    link = _copy(_mapping(value, "attempt2 execution link"))
    _exact_keys(link, EXECUTION_LINK_FIELDS, "attempt2 execution link")
    expected = build_attempt2_execution_link_v1(
        bridge_report,
        consumed_record,
        run_actual_result,
        q05b_artifact,
        bridge_contract,
        consumed_file_evidence=consumed_file_evidence,
        hegel_actual_source_only_import_receipt=(
            hegel_actual_source_only_import_receipt
        ),
        meta_live_source_validation_receipt=meta_live_source_validation_receipt,
        hegel_live_source_validation_receipt=hegel_live_source_validation_receipt,
    )
    if link != expected:
        _fail("attempt2 execution link differs from strict replay")
    return link


def validate_attempt2_failure_link_v1(
    value: Mapping[str, Any],
    bridge_report: Mapping[str, Any],
    consumed_record: Mapping[str, Any],
    failure_code: str,
    failure_detail: str,
    artifact_written: bool,
    bridge_contract: Mapping[str, Any],
    *,
    consumed_file_evidence: Mapping[str, Any],
    hegel_actual_source_only_import_receipt: Mapping[str, Any],
    meta_live_source_validation_receipt: Mapping[str, Any],
    hegel_live_source_validation_receipt: Mapping[str, Any],
    actual_returned: bool,
) -> dict[str, Any]:
    link = _copy(_mapping(value, "attempt2 failure link"))
    _exact_keys(link, FAILURE_LINK_FIELDS, "attempt2 failure link")
    expected = build_attempt2_failure_link_v1(
        bridge_report,
        consumed_record,
        failure_code,
        failure_detail,
        artifact_written,
        bridge_contract,
        consumed_file_evidence=consumed_file_evidence,
        hegel_actual_source_only_import_receipt=(
            hegel_actual_source_only_import_receipt
        ),
        meta_live_source_validation_receipt=meta_live_source_validation_receipt,
        hegel_live_source_validation_receipt=hegel_live_source_validation_receipt,
        actual_returned=actual_returned,
    )
    if link != expected:
        _fail("attempt2 failure link differs from exact failed-closed record")
    return link


def _validate_artifact_source_only_receipt(value: object) -> dict[str, Any]:
    receipt = _copy(_mapping(value, "artifact source-only replay receipt"))
    expected_fields = {
        "policy", "preloaded_hegel_machine_modules",
        "preloaded_modules_within_allowlist", "dont_write_bytecode_active",
        "pycache_prefix", "standard_import_machinery_only",
        "source_file_loader_only_for_project",
        "loaded_hegel_source_module_origins",
        "decode_and_replay_callable_identity_verified",
        "actual_artifact_summary_callable_identity_verified",
        "in_tree_bytecode_loaded", "path_hooks_and_importer_cache_restored",
        "interpreter_import_state_restored",
    }
    _exact_keys(receipt, expected_fields, "artifact source-only replay receipt")
    origins = _mapping(
        receipt.get("loaded_hegel_source_module_origins"),
        "artifact replay loaded origins",
    )
    preloaded = receipt.get("preloaded_hegel_machine_modules")
    allowed = {
        "hegel_machine",
        "hegel_machine.phase3_q05b_attempt2_meta_v11_bridge_v1",
    }
    if (
        receipt.get("policy") != "SOURCE_ONLY_Q05B_ARTIFACT_PUBLIC_REPLAY"
        or type(preloaded) is not list
        or preloaded != sorted(preloaded)
        or not set(preloaded).issubset(allowed)
        or receipt.get("preloaded_modules_within_allowlist") is not True
        or receipt.get("dont_write_bytecode_active") is not True
        or receipt.get("pycache_prefix")
        != "/dev/null/hegel-q05b-attempt2-hegel-artifact-source-only"
        or receipt.get("standard_import_machinery_only") is not True
        or receipt.get("source_file_loader_only_for_project") is not True
        or receipt.get("decode_and_replay_callable_identity_verified") is not True
        or receipt.get("actual_artifact_summary_callable_identity_verified")
        is not True
        or receipt.get("in_tree_bytecode_loaded") is not False
        or receipt.get("path_hooks_and_importer_cache_restored") is not True
        or receipt.get("interpreter_import_state_restored") is not True
    ):
        _fail("artifact source-only replay receipt differs")
    required = "hegel_machine.phase3_q05b_actual_artifact_v1"
    if required not in origins:
        _fail("artifact source-only receipt omits artifact module")
    for name, path in origins.items():
        if type(name) is not str or not name.startswith("hegel_machine."):
            _fail("artifact replay module name differs")
        expected = Path(PROJECT_ROOT) / "src" / (
            name.replace(".", "/") + ".py"
        )
        if path != str(expected):
            _fail(f"artifact replay source origin differs for {name}")
    return receipt


def _validate_outcome_authority(
    value: object,
    *,
    success: bool,
    hegel_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _copy(_mapping(value, "outcome authority_boundary"))
    expected_keys = set(AUTHORITY_CAPABILITIES) | {
        "meta_v11_replay_performed", "meta_v11_qualified",
        "owner_authority_recorded", "destination_scoped_adoption_decided",
        "attempt2_admitted", "ledger_consumed", "actual_called",
        "execution_link_created", "failure_link_created",
        "hegel_live_source_checks_performed", "hegel_unchanged_source_binding",
        "meta_live_source_validation_receipt",
        "hegel_live_source_validation_receipt",
        "hegel_actual_source_only_import_receipt",
        "q05b_artifact_source_only_replay_receipt",
    }
    _exact_keys(authority, expected_keys, "outcome authority_boundary")
    actual_receipt = validate_hegel_actual_source_only_import_receipt_v1(
        authority.get("hegel_actual_source_only_import_receipt")
    )
    meta_live_receipt = validate_meta_live_source_validation_receipt_v1(
        authority.get("meta_live_source_validation_receipt")
    )
    hegel_live_receipt = validate_hegel_live_source_validation_receipt_v1(
        authority.get("hegel_live_source_validation_receipt"),
        hegel_source_binding,
    )
    artifact_receipt = authority.get("q05b_artifact_source_only_replay_receipt")
    if success:
        artifact_receipt = _validate_artifact_source_only_receipt(artifact_receipt)
    elif artifact_receipt is not None:
        _fail("failed outcome must not claim successful artifact replay")
    expected = _outcome_authority(
        success=success,
        actual_source_receipt=actual_receipt,
        artifact_replay_receipt=artifact_receipt,
        unchanged_binding=hegel_source_binding["existing_unchanged_raw_sha256"],
        meta_live_source_receipt=meta_live_receipt,
        hegel_live_source_receipt=hegel_live_receipt,
    )
    if authority != expected:
        _fail("outcome authority boundary differs")
    return authority


def validate_attempt2_execution_link_record_v1(
    value: Mapping[str, Any],
    bridge_report: Mapping[str, Any],
    consumed_record: Mapping[str, Any],
    bridge_contract: Mapping[str, Any],
) -> dict[str, Any]:
    report = validate_phase3_q05b_attempt2_meta_v11_bridge_report_v1(
        bridge_report, bridge_contract
    )
    consumed = validate_attempt2_consumed_record_v1(
        consumed_record, report, bridge_contract
    )
    link = _copy(_mapping(value, "attempt2 execution link record"))
    _exact_keys(link, EXECUTION_LINK_FIELDS, "attempt2 execution link record")
    body = {key: link[key] for key in link if key != "execution_link_digest"}
    hegel_source = report["hegel_destination_binding"]["declared_source_binding"]
    _validate_outcome_authority(
        link.get("authority_boundary"), success=True, hegel_source_binding=hegel_source
    )
    expected_events, expected_head = _record_audit_events([
        "ATTEMPT2_AUTHORIZATION_CONSUMED",
        "HEGEL_Q05B_ACTUAL_RETURNED",
        "Q05B_ARTIFACT_PUBLIC_STRICT_REPLAYED",
        "ATTEMPT2_EXECUTION_LINK_CREATED",
    ])
    if (
        link.get("schema_version") != EXECUTION_LINK_SCHEMA_VERSION
        or link.get("status") != ATTEMPT2_LINKED
        or link.get("attempt_ordinal") != 2
        or link.get("authorization_digest") != OWNER_AUTHORIZATION_DIGEST
        or link.get("bridge_contract_digest") != FROZEN_CONTRACT_DIGEST
        or link.get("bridge_report_digest") != report["report_digest"]
        or link.get("destination_adoption_receipt_digest")
        != report["destination_adoption_receipt"]["receipt_digest"]
        or link.get("meta_v11_report_digest")
        != report["meta_v11_replay_receipt"]["report_digest"]
        or link.get("hegel_source_commit") != consumed["hegel_source_commit"]
        or link.get("consumed_record_digest") != consumed["record_digest"]
        or link.get("q05b_artifact_path") != Q05B_ARTIFACT_PATH
        or link.get("q05b_artifact_raw_sha256")
        != link.get("q05b_artifact_canonical_sha256")
        or link.get("q05b_qualification_count") != 20
        or link.get("q05b_qualification_mask") != 0xFFFFF
        or link.get("q1_state") != "NOT_RUN"
        or link.get("q1_gate_count") != 0
        or link.get("q1_gate_mask") != 0
        or link.get("q1_formal_output_roots") != [None] * 8
        or link.get("artifact_written") is not True
        or link.get("nonclaims") != NONCLAIMS
        or link.get("audit_events") != expected_events
        or link.get("audit_head") != expected_head
        or _domain_digest(DOMAINS["execution_link"], body)
        != link.get("execution_link_digest")
    ):
        _fail("attempt2 execution link standalone record differs")
    for key in (
        "run_actual_result_digest", "q05b_attempt_id",
        "q05b_admission_boundary_root", "q05b_actual_admission_evidence_root",
        "q05b_artifact_raw_sha256", "q05b_artifact_canonical_sha256",
        "q05b_artifact_set_root", "q05b_candidate_receipt_root",
        "q05b_final_receipt_root",
    ):
        if key == "q05b_attempt_id":
            _text(link.get(key), key)
        elif key == "run_actual_result_digest":
            _digest(link.get(key), key)
        else:
            _hex(link.get(key), 64, key)
    validate_consumed_file_evidence_v1(
        link.get("consumed_file_evidence"), consumed
    )
    return link


def validate_attempt2_failure_link_record_v1(
    value: Mapping[str, Any],
    bridge_report: Mapping[str, Any],
    consumed_record: Mapping[str, Any],
    bridge_contract: Mapping[str, Any],
) -> dict[str, Any]:
    report = validate_phase3_q05b_attempt2_meta_v11_bridge_report_v1(
        bridge_report, bridge_contract
    )
    consumed = validate_attempt2_consumed_record_v1(
        consumed_record, report, bridge_contract
    )
    link = _copy(_mapping(value, "attempt2 failure link record"))
    _exact_keys(link, FAILURE_LINK_FIELDS, "attempt2 failure link record")
    returned = _bool(link.get("actual_returned"), "actual_returned")
    written = _bool(link.get("artifact_written"), "artifact_written")
    hegel_source = report["hegel_destination_binding"]["declared_source_binding"]
    _validate_outcome_authority(
        link.get("authority_boundary"), success=False, hegel_source_binding=hegel_source
    )
    expected_events, expected_head = _record_audit_events([
        "ATTEMPT2_AUTHORIZATION_CONSUMED",
        (
            "HEGEL_Q05B_POSTRETURN_VALIDATION_FAILED_CLOSED"
            if returned
            else "HEGEL_Q05B_ACTUAL_FAILED_CLOSED"
        ),
        "ATTEMPT2_FAILURE_LINK_CREATED",
    ])
    body = {key: link[key] for key in link if key != "failure_link_digest"}
    detail = _text(link.get("failure_detail"), "failure_detail", maximum=4096)
    if (
        link.get("schema_version") != FAILURE_LINK_SCHEMA_VERSION
        or link.get("status") != ATTEMPT2_FAILED
        or link.get("attempt_ordinal") != 2
        or link.get("authorization_digest") != OWNER_AUTHORIZATION_DIGEST
        or link.get("bridge_contract_digest") != FROZEN_CONTRACT_DIGEST
        or link.get("bridge_report_digest") != report["report_digest"]
        or link.get("destination_adoption_receipt_digest")
        != report["destination_adoption_receipt"]["receipt_digest"]
        or link.get("meta_v11_report_digest")
        != report["meta_v11_replay_receipt"]["report_digest"]
        or link.get("hegel_source_commit") != consumed["hegel_source_commit"]
        or link.get("consumed_record_digest") != consumed["record_digest"]
        or link.get("failure_detail_sha256") != _raw_sha(detail.encode("utf-8"))
        or link.get("artifact_path_status")
        != ("RESIDUAL_PRESENT_FAIL_CLOSED" if written else "ABSENT")
        or link.get("retry_allowed") is not False
        or link.get("nonclaims") != NONCLAIMS
        or link.get("audit_events") != expected_events
        or link.get("audit_head") != expected_head
        or _domain_digest(DOMAINS["failure_link"], body)
        != link.get("failure_link_digest")
    ):
        _fail("attempt2 failure link standalone record differs")
    _text(link.get("failure_code"), "failure_code", maximum=256)
    validate_consumed_file_evidence_v1(
        link.get("consumed_file_evidence"), consumed
    )
    return link


__all__ = [
    "ALREADY_CONSUMED", "ATTEMPT2_FAILED", "ATTEMPT2_LINKED",
    "BRIDGE_ADMITTED", "BRIDGE_REFUSED", "CONTRACT_ID",
    "CONTRACT_SCHEMA_VERSION", "ERROR_SCHEMA_VERSION",
    "EXACT_FIVE_PATHS", "FROZEN_CONTRACT_DIGEST", "INPUT_SCHEMA_VERSION",
    "LEDGER_NAMES", "LEDGER_ROOT", "MISSING_REAL_V11_REPORT",
    "MissingRealV11ReportError",
    "Phase3Q05BAttempt2MetaV11BridgeResult",
    "Phase3Q05BAttempt2MetaV11BridgeValidationError",
    "REPORT_SCHEMA_VERSION", "build_attempt2_consumed_record_v1",
    "build_attempt2_execution_link_v1", "build_attempt2_failure_link_v1",
    "build_missing_real_v11_report_error_v1",
    "build_attempt2_supervisor_error_v1",
    "build_phase3_q05b_attempt2_meta_v11_bridge_v1",
    "canonical_json_bytes_v1", "validate_attempt2_consumed_record_v1",
    "validate_attempt2_execution_link_v1", "validate_attempt2_failure_link_v1",
    "validate_attempt2_execution_link_record_v1",
    "validate_attempt2_failure_link_record_v1",
    "validate_consumed_file_evidence_v1",
    "validate_hegel_actual_source_only_import_receipt_v1",
    "validate_hegel_live_source_validation_receipt_v1",
    "validate_meta_live_source_validation_receipt_v1",
    "validate_missing_real_v11_report_error_v1",
    "validate_attempt2_supervisor_error_v1",
    "validate_phase3_q05b_attempt2_meta_v11_bridge_contract_v1",
    "validate_phase3_q05b_attempt2_meta_v11_bridge_report_v1",
    "verify_phase3_q05b_attempt2_meta_v11_bridge_v1",
]
