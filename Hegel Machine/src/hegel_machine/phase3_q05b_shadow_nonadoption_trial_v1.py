"""Pure records for the Q0.5b shadow non-adoption mechanics trial.

This module has no filesystem, process, network, Meta, adoption, or Hegel Q1
authority.  It only canonicalizes and validates the source-bound contract,
execution capsule, owner-direction record, and one-shot ledger records.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import re
from typing import Any, Final, Mapping, NoReturn


__all__ = (
    "ShadowNonadoptionTrialCoreError",
    "canonical_json_bytes_v1",
    "validate_shadow_nonadoption_trial_contract_v1",
    "build_shadow_nonadoption_execution_capsule_v1",
    "validate_shadow_nonadoption_execution_capsule_v1",
    "validate_shadow_nonadoption_owner_authorization_v1",
    "validate_shadow_nonadoption_ledger_record_v1",
)


_TAGGED_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256: Final = re.compile(r"[0-9a-f]{64}\Z")
_PLACEHOLDER_SOURCE_STATUS: Final = (
    "PLACEHOLDER_SOURCE_NOT_IMPLEMENTED_NOT_BOUND_EXECUTION_FORBIDDEN"
)
_FINAL_SOURCE_STATUS: Final = (
    "SOURCE_CORE_TOOL_TESTS_PRESENT_AND_BOUND_EXECUTION_STILL_REQUIRES_OWNER_AUTHORIZATION"
)
_SOURCE_PATHS: Final = (
    "Hegel Machine/src/hegel_machine/phase3_q05b_shadow_nonadoption_trial_v1.py",
    "Hegel Machine/tools/phase3_q05b_shadow_nonadoption_trial_v1.py",
    "Hegel Machine/tests/test_phase3_q05b_shadow_nonadoption_trial_v1.py",
)
_FAILURE_STATES: Final = (
    "LEDGER_ROOT_OR_LOCK_PUBLICATION_UNCERTAIN_TERMINAL",
    "INTENT_PUBLICATION_UNCERTAIN_TERMINAL",
    "INTENT_STAGED_PREPARATION_FAILED_CLOSED_UNCONSUMED",
    "CONSUMPTION_PUBLICATION_UNCERTAIN_TERMINAL",
    "CONSUMED_BEFORE_SUPERVISOR_LAUNCH_FAILED",
    "SUPERVISOR_RAISED_TIMED_OUT_OR_CLEANUP_UNCERTAIN",
    "SUPERVISOR_RETURNED_NONZERO_TERMINAL",
    "SUPERVISOR_RETURNED_ARTIFACT_MISSING_OR_INVALID",
    "POSTRETURN_NONADOPTION_OR_ISOLATION_REPLAY_FAILED",
    "SHADOW_OUTCOME_LINK_PUBLICATION_UNCERTAIN",
)
_FAILURE_MATRIX_FACT_FIELDS: Final = (
    "ledger_created",
    "ledger_consumed",
    "consumption_publication_uncertain",
    "supervisor_invocation_count",
    "supervisor_spawn_attempted",
    "supervisor_called",
    "supervisor_returned",
    "docker_or_cargo_may_have_run",
    "artifact_may_exist",
    "cleanup_attempted",
    "cleanup_complete",
    "survivor_absence_verified",
    "attempt2_state_touched",
    "meta_or_production_state_touched",
    "adoption_or_current_pointer_written",
    "failure_record_publication_uncertain",
    "no_retry",
)
_FAILURE_POLICY_RAW_SHA256: Final = (
    "292e89b13a0886be06b4cef5c386ea128584409cc8cd331510079a18c7b83649"
)


class ShadowNonadoptionTrialCoreError(ValueError):
    """Fail-closed pure-record validation error."""


def _fail(detail: str) -> NoReturn:
    raise ShadowNonadoptionTrialCoreError(detail)


def _validate_json_tree(value: object, label: str = "value") -> None:
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return
    if value_type is float:
        if not math.isfinite(value):
            _fail(f"{label} contains a non-finite float")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{label}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(f"{label} contains a non-string key")
            _validate_json_tree(item, f"{label}.{key}")
        return
    _fail(f"{label} contains a non-exact JSON runtime type")


def canonical_json_bytes_v1(value: object) -> bytes:
    """Return exact compact ASCII JSON plus one LF."""

    _validate_json_tree(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        _fail(f"canonical JSON encoding failed: {error}")
    return (text + "\n").encode("ascii")


def _canonical_body(value: object) -> bytes:
    return canonical_json_bytes_v1(value)[:-1]


def _exact_mapping(value: object, fields: list[str] | tuple[str, ...], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} is not an exact mapping")
    if set(value) != set(fields) or len(value) != len(fields):
        _fail(f"{label} fields differ")
    _validate_json_tree(value, label)
    return value


def _tagged(value: object, label: str) -> str:
    if type(value) is not str or _TAGGED_SHA256.fullmatch(value) is None:
        _fail(f"{label} is not a tagged SHA-256 digest")
    return value


def _raw(value: object, label: str) -> str:
    if type(value) is not str or _RAW_SHA256.fullmatch(value) is None:
        _fail(f"{label} is not a raw SHA-256 digest")
    return value


def _domain_digest(domain: object, value: object) -> str:
    if type(domain) is not str or not domain.endswith("\x00"):
        _fail("hash domain differs")
    return "sha256:" + sha256(domain.encode("ascii") + _canonical_body(value)).hexdigest()


def _without(value: Mapping[str, Any], fields: list[str]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    for field in fields:
        if field not in result:
            _fail(f"digest exclusion field is absent: {field}")
        result.pop(field)
    return result


def _contract(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("contract is not an exact mapping")
    _validate_json_tree(value, "contract")
    canonical = value.get("canonicalization")
    if type(canonical) is not dict:
        _fail("contract canonicalization is absent")
    fields = canonical.get("contract_exact_top_level_fields")
    if type(fields) is not list or any(type(item) is not str for item in fields):
        _fail("contract field registry differs")
    _exact_mapping(value, fields, "contract")
    domains = value.get("hash_domains")
    if type(domains) is not dict:
        _fail("contract hash domains differ")
    excludes = canonical.get("contract_digest_excludes")
    if excludes != ["contract_digest"]:
        _fail("contract digest exclusions differ")
    expected = _domain_digest(domains.get("contract"), _without(value, excludes))
    if value.get("contract_digest") != expected:
        _fail("contract digest differs")
    if value.get("schema_version") != "hegel-phase3a-q05b-shadow-nonadoption-trial-contract/1":
        _fail("contract schema differs")
    if value.get("contract_id") != "hegel-phase3a-q05b-shadow-nonadoption-trial-v1":
        _fail("contract id differs")
    failure = value.get("failure_state_policy")
    if type(failure) is not dict or failure.get("states") != list(_FAILURE_STATES):
        _fail("failure state registry differs")
    if failure.get("matrix_fact_fields") != list(_FAILURE_MATRIX_FACT_FIELDS):
        _fail("failure matrix fact registry differs")
    matrix = failure.get("state_fact_matrix")
    if (
        type(matrix) is not list
        or [row.get("error_state") if type(row) is dict else None for row in matrix]
        != list(_FAILURE_STATES)
    ):
        _fail("failure matrix order differs")
    failure_policy_raw_sha256 = sha256(canonical_json_bytes_v1(failure)).hexdigest()
    if failure_policy_raw_sha256 != _FAILURE_POLICY_RAW_SHA256:
        _fail("failure policy differs from the producer-free frozen policy")
    _validate_source_binding(value)
    return value


def _validate_source_binding(contract: Mapping[str, Any]) -> dict[str, Any]:
    canonical = contract["canonicalization"]
    fields = canonical.get("source_binding_exact_fields")
    if type(fields) is not list or any(type(item) is not str for item in fields):
        _fail("source binding field registry differs")
    row = _exact_mapping(contract.get("source_binding"), fields, "source binding")
    if row["expected_future_source_paths"] != list(_SOURCE_PATHS):
        _fail("source binding paths differ")
    if row["normalized_source_digest_semantics"] != (
        "MASKED_IO_TOOL_BYTES_WITH_EXACT_ONE_LINE_SOURCE_BINDING_LITERAL_"
        "REPLACED_BY_CANONICAL_SENTINEL"
    ):
        _fail("normalized source semantics differ")
    if row["source_raw_sha256_semantics"] != (
        "PURE_CORE_RAW_BYTES_NO_CONTRACT_DIGEST_LITERAL"
    ):
        _fail("core source semantics differ")
    if row["source_static_surface_sha256_semantics"] != (
        "RAW_SHA256_OF_CANONICAL_ASCII_JSON_SINGLE_LF_MANIFEST_WITH_SCHEMA_VERSION_"
        "AND_ORDERED_CORE_THEN_TOOL_ROWS_EACH_ROW_PATH_SORTED_IMPORTS_SORTED_PUBLIC_"
        "FUNCTION_SIGNATURE_ROWS_SORTED_PUBLIC_CLASSES_SORTED_SUBPROCESS_CALL_NAMES_"
        "AND_THREE_FORBIDDEN_BOOLEAN_FACTS"
    ):
        _fail("source static surface semantics differ")
    if row["tests_raw_sha256_semantics"] != "FOCUSED_TEST_FILE_RAW_BYTES":
        _fail("tests raw semantics differ")
    if row["tool_binding_policy"] != (
        "IO_TOOL_IS_BOUND_BY_NORMALIZED_SOURCE_DIGEST_NOT_RAW_DIGEST_TO_AVOID_"
        "CONTRACT_LITERAL_CYCLE; UNIQUE_MASKED_REGION_MUST_BE_ONE_DATA_ONLY_ASSIGNMENT"
    ):
        _fail("tool binding policy differs")
    status = row["status"]
    digest_fields = (
        "normalized_source_digest",
        "source_binding_digest",
        "source_raw_sha256",
        "source_static_surface_sha256",
        "tests_raw_sha256",
    )
    if status == _PLACEHOLDER_SOURCE_STATUS:
        if any(row[field] is not None for field in digest_fields):
            _fail("placeholder source binding contains a digest")
        return row
    if status != _FINAL_SOURCE_STATUS:
        _fail("source binding status differs")
    _tagged(row["normalized_source_digest"], "normalized source digest")
    _tagged(row["source_binding_digest"], "source binding digest")
    _raw(row["source_raw_sha256"], "core source raw digest")
    _raw(row["source_static_surface_sha256"], "source static surface digest")
    _raw(row["tests_raw_sha256"], "tests raw digest")
    policy = canonical.get("record_digest_policy", {}).get("source_binding")
    if type(policy) is not dict:
        _fail("source binding digest policy is absent")
    if policy.get("digest_field") != "source_binding_digest":
        _fail("source binding digest field differs")
    if policy.get("excluded_fields") != ["source_binding_digest"]:
        _fail("source binding digest exclusions differ")
    if policy.get("domain_key") != "source_binding":
        _fail("source binding digest domain differs")
    expected = _domain_digest(
        contract["hash_domains"].get("source_binding"),
        _without(row, ["source_binding_digest"]),
    )
    if row["source_binding_digest"] != expected:
        _fail("source binding digest differs")
    return row


def _record_digest(contract: Mapping[str, Any], kind: str, record: Mapping[str, Any]) -> str:
    policy = contract["canonicalization"]["record_digest_policy"].get(kind)
    if type(policy) is not dict:
        _fail(f"record digest policy is absent: {kind}")
    field = policy.get("digest_field")
    excludes = policy.get("excluded_fields")
    if type(field) is not str or excludes != [field]:
        _fail(f"record digest policy differs: {kind}")
    domain_key = policy.get("domain_key")
    if type(domain_key) is not str:
        _fail(f"record domain key differs: {kind}")
    return _domain_digest(contract["hash_domains"].get(domain_key), _without(record, excludes))


def validate_shadow_nonadoption_trial_contract_v1(contract: object) -> dict[str, Any]:
    """Validate and copy the exact immutable construction contract."""

    return deepcopy(_contract(contract))


def _executable_identity(
    value: object, expected_path: str, label: str
) -> dict[str, Any]:
    fields = ("path", "raw_sha256", "mode", "uid", "nlink", "size", "dev", "ino")
    row = _exact_mapping(value, fields, f"{label} executable identity")
    if row["path"] != expected_path:
        _fail(f"{label} executable path differs")
    _raw(row["raw_sha256"], f"{label} executable digest")
    for field in ("mode", "uid", "nlink", "size", "dev", "ino"):
        if type(row[field]) is not int or row[field] < 0:
            _fail(f"{label} executable {field} differs")
    if row["nlink"] < 1 or row["size"] < 1:
        _fail(f"{label} executable identity is not one linked nonempty file")
    return row


def _cargo_identity(value: object, contract: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "source_root",
        "target_root",
        "file_count",
        "directory_count",
        "total_bytes",
        "manifest_digest",
        "target_initially_absent",
        "copy_required",
    )
    row = _exact_mapping(value, fields, "Cargo cache identity")
    policy = contract["actual_execution_policy"]["isolated_cargo_cache_policy"]
    if row["source_root"] != policy["copy_source"] or row["target_root"] != policy["copy_target"]:
        _fail("Cargo cache roots differ")
    for field in ("file_count", "directory_count", "total_bytes"):
        if type(row[field]) is not int or row[field] < 0:
            _fail(f"Cargo cache {field} differs")
    if row["file_count"] < 1 or row["directory_count"] < 1 or row["total_bytes"] < 1:
        _fail("Cargo cache manifest is empty")
    _tagged(row["manifest_digest"], "Cargo cache manifest digest")
    if row["target_initially_absent"] is not True or row["copy_required"] is not True:
        _fail("Cargo cache target expectation differs")
    return row


def _protected_registry(contract: Mapping[str, Any]) -> dict[str, Any]:
    protected = contract["protected_state_policy"]
    source = contract["isolated_source_policy"]
    actual = contract["actual_execution_policy"]
    owner = contract["owner_authorization_policy"]
    ledger = contract["trial_ledger_policy"]
    return {
        "forbidden_exact_roots": deepcopy(protected["forbidden_exact_roots"]),
        "attempt2_artifact_path": protected["attempt2_artifact_path"],
        "trial_authorization_root": owner["authorization_root"],
        "trial_ledger_root": ledger["ledger_root"],
        "trial_clone_root": source["clone_root"],
        "trial_cargo_cache_root": actual["cargo_cache_source"],
    }


def build_shadow_nonadoption_execution_capsule_v1(
    contract: object,
    python_executable_identity: object,
    git_executable_identity: object,
    docker_executable_identity: object,
    cargo_cache_identity: object,
) -> dict[str, Any]:
    """Build the deterministic, non-authoritative execution capsule."""

    frozen = _contract(contract)
    capsule_policy = frozen["capsule_policy"]
    source = frozen["isolated_source_policy"]
    actual = frozen["actual_execution_policy"]
    owner = frozen["owner_authorization_policy"]
    ledger = frozen["trial_ledger_policy"]
    python_row = _executable_identity(
        python_executable_identity, capsule_policy["python_executable_path"], "Python"
    )
    git_row = _executable_identity(
        git_executable_identity, capsule_policy["git_executable_path"], "Git"
    )
    docker_row = _executable_identity(
        docker_executable_identity, capsule_policy["docker_executable_path"], "Docker"
    )
    cargo_row = _cargo_identity(cargo_cache_identity, frozen)
    source_binding = frozen["source_binding"]
    if source_binding.get("status") != _FINAL_SOURCE_STATUS:
        _fail("source binding is not final")
    _tagged(source_binding.get("source_binding_digest"), "source binding digest")
    registry = _protected_registry(frozen)
    protected_digest = _domain_digest(
        frozen["hash_domains"]["protected_path_registry"], registry
    )
    capsule: dict[str, Any] = {
        "schema_version": capsule_policy["schema_version"],
        "trial_id": frozen["trial_identity"]["trial_id"],
        "contract_digest": frozen["contract_digest"],
        "source_binding_digest": source_binding["source_binding_digest"],
        "authorization_root": owner["authorization_root"],
        "ledger_root": ledger["ledger_root"],
        "clone_root": source["clone_root"],
        "clone_project_root": source["clone_project_root"],
        "clone_git_commit": source["checkout_commit"],
        "clone_git_tree": source["checkout_tree"],
        "clone_git_parent": source["checkout_commit_parent"],
        "clone_git_clean": True,
        "old_supervisor_path": actual["old_supervisor_path"],
        "old_supervisor_raw_sha256": actual["old_supervisor_raw_sha256"],
        "ordered_argv": deepcopy(actual["ordered_argv_template"]),
        "python_executable_identity": deepcopy(python_row),
        "git_executable_identity": deepcopy(git_row),
        "docker_executable_identity": deepcopy(docker_row),
        "runtime_environment": deepcopy(capsule_policy["fixed_runtime_environment"]),
        "cargo_cache_identity": deepcopy(cargo_row),
        "input_registry": deepcopy(capsule_policy["fixed_input_registry"]),
        "metric_registry": deepcopy(capsule_policy["fixed_metric_registry"]),
        "seed_registry": deepcopy(capsule_policy["fixed_seed_registry"]),
        "resource_limits": deepcopy(capsule_policy["fixed_resource_limits"]),
        "monotonic_timeout_policy": deepcopy(
            capsule_policy["fixed_monotonic_timeout_policy"]
        ),
        "kill_reap_survivor_policy": deepcopy(
            capsule_policy["fixed_kill_reap_survivor_policy"]
        ),
        "stdio_fd_policy": deepcopy(capsule_policy["fixed_stdio_fd_policy"]),
        "artifact_path": actual["artifact_path"],
        "protected_path_registry_digest": protected_digest,
        "claim_ceiling": frozen["outcome_policy"]["claim_ceiling"],
        "capsule_digest": None,
    }
    _exact_mapping(capsule, capsule_policy["exact_fields"], "capsule")
    capsule["capsule_digest"] = _record_digest(frozen, "capsule", capsule)
    return validate_shadow_nonadoption_execution_capsule_v1(capsule, frozen)


def validate_shadow_nonadoption_execution_capsule_v1(
    capsule: object, contract: object
) -> dict[str, Any]:
    frozen = _contract(contract)
    policy = frozen["capsule_policy"]
    row = _exact_mapping(capsule, policy["exact_fields"], "capsule")
    if row["schema_version"] != policy["schema_version"]:
        _fail("capsule schema differs")
    if row["trial_id"] != frozen["trial_identity"]["trial_id"]:
        _fail("capsule trial id differs")
    if row["contract_digest"] != frozen["contract_digest"]:
        _fail("capsule contract digest differs")
    if row["source_binding_digest"] != frozen["source_binding"]["source_binding_digest"]:
        _fail("capsule source binding differs")
    source = frozen["isolated_source_policy"]
    actual = frozen["actual_execution_policy"]
    owner = frozen["owner_authorization_policy"]
    ledger = frozen["trial_ledger_policy"]
    exact = {
        "authorization_root": owner["authorization_root"],
        "ledger_root": ledger["ledger_root"],
        "clone_root": source["clone_root"],
        "clone_project_root": source["clone_project_root"],
        "clone_git_commit": source["checkout_commit"],
        "clone_git_tree": source["checkout_tree"],
        "clone_git_parent": source["checkout_commit_parent"],
        "clone_git_clean": True,
        "old_supervisor_path": actual["old_supervisor_path"],
        "old_supervisor_raw_sha256": actual["old_supervisor_raw_sha256"],
        "ordered_argv": actual["ordered_argv_template"],
        "runtime_environment": policy["fixed_runtime_environment"],
        "input_registry": policy["fixed_input_registry"],
        "metric_registry": policy["fixed_metric_registry"],
        "seed_registry": policy["fixed_seed_registry"],
        "resource_limits": policy["fixed_resource_limits"],
        "monotonic_timeout_policy": policy["fixed_monotonic_timeout_policy"],
        "kill_reap_survivor_policy": policy["fixed_kill_reap_survivor_policy"],
        "stdio_fd_policy": policy["fixed_stdio_fd_policy"],
        "artifact_path": actual["artifact_path"],
        "claim_ceiling": frozen["outcome_policy"]["claim_ceiling"],
    }
    for field, expected in exact.items():
        if _canonical_body(row[field]) != _canonical_body(expected):
            _fail(f"capsule {field} differs")
    if frozen["source_binding"]["status"] != _FINAL_SOURCE_STATUS:
        _fail("source binding is not final")
    _executable_identity(
        row["python_executable_identity"], policy["python_executable_path"], "Python"
    )
    _executable_identity(
        row["git_executable_identity"], policy["git_executable_path"], "Git"
    )
    _executable_identity(
        row["docker_executable_identity"], policy["docker_executable_path"], "Docker"
    )
    _cargo_identity(row["cargo_cache_identity"], frozen)
    expected_protected = _domain_digest(
        frozen["hash_domains"]["protected_path_registry"],
        _protected_registry(frozen),
    )
    if row["protected_path_registry_digest"] != expected_protected:
        _fail("capsule protected registry differs")
    expected_digest = _record_digest(frozen, "capsule", row)
    if row["capsule_digest"] != expected_digest:
        _fail("capsule digest differs")
    return deepcopy(row)


def validate_shadow_nonadoption_owner_authorization_v1(
    authorization: object, capsule: object, contract: object
) -> dict[str, Any]:
    frozen = _contract(contract)
    cap = validate_shadow_nonadoption_execution_capsule_v1(capsule, frozen)
    policy = frozen["owner_authorization_policy"]
    row = _exact_mapping(
        authorization, policy["authorization_record_exact_fields"], "authorization"
    )
    if row["schema_version"] != policy["schema_version"]:
        _fail("authorization schema differs")
    if row["trial_id"] != frozen["trial_identity"]["trial_id"]:
        _fail("authorization trial id differs")
    if row["contract_digest"] != frozen["contract_digest"]:
        _fail("authorization contract digest differs")
    if row["capsule_digest"] != cap["capsule_digest"]:
        _fail("authorization capsule digest differs")
    for field, expected in policy["authorization_fixed_fields"].items():
        if type(row[field]) is not type(expected) or row[field] != expected:
            _fail(f"authorization {field} differs")
    if row["authorization_digest"] != _record_digest(frozen, "authorization", row):
        _fail("authorization digest differs")
    return deepcopy(row)


def _validate_intent(row: dict[str, Any], contract: Mapping[str, Any]) -> None:
    ledger = contract["trial_ledger_policy"]
    if row["schema_version"] != ledger["intent_record_schema"]["schema_version"]:
        _fail("intent schema differs")
    expected = {
        "status": ledger["state_registry"]["intent_staged"],
        "trial_id": contract["trial_identity"]["trial_id"],
        "contract_digest": contract["contract_digest"],
        "clone_git_commit": contract["isolated_source_policy"]["checkout_commit"],
        "clone_git_tree": contract["isolated_source_policy"]["checkout_tree"],
        "old_supervisor_raw_sha256": contract["actual_execution_policy"]["old_supervisor_raw_sha256"],
        "artifact_path": contract["actual_execution_policy"]["artifact_path"],
        "previous_ledger_state": ledger["state_registry"]["open"],
        "no_retry": True,
    }
    for field, value in expected.items():
        if type(row[field]) is not type(value) or row[field] != value:
            _fail(f"intent {field} differs")
    for field in ("capsule_digest", "authorization_digest"):
        _tagged(row[field], f"intent {field}")


def _validate_consumed(row: dict[str, Any], contract: Mapping[str, Any]) -> None:
    ledger = contract["trial_ledger_policy"]
    if row["schema_version"] != ledger["consumed_record_schema"]["schema_version"]:
        _fail("consumed schema differs")
    expected = {
        "status": ledger["state_registry"]["consumed_outcome_unknown"],
        "trial_id": contract["trial_identity"]["trial_id"],
        "contract_digest": contract["contract_digest"],
        "previous_ledger_state": ledger["state_registry"]["intent_staged"],
        "consumed_before_supervisor": True,
        "retry_allowed": False,
    }
    for field, value in expected.items():
        if type(row[field]) is not type(value) or row[field] != value:
            _fail(f"consumed {field} differs")
    for field in ("capsule_digest", "authorization_digest", "intent_record_digest"):
        _tagged(row[field], f"consumed {field}")


def _validate_success(row: dict[str, Any], contract: Mapping[str, Any]) -> None:
    ledger = contract["trial_ledger_policy"]
    outcome = contract["outcome_policy"]
    if row["schema_version"] != ledger["success_record_schema"]["schema_version"]:
        _fail("success schema differs")
    expected = {
        "status": outcome["shadow_success_status"],
        "trial_id": contract["trial_identity"]["trial_id"],
        "contract_digest": contract["contract_digest"],
        "supervisor_invocation_count": 1,
        "supervisor_returned": True,
        "supervisor_exit_code": 0,
        "artifact_path": contract["actual_execution_policy"]["artifact_path"],
        "q05b_mechanics_predicate_count": 20,
        "q05b_mechanics_predicate_mask": 1048575,
        "q1_state": "NOT_RUN",
        "mechanics_only": True,
        "efficacy_claimed": False,
        "c1_claimed": False,
        "c2_claimed": False,
        "production_consumption_allowed": False,
        "meta_adoption_promotion_current_written": False,
        "attempt2_state_touched": False,
        "protected_state_replay_passed": True,
        "no_retry": True,
    }
    for field, value in expected.items():
        if type(row[field]) is not type(value) or row[field] != value:
            _fail(f"success {field} differs")
    for field in (
        "capsule_digest",
        "authorization_digest",
        "intent_record_digest",
        "consumed_record_digest",
    ):
        _tagged(row[field], f"success {field}")
    for field in (
        "supervisor_stdout_raw_sha256",
        "supervisor_stderr_raw_sha256",
        "artifact_raw_sha256",
    ):
        _raw(row[field], f"success {field}")


def _validate_failure(row: dict[str, Any], contract: Mapping[str, Any]) -> None:
    policy = contract["failure_state_policy"]
    if row["schema_version"] != policy["schema_version"]:
        _fail("failure schema differs")
    if row["status"] != policy["fixed_fields"]["status"]:
        _fail("failure status differs")
    if row["trial_id"] != contract["trial_identity"]["trial_id"]:
        _fail("failure trial id differs")
    if row["contract_digest"] != contract["contract_digest"]:
        _fail("failure contract digest differs")
    if type(row["error_state"]) is not str or row["error_state"] not in policy["states"]:
        _fail("failure state differs")
    if type(row["detail"]) is not str or not row["detail"]:
        _fail("failure detail differs")
    for field in ("capsule_digest", "authorization_digest", "intent_record_digest"):
        _tagged(row[field], f"failure {field}")
    for field, expected in policy["fixed_fields"].items():
        if field == "status":
            continue
        if type(row[field]) is not type(expected) or row[field] != expected:
            _fail(f"failure {field} differs")
    matrix = policy.get("state_fact_matrix")
    fact_fields = policy.get("matrix_fact_fields")
    if (
        type(matrix) is not list
        or type(fact_fields) is not list
        or any(type(field) is not str for field in fact_fields)
        or len(fact_fields) != len(set(fact_fields))
    ):
        _fail("failure matrix registry differs")
    matching = [item for item in matrix if type(item) is dict and item.get("error_state") == row["error_state"]]
    if len(matching) != 1:
        _fail("failure state matrix cardinality differs")
    state = matching[0]
    if set(state) != {
        "error_state",
        "consumed_record_digest_policy",
        "supervisor_exit_code_policy",
        "supervisor_output_digest_policy",
        "phase_ids",
        "allowed_facts",
    }:
        _fail("failure state matrix fields differ")
    phases = state["phase_ids"]
    if (
        type(phases) is not list
        or not phases
        or any(type(item) is not str or not item for item in phases)
        or len(phases) != len(set(phases))
        or row["phase_id"] not in phases
    ):
        _fail("failure phase differs")
    allowed_facts = state["allowed_facts"]
    if type(allowed_facts) is not dict or set(allowed_facts) != set(fact_fields):
        _fail("failure allowed-fact fields differ")
    for field in fact_fields:
        allowed = allowed_facts[field]
        if type(allowed) is not list or not allowed:
            _fail(f"failure allowed values differ: {field}")
        if not any(type(row[field]) is type(item) and row[field] == item for item in allowed):
            _fail(f"failure fact differs for state: {field}")
    consumed_policy = state["consumed_record_digest_policy"]
    consumed_digest = row["consumed_record_digest"]
    if consumed_policy == "NULL":
        if consumed_digest is not None:
            _fail("failure consumed digest must be null")
    elif consumed_policy == "TAGGED":
        _tagged(consumed_digest, "failure consumed digest")
    elif consumed_policy == "NULL_OR_TAGGED":
        if row["ledger_consumed"] is True:
            _tagged(consumed_digest, "failure consumed digest")
        elif row["ledger_consumed"] is None and consumed_digest is not None:
            _tagged(consumed_digest, "failure consumed digest")
        elif row["ledger_consumed"] is False and consumed_digest is not None:
            _fail("failure unconsumed record has a consumed digest")
    else:
        _fail("failure consumed digest policy differs")
    exit_policy = state["supervisor_exit_code_policy"]
    exit_code = row["supervisor_exit_code"]
    if exit_policy == "NULL":
        if exit_code is not None:
            _fail("failure supervisor exit code must be null")
    elif exit_policy == "ZERO":
        if type(exit_code) is not int or exit_code != 0:
            _fail("failure supervisor exit code differs")
    elif exit_policy == "NONZERO_INT":
        if type(exit_code) is not int or exit_code == 0:
            _fail("failure supervisor nonzero exit code differs")
    elif exit_policy == "NULL_OR_INT":
        if exit_code is not None and type(exit_code) is not int:
            _fail("failure supervisor nullable exit code differs")
    else:
        _fail("failure supervisor exit policy differs")
    output_policy = state["supervisor_output_digest_policy"]
    for field in ("supervisor_stdout_raw_sha256", "supervisor_stderr_raw_sha256"):
        value = row[field]
        if output_policy == "NULL":
            if value is not None:
                _fail(f"failure {field} must be null")
        elif output_policy == "RAW":
            _raw(value, f"failure {field}")
        elif output_policy == "NULL_OR_RAW":
            if value is not None:
                _raw(value, f"failure {field}")
        else:
            _fail("failure supervisor output policy differs")
    called = row["supervisor_called"]
    spawned = row["supervisor_spawn_attempted"]
    invocation_count = row["supervisor_invocation_count"]
    returned = row["supervisor_returned"]
    if called is False:
        if (
            spawned is not False
            or invocation_count != 0
            or returned is not False
            or row["supervisor_exit_code"] is not None
            or row["supervisor_stdout_raw_sha256"] is not None
            or row["supervisor_stderr_raw_sha256"] is not None
        ):
            _fail("failure prelaunch supervisor relationship differs")
    else:
        if called is not True or spawned is not True or invocation_count != 1:
            _fail("failure launched supervisor relationship differs")
        if returned is True:
            if type(row["supervisor_exit_code"]) is not int:
                _fail("returned supervisor lacks an exit code")
            _raw(row["supervisor_stdout_raw_sha256"], "returned supervisor stdout")
            _raw(row["supervisor_stderr_raw_sha256"], "returned supervisor stderr")
        elif returned is None:
            if (
                row["supervisor_exit_code"] is not None
                or row["supervisor_stdout_raw_sha256"] is not None
                or row["supervisor_stderr_raw_sha256"] is not None
            ):
                _fail("uncertain supervisor return contains exact outputs")
        else:
            _fail("called supervisor cannot be known not to return")
    if row["cleanup_complete"] is not row["survivor_absence_verified"]:
        _fail("failure cleanup and survivor relationship differs")


def validate_shadow_nonadoption_ledger_record_v1(
    record: object, kind: object, context: object, contract: object
) -> dict[str, Any]:
    frozen = _contract(contract)
    if type(kind) is not str or kind not in ("intent", "consumed", "success", "failure"):
        _fail("ledger record kind differs")
    if kind == "intent":
        fields = frozen["trial_ledger_policy"]["intent_record_schema"]["exact_fields"]
        digest_field = "record_digest"
        validator = _validate_intent
    elif kind == "consumed":
        fields = frozen["trial_ledger_policy"]["consumed_record_schema"]["exact_fields"]
        digest_field = "record_digest"
        validator = _validate_consumed
    elif kind == "success":
        fields = frozen["trial_ledger_policy"]["success_record_schema"]["exact_fields"]
        digest_field = "record_digest"
        validator = _validate_success
    else:
        fields = frozen["failure_state_policy"]["receipt_exact_fields"]
        digest_field = "receipt_digest"
        validator = _validate_failure
    row = _exact_mapping(record, fields, f"{kind} record")
    validator(row, frozen)
    context_fields = {
        "intent": ("capsule_digest", "authorization_digest"),
        "consumed": (
            "capsule_digest",
            "authorization_digest",
            "intent_record_digest",
        ),
        "success": (
            "capsule_digest",
            "authorization_digest",
            "intent_record_digest",
            "consumed_record_digest",
        ),
        "failure": (
            "capsule_digest",
            "authorization_digest",
            "intent_record_digest",
            "consumed_record_digest",
            "receipt_channel",
        ),
    }[kind]
    joined = _exact_mapping(context, context_fields, f"{kind} lineage context")
    for field in context_fields:
        if field == "receipt_channel":
            continue
        if row[field] != joined[field]:
            _fail(f"{kind} lineage differs: {field}")
    if kind == "failure":
        channel = joined["receipt_channel"]
        if channel not in ("LEDGER_RECORD", "TRANSPORT_ONLY"):
            _fail("failure receipt channel differs")
        if channel == "LEDGER_RECORD":
            if row["failure_record_publication_uncertain"] is not False:
                _fail("uncertain failure receipt cannot be a ledger record")
            if row["error_state"] == "SHADOW_OUTCOME_LINK_PUBLICATION_UNCERTAIN":
                _fail("success-publication uncertainty is transport-only")
    if row[digest_field] != _record_digest(frozen, kind, row):
        _fail(f"{kind} record digest differs")
    return deepcopy(row)
