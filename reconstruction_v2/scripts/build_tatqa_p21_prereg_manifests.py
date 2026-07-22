#!/usr/bin/env python3
"""Build the source-free TAT-QA P21 preregistration manifests.

P21 is an independent study after P20 terminated before fingerprint, canary,
model inference, or formal-source access.  It preserves P20's candidate,
cohort, metric, promotion, and gate contracts object-for-object.  Its only
semantic correction splits the launch contract into two phases: an exact
12-name entry environment and an exact post-inventory environment that may
add only ``CUDA_MODULE_LOADING=LAZY`` while allowing MiniLM to clear
``CUDA_VISIBLE_DEVICES``.  Nested workers still inherit only a fixed safe
subset.

The committed P19 HippoRAG attestation remains static source-free identity
evidence.  The P20 marker and terminal failure are bound only as a predecessor
boundary and never replayed.  This builder has no formal TAT-QA loader.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"
P20_DESIGN = MANIFESTS / "tatqa_p20_typed_evaluator_study_design_v1.json"
P20_CUSTODY = MANIFESTS / "tatqa_p20_public_source_custody_v1.json"
P19_HIPPORAG_ATTESTATION = (
    MANIFESTS / "tatqa_p19_hipporag_runtime_attestation_v1.json"
)
P20_TERMINAL_MARKER = (
    ROOT
    / "artifacts/tatqa_p20_runtime_qualification_v1/qualification.one_shot_marker.json"
)
P20_TERMINAL_FAILURE = (
    ROOT
    / "artifacts/tatqa_p20_runtime_qualification_v1/qualification.terminal_failure.json"
)
P21_DESIGN = MANIFESTS / "tatqa_p21_typed_evaluator_study_design_v1.json"
P21_CUSTODY = MANIFESTS / "tatqa_p21_public_source_custody_v1.json"

EXPECTED_P20_DESIGN_FILE_SHA256 = (
    "c25f8b4a84a9a041d70055110be400876b367f15d7a41dd7272ac93e42bf0e48"
)
EXPECTED_P20_DESIGN_SELF_SHA256 = (
    "f24a76a665338e608e7a8e1455f58509ee6e8073761b11506360e7b6aa4f65e5"
)
EXPECTED_P20_CUSTODY_FILE_SHA256 = (
    "7b9ab001a2a688303d3e022b9de391069a20a474dec4ed7cd7af264441090bd4"
)
EXPECTED_P20_CUSTODY_SELF_SHA256 = (
    "561f707ac00532dfd882819c5a6789e855abb3c54770a239b3713744912be302"
)
EXPECTED_P19_HIPPORAG_ATTESTATION_FILE_SHA256 = (
    "96479f597bbf6ae9f69998df375816db9d870634d787976513ccb5bbef173955"
)
EXPECTED_P19_HIPPORAG_ATTESTATION_RECEIPT_SHA256 = (
    "f12863b59a83e19188ccbf35208cafdf2b7c857daf404749a58e7f7787a07618"
)
EXPECTED_P20_TERMINAL_MARKER_FILE_SHA256 = (
    "7eb226c8fa7e08a366db772feb952c404602943945fd6c4f2552f034baf51585"
)
EXPECTED_P20_TERMINAL_MARKER_SHA256 = (
    "7ccab84057bbabcce626c0042eaabcc0f95b448923c211223b63adb003e6376c"
)
EXPECTED_P20_TERMINAL_FAILURE_FILE_SHA256 = (
    "005b7607cdccdca1841138d063225009040446c803adeef2e9d199b159c62d19"
)
EXPECTED_P20_TERMINAL_FAILURE_SELF_SHA256 = (
    "3a7257e3f699ec0f613fc7263a8fe54ba6da07362ad15cb4343557554e5ff00c"
)

INVARIANT_DESIGN_SECTIONS = (
    "action_contract",
    "block_contract",
    "canonical_evidence_contract",
    "claim_contract",
    "evaluator_contract",
    "statistical_contract",
    "structural_contrast_contract",
)


class P21ManifestBuildError(RuntimeError):
    """A committed P20 input or deterministic P21 output drifted."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def semantic_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    if "self_sha256" in body:
        raise P21ManifestBuildError("self hash must be added only after construction")
    return {**body, "self_sha256": semantic_hash(body)}


def pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("ascii")


def load_self_bound(
    path: Path, *, file_sha256: str, self_sha256: str
) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != file_sha256:
        raise P21ManifestBuildError(f"{path.name} file hash drifted")
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict):
        raise P21ManifestBuildError(f"{path.name} is not an object")
    body = dict(value)
    observed = body.pop("self_sha256", None)
    if observed != self_sha256 or semantic_hash(body) != observed:
        raise P21ManifestBuildError(f"{path.name} self hash drifted")
    return value, raw


def load_marker_bound(
    path: Path, *, file_sha256: str, marker_sha256: str
) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != file_sha256:
        raise P21ManifestBuildError(f"{path.name} file hash drifted")
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict):
        raise P21ManifestBuildError(f"{path.name} is not an object")
    body = dict(value)
    observed = body.pop("marker_sha256", None)
    if observed != marker_sha256 or semantic_hash(body) != observed:
        raise P21ManifestBuildError(f"{path.name} marker hash drifted")
    return value, raw


def load_receipt_bound(
    path: Path, *, file_sha256: str, receipt_sha256: str
) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != file_sha256:
        raise P21ManifestBuildError(f"{path.name} file hash drifted")
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict):
        raise P21ManifestBuildError(f"{path.name} is not an object")
    body = dict(value)
    observed = body.pop("receipt_sha256", None)
    if observed != receipt_sha256 or semantic_hash(body) != observed:
        raise P21ManifestBuildError(f"{path.name} receipt hash drifted")
    if value.get("schema") != "tatqa_p19_hipporag_runtime_attestation_v1":
        raise P21ManifestBuildError("P19 HippoRAG attestation schema drifted")
    scope = value.get("source_free_scope")
    if not isinstance(scope, dict) or scope.get(
        "formal_TAT_QA_source_or_rows_accessed"
    ) is not False:
        raise P21ManifestBuildError("P19 HippoRAG attestation is not source-free")
    return value, raw


def write_new_or_verify(path: Path, value: Mapping[str, Any]) -> str:
    raw = pretty_bytes(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise P21ManifestBuildError(f"refusing to overwrite drifted {path.name}")
    else:
        path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _p20_terminal_binding() -> dict[str, object]:
    return {
        "failure_stage": "systemd_network_preflight",
        "failure_file_sha256": EXPECTED_P20_TERMINAL_FAILURE_FILE_SHA256,
        "failure_self_sha256": EXPECTED_P20_TERMINAL_FAILURE_SELF_SHA256,
        "formal_source_opened": False,
        "marker_file_sha256": EXPECTED_P20_TERMINAL_MARKER_FILE_SHA256,
        "marker_sha256": EXPECTED_P20_TERMINAL_MARKER_SHA256,
        "replay_retry_resume_or_root_reuse_authorized": False,
        "status": "post_inventory_environment_validation_invalid_efficacy_unknown",
    }


def build_custody(p20: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(p20))
    value.pop("self_sha256", None)
    value["recorded_date"] = "2026-07-23"
    value["schema"] = "tatqa_p21_public_source_custody_v1"
    value["root_contract"] = {
        "formal_acquisition_root_relative": "artifacts/tatqa_p21_formal_v1/acquisition",
        "official_source_root_relative": "artifacts/tatqa_p21_official_source_v1/TAT-QA",
        "p20_formal_or_source_root_reused": False,
    }
    value["study_boundary"] = {
        "current_study_id": "TATQA_P21_TYPED_EVIDENCE_COEVOLUTION_V1",
        "new_selection_secret_required_after_p21_source_qualification": True,
        "p20_composite_public_canary_count": 0,
        "p20_composite_runtime_fingerprint_count": 0,
        "p20_formal_source_download_count": 0,
        "p20_formal_source_payload_open_count": 0,
        "p20_formal_source_row_parse_count": 0,
        "p20_model_inference_count": 0,
        "p20_replay_retry_or_resume_authorized": False,
        "p20_terminal_binding": _p20_terminal_binding(),
        "predecessor_study_id": "TATQA_P20_TYPED_EVIDENCE_COEVOLUTION_V1",
        "study_identity": "new_independent_preregistered_study",
    }
    return self_hashed(value)


def _runtime_capability_contract() -> dict[str, object]:
    return {
        "HippoRAG": {
            "capability_id": "TATQA_P21_HIPPORAG_RUNTIME_PYTHON_V1",
            "nested_subfingerprint_key": "hipporag_runtime_python",
            "nested_subfingerprint_schema": (
                "tatqa_p21_hipporag_runtime_python_subfingerprint_v1"
            ),
            "nested_subfingerprint_self_hash_field": "self_sha256",
            "permitted_capability": "official_HippoRAG_item_local_retrieve_only_worker",
            "runtime_python": (
                "dedicated_lexical_venv_bin_python_path_bound_by_file_tree_"
                "dependency_and_pyvenv_receipts"
            ),
        },
        "composite_runtime_fingerprint": {
            "cross_bindings": [
                "typed_plan_and_MiniLM_runtime_python_path_to_typed_subfingerprint",
                "HippoRAG_runtime_python_path_to_hipporag_subfingerprint",
                "both_subfingerprint_self_hashes_to_composite_self_hash",
                "composite_canary_receipts_to_both_subfingerprint_self_hashes",
                "two_phase_safe_user_systemd_launch_envelope_to_composite_self_hash",
            ],
            "nested_subfingerprint_keys": [
                "typed_plan_minilm_runtime_python",
                "hipporag_runtime_python",
            ],
            "relative_path": (
                "manifests/tatqa_p21_composite_runtime_fingerprint_v1.json"
            ),
            "schema": "tatqa_p21_composite_runtime_fingerprint_v1",
            "self_hash_field": "self_sha256",
        },
        "independence": {
            "cross_capability_dependency_satisfaction_forbidden": True,
            "nested_subfingerprints_separately_canonical_and_self_hashed": True,
            "runtime_python_lexical_paths_must_be_distinct": True,
            "shared_fallback_or_provider_switch": False,
        },
        "safe_user_systemd_launch_envelope": {
            "additional_effect_performance_or_promotion_gate": False,
            "capability_count": 1,
            "capability_id": "TATQA_P21_SAFE_USER_SYSTEMD_LAUNCH_ENVELOPE_V1",
            "environment_inheritance": (
                "deny_all_then_supply_exact_entry_allowlist_and_copy_only_fixed_"
                "safe_subset_to_nested_workers"
            ),
            "forbidden_environment": (
                "API_credentials_proxy_credentials_provider_routes_or_unlisted_variables"
            ),
            "entry_phase_exact_outer_environment_allowlist": {
                "CUDA_VISIBLE_DEVICES": "1",
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/<effective_uid>/bus",
                "HF_HUB_OFFLINE": "1",
                "HOME": "/home/erzhu419",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
                "XDG_RUNTIME_DIR": "/run/user/<effective_uid>",
            },
            "post_runtime_inventory_phase_exact_environment_contract": {
                "CUDA_MODULE_LOADING": "LAZY",
                "CUDA_VISIBLE_DEVICES": "1",
                "all_other_entry_names_and_values": "exactly_unchanged",
                "extra_name_count": 1,
                "extra_names_other_than_CUDA_MODULE_LOADING": [],
            },
            "post_minilm_phase_exact_environment_contract": {
                "CUDA_MODULE_LOADING": "LAZY",
                "CUDA_VISIBLE_DEVICES": "",
                "all_other_post_inventory_names_and_values": "exactly_unchanged",
            },
            "phase_evidence_chain": {
                "entry_and_post_runtime_inventory_receipts_bound_in": (
                    "composite_runtime_fingerprint"
                ),
                "post_minilm_receipt_bound_in": [
                    "public_production_canary",
                    "runtime_qualification_terminal",
                ],
                "phase_order": [
                    "entry",
                    "post_runtime_inventory",
                    "post_minilm",
                ],
            },
            "nested_launch_environment_contract": {
                "copied_outer_names": [
                    "DBUS_SESSION_BUS_ADDRESS",
                    "HOME",
                    "LANG",
                    "PATH",
                    "XDG_RUNTIME_DIR",
                ],
                "CUDA_MODULE_LOADING_copied": False,
                "credentials_or_unlisted_variables_copied": False,
            },
            "required_offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
            },
            "required_user_bus_environment": {
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/<effective_uid>/bus",
                "XDG_RUNTIME_DIR": "/run/user/<effective_uid>",
            },
            "receipt_contract": {
                "composite_fingerprint_binding_required": True,
                "raw_environment_values_or_credentials_recorded": False,
                "recorded_fields": [
                    "effective_uid_sha256",
                    "entry_phase_variable_name_allowlist",
                    "phase_order",
                    "phase_receipts",
                    "allowed_post_inventory_mutation",
                    "allowed_post_minilm_mutation",
                    "nested_launch_copied_variable_name_allowlist",
                    "path_address_and_socket_path_SHA256_values",
                    "socket_type_and_effective_uid_ownership_booleans",
                    "self_sha256",
                ],
                "schema": "tatqa_p21_user_systemd_launcher_capability_v1",
            },
            "systemd_network_confinement": {
                "nested_transient_workers": {
                    "IPAddressDeny": "any",
                    "RestrictAddressFamilies": ["AF_UNIX"],
                },
                "outer_qualification_service": {
                    "IPAddressDeny": "any",
                    "RestrictAddressFamilies": ["AF_UNIX"],
                },
            },
            "validation": (
                "validate_exact_12_name_entry_phase_then_exact_13_name_post_runtime_"
                "inventory_phase_with_only_CUDA_MODULE_LOADING_LAZY_added_and_"
                "CUDA_VISIBLE_DEVICES_1_then_exact_post_MiniLM_phase_with_only_"
                "CUDA_VISIBLE_DEVICES_empty_then_fail_closed_unless_user_bus_and_"
                "user_systemd_private_are_effective_uid_owned_AF_UNIX_sockets"
            ),
        },
        "source_free_qualification": {
            "additional_effect_or_promotion_gate": False,
            "composite_canary_count": 1,
            "composite_canary_relative_path": (
                "manifests/tatqa_p21_public_synthetic_production_canary_v1.json"
            ),
            "qualification_root_count": 1,
            "qualification_root_relative": (
                "artifacts/tatqa_p21_runtime_qualification_v1"
            ),
            "safe_user_systemd_launch_envelope_validation_count": 1,
            "terminal_count": 1,
            "terminal_schema": (
                "tatqa_p21_runtime_qualification_v1_terminal_success_v1"
            ),
            "validates_both_nested_subfingerprints_before_source_download": True,
        },
        "typed_plan_and_MiniLM": {
            "capability_id": "TATQA_P21_TYPED_PLAN_MINILM_RUNTIME_PYTHON_V1",
            "nested_subfingerprint_key": "typed_plan_minilm_runtime_python",
            "nested_subfingerprint_schema": (
                "tatqa_p21_typed_minilm_runtime_python_subfingerprint_v1"
            ),
            "nested_subfingerprint_self_hash_field": "self_sha256",
            "permitted_capabilities": [
                "local_Qwen_typed_plan_generation",
                "exact_offline_MiniLM_encoding",
            ],
            "runtime_python": (
                "dedicated_lexical_venv_bin_python_path_bound_by_file_tree_"
                "dependency_and_pyvenv_receipts"
            ),
        },
    }


def build_design(
    p20: Mapping[str, Any], *, custody_self_sha256: str
) -> dict[str, Any]:
    value = deepcopy(dict(p20))
    value.pop("self_sha256", None)
    value["recorded_date"] = "2026-07-23"
    value["schema"] = "tatqa_p21_typed_evaluator_study_design_v1"
    value["study_id"] = "TATQA_P21_TYPED_EVIDENCE_COEVOLUTION_V1"
    value["objective"] = (
        "one_new_real_domain_study_of_typed_candidate_expansion_and_evaluator_"
        "transition_with_a_two_phase_safe_user_systemd_launch_envelope_without_"
        "replaying_P20_or_adding_gates"
    )

    acquisition = value["acquisition_contract"]
    acquisition["public_example_exclusion_binding"] = (
        "tatqa_p21_public_source_custody_v1"
    )
    source = value["source_binding"]
    source["custody_relative_path"] = (
        "manifests/tatqa_p21_public_source_custody_v1.json"
    )
    source["custody_self_sha256"] = custody_self_sha256

    value["root_contract"] = {
        "composite_public_canary_relative": (
            "manifests/tatqa_p21_public_synthetic_production_canary_v1.json"
        ),
        "composite_runtime_fingerprint_relative": (
            "manifests/tatqa_p21_composite_runtime_fingerprint_v1.json"
        ),
        "composite_runtime_qualification_root_relative": (
            "artifacts/tatqa_p21_runtime_qualification_v1"
        ),
        "formal_root_relative": "artifacts/tatqa_p21_formal_v1",
        "implementation_freeze_relative": (
            "manifests/tatqa_p21_implementation_freeze_v1.json"
        ),
        "official_source_root_relative": (
            "artifacts/tatqa_p21_official_source_v1/TAT-QA"
        ),
        "p20_control_source_or_runtime_root_reused": False,
        "runtime_qualification_terminal_relative": (
            "artifacts/tatqa_p21_runtime_qualification_v1/"
            "qualification.terminal_success.json"
        ),
    }
    value["study_boundary"] = {
        "p20_candidate_cohort_or_efficacy_result_reused": False,
        "p20_composite_public_canary_count": 0,
        "p20_composite_runtime_fingerprint_count": 0,
        "p20_formal_source_download_count": 0,
        "p20_formal_source_payload_open_count": 0,
        "p20_formal_source_row_parse_count": 0,
        "p20_model_inference_count": 0,
        "p20_replay_retry_resume_or_requalification_authorized": False,
        "p20_terminal_binding": _p20_terminal_binding(),
        "p21_new_selection_secret_and_one_shot_roots_required": True,
        "predecessor_study_id": "TATQA_P20_TYPED_EVIDENCE_COEVOLUTION_V1",
        "study_identity": "new_independent_preregistered_study",
    }
    value["inherited_source_free_asset_contract"] = {
        "attestation_file_sha256": (
            EXPECTED_P19_HIPPORAG_ATTESTATION_FILE_SHA256
        ),
        "attestation_receipt_sha256": (
            EXPECTED_P19_HIPPORAG_ATTESTATION_RECEIPT_SHA256
        ),
        "attestation_relative_path": (
            "manifests/tatqa_p19_hipporag_runtime_attestation_v1.json"
        ),
        "formal_TAT_QA_source_or_rows_accessed": False,
        "inheritance_scope": (
            "committed_static_source_free_runtime_and_asset_identity_evidence_only"
        ),
        "p19_model_output_action_fingerprint_or_canary_replay": False,
    }
    value["source_free_infrastructure_feasibility_record"] = {
        "additional_effect_performance_promotion_or_qualification_gate": False,
        "composite_canary_count": 0,
        "composite_runtime_fingerprint_count": 0,
        "formal_TAT_QA_source_or_rows_accessed": False,
        "model_inference_count": 0,
        "nested_systemd_network_preflight": {
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus",
            "IPAddressDeny": "any",
            "RestrictAddressFamilies": ["AF_UNIX"],
            "XDG_RUNTIME_DIR": "/run/user/1001",
            "returncode": 0,
            "stderr_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
            "stdout_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
        },
        "outer_service_network_confinement": {
            "IPAddressDeny": "any",
            "RestrictAddressFamilies": ["AF_UNIX"],
        },
        "plaintext_user_bus_values_are_public_host_paths_not_qualification_"
        "receipt_values_or_credentials": True,
        "scope": "pre_prereg_source_free_launcher_infrastructure_feasibility_only",
    }

    execution = value["execution_contract"]
    execution["runtime_qualification"] = (
        "one_source_free_qualification_atomically_validates_one_two_phase_safe_"
        "user_systemd_launch_envelope_one_composite_fingerprint_containing_two_"
        "independent_runtime_Python_subfingerprints_and_one_composite_public_"
        "synthetic_canary_before_source_row_qualification_secret_or_formal_item_identity"
    )
    execution["runtime_capability_contract"] = _runtime_capability_contract()

    freeze = value["implementation_freeze_contract"]
    freeze["must_bind"] = [
        item
        for item in freeze["must_bind"]
        if "two_distinct_runtime_Python_paths" not in item
    ]
    freeze["must_bind"].extend(
        [
            "two_distinct_runtime_Python_paths_two_nested_self_hashed_"
            "subfingerprints_one_composite_fingerprint_one_qualification_"
            "terminal_and_one_composite_canary_cross_binding",
            "one_safe_user_systemd_launch_envelope_with_exact_12_name_entry_"
            "allowlist_exact_post_inventory_CUDA_MODULE_LOADING_LAZY_mutation_"
            "and_fixed_nested_subset_without_an_additional_gate",
        ]
    )
    lifecycle = value["lifecycle"]
    lifecycle["forbidden"].extend(
        [
            "P20_retry_replay_resume_requalification_or_root_reuse",
            "second_launch_envelope_or_second_runtime_qualification",
            "launch_envelope_as_performance_promotion_or_efficacy_gate",
        ]
    )

    value.pop("p19_design_invariance", None)
    value["p20_design_invariance"] = {
        name: semantic_hash(p20[name]) for name in INVARIANT_DESIGN_SECTIONS
    }
    value["p20_design_invariance"]["acquisition_cohort_contract_sha256"] = (
        semantic_hash(p20["acquisition_contract"])
    )
    value["p20_design_invariance"]["execution_change_scope"] = (
        "two_phase_launch_validation_and_P21_administrative_paths_only"
    )
    return self_hashed(value)


def main() -> None:
    p20_design, p20_design_raw = load_self_bound(
        P20_DESIGN,
        file_sha256=EXPECTED_P20_DESIGN_FILE_SHA256,
        self_sha256=EXPECTED_P20_DESIGN_SELF_SHA256,
    )
    p20_custody, p20_custody_raw = load_self_bound(
        P20_CUSTODY,
        file_sha256=EXPECTED_P20_CUSTODY_FILE_SHA256,
        self_sha256=EXPECTED_P20_CUSTODY_SELF_SHA256,
    )
    _, p19_attestation_raw = load_receipt_bound(
        P19_HIPPORAG_ATTESTATION,
        file_sha256=EXPECTED_P19_HIPPORAG_ATTESTATION_FILE_SHA256,
        receipt_sha256=EXPECTED_P19_HIPPORAG_ATTESTATION_RECEIPT_SHA256,
    )
    p20_marker, p20_marker_raw = load_marker_bound(
        P20_TERMINAL_MARKER,
        file_sha256=EXPECTED_P20_TERMINAL_MARKER_FILE_SHA256,
        marker_sha256=EXPECTED_P20_TERMINAL_MARKER_SHA256,
    )
    p20_failure, p20_failure_raw = load_self_bound(
        P20_TERMINAL_FAILURE,
        file_sha256=EXPECTED_P20_TERMINAL_FAILURE_FILE_SHA256,
        self_sha256=EXPECTED_P20_TERMINAL_FAILURE_SELF_SHA256,
    )
    if (
        p20_marker.get("formal_source_opened") is not False
        or p20_marker.get("retry_requalification") != 0
        or p20_failure.get("failure_stage") != "systemd_network_preflight"
        or p20_failure.get("formal_source_opened") is not False
        or p20_failure.get("api_or_online_evaluator_calls") != 0
    ):
        raise P21ManifestBuildError("P20 terminal source-free boundary drifted")

    p21_custody = build_custody(p20_custody)
    p21_design = build_design(
        p20_design, custody_self_sha256=p21_custody["self_sha256"]
    )
    custody_file_sha = write_new_or_verify(P21_CUSTODY, p21_custody)
    design_file_sha = write_new_or_verify(P21_DESIGN, p21_design)

    if (
        P20_DESIGN.read_bytes() != p20_design_raw
        or P20_CUSTODY.read_bytes() != p20_custody_raw
        or P19_HIPPORAG_ATTESTATION.read_bytes() != p19_attestation_raw
        or P20_TERMINAL_MARKER.read_bytes() != p20_marker_raw
        or P20_TERMINAL_FAILURE.read_bytes() != p20_failure_raw
    ):
        raise P21ManifestBuildError("input bytes changed during P21 build")
    print(
        json.dumps(
            {
                "custody_file_sha256": custody_file_sha,
                "custody_self_sha256": p21_custody["self_sha256"],
                "design_file_sha256": design_file_sha,
                "design_self_sha256": p21_design["self_sha256"],
                "formal_source_download_open_model_or_canary_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
