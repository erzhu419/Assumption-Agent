#!/usr/bin/env python3
"""Build the source-free TAT-QA P19 preregistration manifests.

P19 preserves P18's candidate grammar, cohort sizes, labels, metrics, gates,
and exact statistical decisions.  Its only substantive execution correction
is to preregister two distinct Python capabilities, each represented by a
canonical nested subfingerprint inside one composite runtime fingerprint: one
for typed-plan generation plus MiniLM and one for official HippoRAG.  P18 ended
before any formal source download or row open and is terminal with no replay.

This builder reads only the two public P18 manifests and the source-free P19
attestation of the actual P17 HippoRAG runtime/source tree.  It has no formal
TAT-QA source path or dataset loader, refuses to overwrite non-identical output,
and verifies that every input byte is unchanged after generation.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"
P18_DESIGN = MANIFESTS / "tatqa_p18_typed_evaluator_study_design_v1.json"
P18_CUSTODY = MANIFESTS / "tatqa_p18_public_source_custody_v1.json"
P19_DESIGN = MANIFESTS / "tatqa_p19_typed_evaluator_study_design_v1.json"
P19_CUSTODY = MANIFESTS / "tatqa_p19_public_source_custody_v1.json"
P19_HIPPORAG_ATTESTATION = (
    MANIFESTS / "tatqa_p19_hipporag_runtime_attestation_v1.json"
)

EXPECTED_P18_DESIGN_FILE_SHA256 = (
    "73b237aa9d43e4eb34512f96ced9a156e6d715bd0083d24800b3c665e5992669"
)
EXPECTED_P18_DESIGN_SELF_SHA256 = (
    "48bb31c7f906676703fa8f1eff8ee9dd91100d2026dc2cbb977c752791179307"
)
EXPECTED_P18_CUSTODY_FILE_SHA256 = (
    "ae18ea8234acbd1f6ead1e53ec68868a28a3424389857d16c10add9974721f71"
)
EXPECTED_P18_CUSTODY_SELF_SHA256 = (
    "0544098eb1bad00bf559f15ab35692ae0fe0382d9c7de9ce4f2221a6d7aed6d8"
)
EXPECTED_P19_HIPPORAG_ATTESTATION_FILE_SHA256 = (
    "96479f597bbf6ae9f69998df375816db9d870634d787976513ccb5bbef173955"
)
EXPECTED_P19_HIPPORAG_ATTESTATION_RECEIPT_SHA256 = (
    "f12863b59a83e19188ccbf35208cafdf2b7c857daf404749a58e7f7787a07618"
)
EXPECTED_P19_HIPPORAG_ATTESTATION_SCHEMA = (
    "tatqa_p19_hipporag_runtime_attestation_v1"
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


class P19ManifestBuildError(RuntimeError):
    """The immutable P18 input or deterministic P19 output drifted."""


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
        raise P19ManifestBuildError("self hash must be added only after construction")
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


def load_bound(path: Path, *, file_sha256: str, self_sha256: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != file_sha256:
        raise P19ManifestBuildError(f"{path.name} file hash drifted")
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict):
        raise P19ManifestBuildError(f"{path.name} is not an object")
    body = dict(value)
    observed = body.pop("self_sha256", None)
    if observed != self_sha256 or semantic_hash(body) != observed:
        raise P19ManifestBuildError(f"{path.name} self hash drifted")
    return value, raw


def load_receipt_bound(
    path: Path, *, file_sha256: str, receipt_sha256: str, schema: str
) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != file_sha256:
        raise P19ManifestBuildError(f"{path.name} file hash drifted")
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise P19ManifestBuildError(f"{path.name} schema drifted")
    body = dict(value)
    observed = body.pop("receipt_sha256", None)
    if observed != receipt_sha256 or semantic_hash(body) != observed:
        raise P19ManifestBuildError(f"{path.name} receipt hash drifted")
    scope = value.get("source_free_scope")
    if not isinstance(scope, dict) or scope.get(
        "formal_TAT_QA_source_or_rows_accessed"
    ) is not False:
        raise P19ManifestBuildError(f"{path.name} is not source-free")
    return value, raw


def write_new_or_verify(path: Path, value: Mapping[str, Any]) -> str:
    raw = pretty_bytes(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise P19ManifestBuildError(f"refusing to overwrite drifted {path.name}")
    else:
        path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def build_custody(p18: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(p18))
    value.pop("self_sha256", None)
    value["recorded_date"] = "2026-07-23"
    value["schema"] = "tatqa_p19_public_source_custody_v1"
    value["root_contract"] = {
        "formal_acquisition_root_relative": "artifacts/tatqa_p19_formal_v1/acquisition",
        "official_source_root_relative": "artifacts/tatqa_p19_official_source_v1/TAT-QA",
        "p18_formal_or_source_root_reused": False,
    }
    value["study_boundary"] = {
        "current_study_id": "TATQA_P19_TYPED_EVIDENCE_COEVOLUTION_V1",
        "new_selection_secret_required_after_p19_source_qualification": True,
        "p18_formal_source_download_count": 0,
        "p18_formal_source_payload_open_count": 0,
        "p18_formal_source_row_parse_count": 0,
        "p18_replay_retry_or_resume_authorized": False,
        "p18_terminal_status": "source_free_runtime_inventory_terminal_invalid",
        "predecessor_study_id": "TATQA_P18_TYPED_EVIDENCE_COEVOLUTION_V1",
        "study_identity": "new_independent_preregistered_study",
    }
    return self_hashed(value)


def build_design(
    p18: Mapping[str, Any], *, custody_self_sha256: str
) -> dict[str, Any]:
    value = deepcopy(dict(p18))
    value.pop("self_sha256", None)
    value["recorded_date"] = "2026-07-23"
    value["schema"] = "tatqa_p19_typed_evaluator_study_design_v1"
    value["study_id"] = "TATQA_P19_TYPED_EVIDENCE_COEVOLUTION_V1"
    value["objective"] = (
        "one_new_real_domain_study_of_typed_candidate_expansion_and_evaluator_"
        "transition_without_replaying_P18_or_adding_gates"
    )

    acquisition = value["acquisition_contract"]
    acquisition["public_example_exclusion_binding"] = (
        "tatqa_p19_public_source_custody_v1"
    )
    source = value["source_binding"]
    source["custody_relative_path"] = (
        "manifests/tatqa_p19_public_source_custody_v1.json"
    )
    source["custody_self_sha256"] = custody_self_sha256

    offline_assets = value["offline_asset_bindings"]
    offline_assets["HippoRAG_attestation_file_sha256"] = (
        EXPECTED_P19_HIPPORAG_ATTESTATION_FILE_SHA256
    )

    value["root_contract"] = {
        "composite_public_canary_relative": (
            "manifests/tatqa_p19_public_synthetic_production_canary_v1.json"
        ),
        "composite_runtime_fingerprint_relative": (
            "manifests/tatqa_p19_composite_runtime_fingerprint_v1.json"
        ),
        "composite_runtime_qualification_root_relative": (
            "artifacts/tatqa_p19_runtime_qualification_v1"
        ),
        "formal_root_relative": "artifacts/tatqa_p19_formal_v1",
        "implementation_freeze_relative": (
            "manifests/tatqa_p19_implementation_freeze_v1.json"
        ),
        "official_source_root_relative": (
            "artifacts/tatqa_p19_official_source_v1/TAT-QA"
        ),
        "p18_control_source_or_runtime_root_reused": False,
        "runtime_qualification_terminal_relative": (
            "artifacts/tatqa_p19_runtime_qualification_v1/"
            "qualification.terminal_success.json"
        ),
    }
    value["study_boundary"] = {
        "p18_candidate_or_cohort_result_reused": False,
        "p18_formal_source_download_count": 0,
        "p18_formal_source_payload_open_count": 0,
        "p18_formal_source_row_parse_count": 0,
        "p18_replay_retry_resume_or_requalification_authorized": False,
        "p18_terminal_status": "source_free_runtime_inventory_terminal_invalid",
        "p19_new_selection_secret_and_one_shot_roots_required": True,
        "predecessor_study_id": "TATQA_P18_TYPED_EVIDENCE_COEVOLUTION_V1",
        "study_identity": "new_independent_preregistered_study",
    }

    execution = value["execution_contract"]
    execution["runtime_qualification"] = (
        "one_source_free_qualification_atomically_validates_one_composite_"
        "fingerprint_containing_two_independent_runtime_Python_"
        "subfingerprints_and_one_composite_public_synthetic_canary_before_"
        "source_row_qualification_secret_or_formal_item_identity"
    )
    execution["runtime_capability_contract"] = {
        "HippoRAG": {
            "capability_id": "TATQA_P19_HIPPORAG_RUNTIME_PYTHON_V1",
            "nested_subfingerprint_key": "hipporag_runtime_python",
            "nested_subfingerprint_schema": (
                "tatqa_p19_hipporag_runtime_python_subfingerprint_v1"
            ),
            "nested_subfingerprint_self_hash_field": "self_sha256",
            "permitted_capability": (
                "official_HippoRAG_item_local_retrieve_only_worker"
            ),
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
            ],
            "nested_subfingerprint_keys": [
                "typed_plan_minilm_runtime_python",
                "hipporag_runtime_python",
            ],
            "relative_path": (
                "manifests/tatqa_p19_composite_runtime_fingerprint_v1.json"
            ),
            "schema": "tatqa_p19_composite_runtime_fingerprint_v1",
            "self_hash_field": "self_sha256",
        },
        "independence": {
            "cross_capability_dependency_satisfaction_forbidden": True,
            "nested_subfingerprints_separately_canonical_and_self_hashed": True,
            "runtime_python_lexical_paths_must_be_distinct": True,
            "shared_fallback_or_provider_switch": False,
        },
        "source_free_qualification": {
            "additional_effect_or_promotion_gate": False,
            "composite_canary_count": 1,
            "composite_canary_relative_path": (
                "manifests/tatqa_p19_public_synthetic_production_canary_v1.json"
            ),
            "qualification_root_count": 1,
            "qualification_root_relative": (
                "artifacts/tatqa_p19_runtime_qualification_v1"
            ),
            "terminal_count": 1,
            "terminal_schema": "tatqa_p19_runtime_qualification_v1_terminal_success_v1",
            "validates_both_nested_subfingerprints_before_source_download": True,
        },
        "typed_plan_and_MiniLM": {
            "capability_id": "TATQA_P19_TYPED_PLAN_MINILM_RUNTIME_PYTHON_V1",
            "nested_subfingerprint_key": "typed_plan_minilm_runtime_python",
            "nested_subfingerprint_schema": (
                "tatqa_p19_typed_minilm_runtime_python_subfingerprint_v1"
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

    freeze = value["implementation_freeze_contract"]
    freeze["must_bind"].append(
        "two_distinct_runtime_Python_paths_two_nested_self_hashed_"
        "subfingerprints_one_composite_fingerprint_one_qualification_terminal_"
        "and_one_composite_canary_cross_binding"
    )
    lifecycle = value["lifecycle"]
    lifecycle["allowed_order"][0] = (
        "implementation_freeze_composite_runtime_fingerprint_and_public_synthetic_diagnostic"
    )
    lifecycle["forbidden"].extend(
        [
            "P18_retry_replay_resume_or_root_reuse",
            "shared_typed_minilm_and_HippoRAG_runtime_Python_capability",
        ]
    )

    value["p18_design_invariance"] = {
        name: semantic_hash(p18[name]) for name in INVARIANT_DESIGN_SECTIONS
    }
    value["p18_design_invariance"]["acquisition_cohort_contract_sha256"] = (
        semantic_hash(p18["acquisition_contract"])
    )
    value["p18_design_invariance"]["execution_change_scope"] = (
        "runtime_Python_capability_separation_and_P19_administrative_paths_only"
    )
    return self_hashed(value)


def main() -> None:
    p18_design, p18_design_raw = load_bound(
        P18_DESIGN,
        file_sha256=EXPECTED_P18_DESIGN_FILE_SHA256,
        self_sha256=EXPECTED_P18_DESIGN_SELF_SHA256,
    )
    p18_custody, p18_custody_raw = load_bound(
        P18_CUSTODY,
        file_sha256=EXPECTED_P18_CUSTODY_FILE_SHA256,
        self_sha256=EXPECTED_P18_CUSTODY_SELF_SHA256,
    )
    _, p19_hipporag_attestation_raw = load_receipt_bound(
        P19_HIPPORAG_ATTESTATION,
        file_sha256=EXPECTED_P19_HIPPORAG_ATTESTATION_FILE_SHA256,
        receipt_sha256=EXPECTED_P19_HIPPORAG_ATTESTATION_RECEIPT_SHA256,
        schema=EXPECTED_P19_HIPPORAG_ATTESTATION_SCHEMA,
    )
    p19_custody = build_custody(p18_custody)
    p19_design = build_design(
        p18_design, custody_self_sha256=p19_custody["self_sha256"]
    )

    custody_file_sha = write_new_or_verify(P19_CUSTODY, p19_custody)
    design_file_sha = write_new_or_verify(P19_DESIGN, p19_design)
    if (
        P18_DESIGN.read_bytes() != p18_design_raw
        or P18_CUSTODY.read_bytes() != p18_custody_raw
        or P19_HIPPORAG_ATTESTATION.read_bytes()
        != p19_hipporag_attestation_raw
    ):
        raise P19ManifestBuildError("input bytes changed during P19 build")
    print(
        json.dumps(
            {
                "custody_file_sha256": custody_file_sha,
                "custody_self_sha256": p19_custody["self_sha256"],
                "design_file_sha256": design_file_sha,
                "design_self_sha256": p19_design["self_sha256"],
                "formal_source_download_or_open_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
