from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"


def _load(name: str) -> dict[str, object]:
    value = json.loads((MANIFESTS / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _self_hash(value: dict[str, object]) -> str:
    body = dict(value)
    expected = body.pop("self_sha256")
    raw = json.dumps(
        body,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert isinstance(expected, str)
    return hashlib.sha256(raw).hexdigest()


def test_public_custody_is_row_zero_and_self_hashed() -> None:
    custody = _load("tatqa_p19_public_source_custody_v1.json")
    assert custody["schema"] == "tatqa_p19_public_source_custody_v1"
    assert custody["self_sha256"] == _self_hash(custody)
    access = custody["access_boundary"]
    assert isinstance(access, dict)
    assert access["dataset_payload_body_open_count"] == 0
    assert access["dataset_row_parse_count"] == 0
    assert access["test_payload_open_count"] == 0
    assert access["formal_marker_or_selection_secret_created"] is False


def test_design_binds_custody_and_has_one_fixed_lifecycle() -> None:
    custody = _load("tatqa_p19_public_source_custody_v1.json")
    design = _load("tatqa_p19_typed_evaluator_study_design_v1.json")
    assert design["schema"] == "tatqa_p19_typed_evaluator_study_design_v1"
    assert design["self_sha256"] == _self_hash(design)
    source = design["source_binding"]
    assert isinstance(source, dict)
    assert source["custody_self_sha256"] == custody["self_sha256"]
    assert source["formal_files_not_opened_before_design"] is True
    lifecycle = design["lifecycle"]
    assert isinstance(lifecycle, dict)
    assert lifecycle["allowed_order"] == [
        "implementation_freeze_composite_runtime_fingerprint_and_public_synthetic_diagnostic",
        "source_download_and_aggregate_qualification",
        "one_shot_acquisition",
        "A_form_action_and_E1_fit",
        "F_search_policy_freeze",
        "A_hold_action_and_promotion",
        "conditional_epoch_transition",
        "conditional_M_search",
        "terminal_disposition",
    ]


def test_blocks_are_fixed_balanced_and_have_no_rescue() -> None:
    design = _load("tatqa_p19_typed_evaluator_study_design_v1.json")
    blocks = design["block_contract"]
    assert isinstance(blocks, dict)
    expected = {
        "A_form": (48, 16),
        "F_search": (36, 12),
        "A_hold": (30, 10),
        "M_search": (30, 10),
    }
    for block, (total, per_family) in expected.items():
        value = blocks[block]
        assert isinstance(value, dict)
        assert value["total"] == total
        assert value["per_family"] == per_family
    assert blocks["total_selected_items"] == 144
    assert blocks["reserve_or_backup"] is False
    acquisition = design["acquisition_contract"]
    assert isinstance(acquisition, dict)
    assert acquisition["family_order"] == ["TABLE", "TEXT", "TABLE_TEXT"]
    assert acquisition["test_split_opened"] is False
    evidence = design["canonical_evidence_contract"]
    assert isinstance(evidence, dict)
    qualification = evidence["qualification"]
    assert isinstance(qualification, dict)
    assert qualification["minimum_canonical_unit_count"] == 5
    assert qualification["maximum_canonical_unit_count"] == 96


def test_primary_and_promotion_are_not_expandable_gates() -> None:
    design = _load("tatqa_p19_typed_evaluator_study_design_v1.json")
    claim = design["claim_contract"]
    assert isinstance(claim, dict)
    assert "p_at_most_0.10" in claim["A_hold_promotion"]
    primary = claim["joint_primary"]
    assert isinstance(primary, dict)
    assert primary["operator"] == "AND"
    assert primary["primary_count"] == 1
    assert set(primary) == {
        "condition_1",
        "condition_2",
        "condition_3",
        "condition_4",
        "operator",
        "primary_count",
    }
    for key in ("condition_3", "condition_4"):
        condition = primary[key]
        assert isinstance(condition, dict)
        assert condition["families"] == ["TABLE", "TEXT", "TABLE_TEXT"]
    evaluator = design["evaluator_contract"]
    assert isinstance(evaluator, dict)
    assert evaluator["threshold_or_model_search"] is False
    lifecycle = design["lifecycle"]
    assert isinstance(lifecycle, dict)
    assert "additional_gate_or_runner_up_candidate" in lifecycle["forbidden"]
    freeze = design["implementation_freeze_contract"]
    assert isinstance(freeze, dict)
    assert freeze[
        "commit_and_self_hashed_manifest_required_before_any_formal_source_row_parse_or_selection_secret"
    ] is True


def test_execution_is_offline_eager_and_uses_a_dedicated_hippo_cap() -> None:
    design = _load("tatqa_p19_typed_evaluator_study_design_v1.json")
    execution = design["execution_contract"]
    assert isinstance(execution, dict)
    assert execution["api_or_online_evaluator_calls"] == 0
    assert execution["external_network_during_action_or_scoring"] == 0
    assert execution["retry_replay_resample_provider_switch"] == 0
    assert execution["configured_HippoRAG_process_concurrency"] == 8
    assert "dedicated" in execution["HippoRAG_executor"]
    assert "eagerly_submit_every" in execution["action_submission"]
    assert execution["scoring"] == "local_offline_exact_only"


def test_p19_is_new_and_p18_is_terminal_without_replay_or_root_reuse() -> None:
    custody = _load("tatqa_p19_public_source_custody_v1.json")
    design = _load("tatqa_p19_typed_evaluator_study_design_v1.json")
    assert design["study_id"] == "TATQA_P19_TYPED_EVIDENCE_COEVOLUTION_V1"
    for value in (custody, design):
        boundary = value["study_boundary"]
        assert isinstance(boundary, dict)
        assert boundary["p18_terminal_status"] == (
            "source_free_runtime_inventory_terminal_invalid"
        )
        assert boundary["p18_formal_source_download_count"] == 0
        assert boundary["p18_formal_source_payload_open_count"] == 0
        assert boundary["p18_formal_source_row_parse_count"] == 0
        assert boundary["study_identity"] == "new_independent_preregistered_study"
    assert custody["study_boundary"]["p18_replay_retry_or_resume_authorized"] is False
    assert design["study_boundary"][
        "p18_replay_retry_resume_or_requalification_authorized"
    ] is False
    assert design["study_boundary"]["p18_candidate_or_cohort_result_reused"] is False
    custody_roots = custody["root_contract"]
    design_roots = design["root_contract"]
    assert isinstance(custody_roots, dict) and isinstance(design_roots, dict)
    assert custody_roots["p18_formal_or_source_root_reused"] is False
    assert design_roots["p18_control_source_or_runtime_root_reused"] is False
    assert all(
        "tatqa_p19" in value
        for value in (*custody_roots.values(), *design_roots.values())
        if isinstance(value, str)
    )


def test_two_runtime_python_capabilities_are_independent_and_fingerprinted() -> None:
    design = _load("tatqa_p19_typed_evaluator_study_design_v1.json")
    execution = design["execution_contract"]
    assert isinstance(execution, dict)
    capabilities = execution["runtime_capability_contract"]
    assert isinstance(capabilities, dict)
    assert set(capabilities) == {
        "typed_plan_and_MiniLM",
        "HippoRAG",
        "independence",
        "composite_runtime_fingerprint",
        "source_free_qualification",
    }
    typed = capabilities["typed_plan_and_MiniLM"]
    hippo = capabilities["HippoRAG"]
    independence = capabilities["independence"]
    composite = capabilities["composite_runtime_fingerprint"]
    qualification = capabilities["source_free_qualification"]
    for row in (typed, hippo, independence, composite, qualification):
        assert isinstance(row, dict)
    assert typed["capability_id"] != hippo["capability_id"]
    assert typed["nested_subfingerprint_key"] == (
        "typed_plan_minilm_runtime_python"
    )
    assert hippo["nested_subfingerprint_key"] == "hipporag_runtime_python"
    assert typed["nested_subfingerprint_schema"] == (
        "tatqa_p19_typed_minilm_runtime_python_subfingerprint_v1"
    )
    assert hippo["nested_subfingerprint_schema"] == (
        "tatqa_p19_hipporag_runtime_python_subfingerprint_v1"
    )
    assert typed["nested_subfingerprint_self_hash_field"] == "self_sha256"
    assert hippo["nested_subfingerprint_self_hash_field"] == "self_sha256"
    assert "fingerprint_relative_path" not in typed
    assert "fingerprint_relative_path" not in hippo
    assert "qualification_root_relative" not in typed
    assert "qualification_root_relative" not in hippo
    assert composite == {
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
    }
    assert independence == {
        "cross_capability_dependency_satisfaction_forbidden": True,
        "nested_subfingerprints_separately_canonical_and_self_hashed": True,
        "runtime_python_lexical_paths_must_be_distinct": True,
        "shared_fallback_or_provider_switch": False,
    }
    assert qualification == {
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
        "terminal_schema": (
            "tatqa_p19_runtime_qualification_v1_terminal_success_v1"
        ),
        "validates_both_nested_subfingerprints_before_source_download": True,
    }


def test_candidate_cohort_gate_metric_and_statistics_are_identical_to_p18() -> None:
    p18 = _load("tatqa_p18_typed_evaluator_study_design_v1.json")
    p19 = _load("tatqa_p19_typed_evaluator_study_design_v1.json")
    invariant_sections = (
        "action_contract",
        "block_contract",
        "canonical_evidence_contract",
        "claim_contract",
        "evaluator_contract",
        "statistical_contract",
        "structural_contrast_contract",
    )
    for section in invariant_sections:
        assert p19[section] == p18[section]

    p19_acquisition = dict(p19["acquisition_contract"])
    p19_acquisition["public_example_exclusion_binding"] = (
        p18["acquisition_contract"]["public_example_exclusion_binding"]
    )
    assert p19_acquisition == p18["acquisition_contract"]
    p19_assets = dict(p19["offline_asset_bindings"])
    p18_assets = dict(p18["offline_asset_bindings"])
    p19_hippo_attestation_hash = p19_assets.pop(
        "HippoRAG_attestation_file_sha256"
    )
    p18_hippo_attestation_hash = p18_assets.pop(
        "HippoRAG_attestation_file_sha256"
    )
    assert p19_assets == p18_assets
    assert p19_hippo_attestation_hash == (
        "96479f597bbf6ae9f69998df375816db9d870634d787976513ccb5bbef173955"
    )
    assert p19_hippo_attestation_hash != p18_hippo_attestation_hash

    attestation_path = MANIFESTS / "tatqa_p19_hipporag_runtime_attestation_v1.json"
    attestation_raw = attestation_path.read_bytes()
    assert hashlib.sha256(attestation_raw).hexdigest() == p19_hippo_attestation_hash
    attestation = json.loads(attestation_raw.decode("ascii"))
    assert attestation["schema"] == "tatqa_p19_hipporag_runtime_attestation_v1"
    assert attestation["source_free_scope"][
        "formal_TAT_QA_source_or_rows_accessed"
    ] is False
    attestation_body = dict(attestation)
    receipt = attestation_body.pop("receipt_sha256")
    assert receipt == (
        "f12863b59a83e19188ccbf35208cafdf2b7c857daf404749a58e7f7787a07618"
    )
    assert hashlib.sha256(
        json.dumps(
            attestation_body,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest() == receipt

    p19_execution = dict(p19["execution_contract"])
    p19_execution.pop("runtime_capability_contract")
    p19_execution["runtime_qualification"] = p18["execution_contract"][
        "runtime_qualification"
    ]
    assert p19_execution == p18["execution_contract"]

    recorded = p19["p18_design_invariance"]
    assert isinstance(recorded, dict)
    for section in invariant_sections:
        expected = hashlib.sha256(
            json.dumps(
                p18[section],
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        assert recorded[section] == expected
