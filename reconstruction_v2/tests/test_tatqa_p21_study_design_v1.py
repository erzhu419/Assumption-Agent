from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"


def _load(name: str) -> dict[str, object]:
    value = json.loads((MANIFESTS / name).read_text(encoding="ascii"))
    assert isinstance(value, dict)
    return value


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _self_hash(value: dict[str, object]) -> str:
    body = dict(value)
    expected = body.pop("self_sha256")
    assert isinstance(expected, str)
    return _semantic_hash(body)


def test_public_custody_is_row_zero_new_and_self_hashed() -> None:
    custody = _load("tatqa_p21_public_source_custody_v1.json")
    assert custody["schema"] == "tatqa_p21_public_source_custody_v1"
    assert custody["self_sha256"] == _self_hash(custody)
    access = custody["access_boundary"]
    assert isinstance(access, dict)
    assert access["dataset_payload_body_open_count"] == 0
    assert access["dataset_row_parse_count"] == 0
    assert access["test_payload_open_count"] == 0
    assert access["formal_marker_or_selection_secret_created"] is False
    roots = custody["root_contract"]
    assert isinstance(roots, dict)
    assert roots == {
        "formal_acquisition_root_relative": (
            "artifacts/tatqa_p21_formal_v1/acquisition"
        ),
        "official_source_root_relative": (
            "artifacts/tatqa_p21_official_source_v1/TAT-QA"
        ),
        "p20_formal_or_source_root_reused": False,
    }


def test_design_binds_custody_and_preserves_one_lifecycle() -> None:
    custody = _load("tatqa_p21_public_source_custody_v1.json")
    design = _load("tatqa_p21_typed_evaluator_study_design_v1.json")
    assert design["schema"] == "tatqa_p21_typed_evaluator_study_design_v1"
    assert design["study_id"] == "TATQA_P21_TYPED_EVIDENCE_COEVOLUTION_V1"
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


def test_p20_terminal_boundary_is_exact_and_has_no_replay() -> None:
    custody = _load("tatqa_p21_public_source_custody_v1.json")
    design = _load("tatqa_p21_typed_evaluator_study_design_v1.json")
    for value in (custody, design):
        boundary = value["study_boundary"]
        assert isinstance(boundary, dict)
        assert boundary["p20_composite_public_canary_count"] == 0
        assert boundary["p20_composite_runtime_fingerprint_count"] == 0
        assert boundary["p20_formal_source_download_count"] == 0
        assert boundary["p20_formal_source_payload_open_count"] == 0
        assert boundary["p20_formal_source_row_parse_count"] == 0
        assert boundary["p20_model_inference_count"] == 0
        assert boundary["study_identity"] == "new_independent_preregistered_study"
        terminal = boundary["p20_terminal_binding"]
        assert isinstance(terminal, dict)
        assert terminal == {
            "failure_stage": "systemd_network_preflight",
            "failure_file_sha256": (
                "005b7607cdccdca1841138d063225009040446c803adeef2e9d199b159c62d19"
            ),
            "failure_self_sha256": (
                "3a7257e3f699ec0f613fc7263a8fe54ba6da07362ad15cb4343557554e5ff00c"
            ),
            "formal_source_opened": False,
            "marker_file_sha256": (
                "7eb226c8fa7e08a366db772feb952c404602943945fd6c4f2552f034baf51585"
            ),
            "marker_sha256": (
                "7ccab84057bbabcce626c0042eaabcc0f95b448923c211223b63adb003e6376c"
            ),
            "replay_retry_resume_or_root_reuse_authorized": False,
            "status": "post_inventory_environment_validation_invalid_efficacy_unknown",
        }
    assert custody["study_boundary"][
        "p20_replay_retry_or_resume_authorized"
    ] is False
    assert design["study_boundary"][
        "p20_replay_retry_resume_or_requalification_authorized"
    ] is False
    assert design["study_boundary"][
        "p20_candidate_cohort_or_efficacy_result_reused"
    ] is False


def test_blocks_candidate_metric_promotion_and_gates_are_identical_to_p20() -> None:
    p19 = _load("tatqa_p20_typed_evaluator_study_design_v1.json")
    p21 = _load("tatqa_p21_typed_evaluator_study_design_v1.json")
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
        assert p21[section] == p19[section]
    assert p21["block_contract"]["total_selected_items"] == 144
    assert p21["block_contract"]["reserve_or_backup"] is False
    assert p21["claim_contract"]["joint_primary"]["primary_count"] == 1
    assert p21["claim_contract"]["joint_primary"]["operator"] == "AND"
    assert p21["evaluator_contract"]["threshold_or_model_search"] is False

    p20_acquisition = dict(p21["acquisition_contract"])
    p20_acquisition["public_example_exclusion_binding"] = p19[
        "acquisition_contract"
    ]["public_example_exclusion_binding"]
    assert p20_acquisition == p19["acquisition_contract"]
    assert p21["offline_asset_bindings"] == p19["offline_asset_bindings"]

    p20_execution = dict(p21["execution_contract"])
    p20_execution.pop("runtime_capability_contract")
    p20_execution["runtime_qualification"] = p19["execution_contract"][
        "runtime_qualification"
    ]
    p19_execution = dict(p19["execution_contract"])
    p19_execution.pop("runtime_capability_contract")
    assert p20_execution == p19_execution

    recorded = p21["p20_design_invariance"]
    assert isinstance(recorded, dict)
    for section in invariant_sections:
        assert recorded[section] == _semantic_hash(p19[section])


def test_runtime_has_two_nested_subfingerprints_but_one_composite_qualification() -> None:
    design = _load("tatqa_p21_typed_evaluator_study_design_v1.json")
    execution = design["execution_contract"]
    assert isinstance(execution, dict)
    contract = execution["runtime_capability_contract"]
    assert isinstance(contract, dict)
    assert set(contract) == {
        "typed_plan_and_MiniLM",
        "HippoRAG",
        "independence",
        "composite_runtime_fingerprint",
        "safe_user_systemd_launch_envelope",
        "source_free_qualification",
    }
    typed = contract["typed_plan_and_MiniLM"]
    hippo = contract["HippoRAG"]
    composite = contract["composite_runtime_fingerprint"]
    qualification = contract["source_free_qualification"]
    for value in (typed, hippo, composite, qualification):
        assert isinstance(value, dict)
    assert typed["nested_subfingerprint_key"] == (
        "typed_plan_minilm_runtime_python"
    )
    assert hippo["nested_subfingerprint_key"] == "hipporag_runtime_python"
    assert composite["nested_subfingerprint_keys"] == [
        "typed_plan_minilm_runtime_python",
        "hipporag_runtime_python",
    ]
    assert composite["relative_path"] == (
        "manifests/tatqa_p21_composite_runtime_fingerprint_v1.json"
    )
    assert qualification["qualification_root_count"] == 1
    assert qualification["composite_canary_count"] == 1
    assert qualification["terminal_count"] == 1
    assert qualification["additional_effect_or_promotion_gate"] is False
    assert "qualification_root_relative" not in typed
    assert "qualification_root_relative" not in hippo


def test_safe_user_systemd_launch_envelope_is_single_explicit_and_not_a_gate() -> None:
    design = _load("tatqa_p21_typed_evaluator_study_design_v1.json")
    contract = design["execution_contract"]["runtime_capability_contract"]
    launch = contract["safe_user_systemd_launch_envelope"]
    assert isinstance(launch, dict)
    assert launch["capability_count"] == 1
    assert launch["capability_id"] == (
        "TATQA_P21_SAFE_USER_SYSTEMD_LAUNCH_ENVELOPE_V1"
    )
    assert launch["additional_effect_performance_or_promotion_gate"] is False
    assert "exact_entry_allowlist" in launch["environment_inheritance"]
    assert launch["required_user_bus_environment"] == {
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/<effective_uid>/bus",
        "XDG_RUNTIME_DIR": "/run/user/<effective_uid>",
    }
    assert launch["entry_phase_exact_outer_environment_allowlist"] == {
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
    }
    assert launch["post_runtime_inventory_phase_exact_environment_contract"][
        "CUDA_MODULE_LOADING"
    ] == "LAZY"
    assert launch["post_runtime_inventory_phase_exact_environment_contract"][
        "CUDA_VISIBLE_DEVICES"
    ] == "1"
    assert launch["post_minilm_phase_exact_environment_contract"][
        "CUDA_VISIBLE_DEVICES"
    ] == ""
    assert launch["phase_evidence_chain"]["phase_order"] == [
        "entry",
        "post_runtime_inventory",
        "post_minilm",
    ]
    assert launch["required_offline_environment"] == {
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    receipt = launch["receipt_contract"]
    assert isinstance(receipt, dict)
    assert receipt["schema"] == "tatqa_p21_user_systemd_launcher_capability_v1"
    assert receipt["raw_environment_values_or_credentials_recorded"] is False
    assert receipt["composite_fingerprint_binding_required"] is True
    assert "effective_uid_sha256" in receipt["recorded_fields"]
    assert "effective_uid" not in receipt["recorded_fields"]
    confinement = launch["systemd_network_confinement"]
    assert isinstance(confinement, dict)
    assert confinement == {
        "nested_transient_workers": {
            "IPAddressDeny": "any",
            "RestrictAddressFamilies": ["AF_UNIX"],
        },
        "outer_qualification_service": {
            "IPAddressDeny": "any",
            "RestrictAddressFamilies": ["AF_UNIX"],
        },
    }
    lifecycle = design["lifecycle"]
    assert "second_launch_envelope_or_second_runtime_qualification" in lifecycle[
        "forbidden"
    ]
    assert "launch_envelope_as_performance_promotion_or_efficacy_gate" in lifecycle[
        "forbidden"
    ]


def test_source_free_launcher_feasibility_is_not_a_qualification_or_gate() -> None:
    design = _load("tatqa_p21_typed_evaluator_study_design_v1.json")
    record = design["source_free_infrastructure_feasibility_record"]
    assert isinstance(record, dict)
    assert record["scope"] == (
        "pre_prereg_source_free_launcher_infrastructure_feasibility_only"
    )
    assert record["additional_effect_performance_promotion_or_qualification_gate"] is False
    assert record["formal_TAT_QA_source_or_rows_accessed"] is False
    assert record["model_inference_count"] == 0
    assert record["composite_runtime_fingerprint_count"] == 0
    assert record["composite_canary_count"] == 0
    assert record[
        "plaintext_user_bus_values_are_public_host_paths_not_qualification_"
        "receipt_values_or_credentials"
    ] is True
    nested = record["nested_systemd_network_preflight"]
    assert isinstance(nested, dict)
    assert nested["returncode"] == 0
    assert nested["XDG_RUNTIME_DIR"] == "/run/user/1001"
    assert nested["DBUS_SESSION_BUS_ADDRESS"] == (
        "unix:path=/run/user/1001/bus"
    )
    assert nested["stdout_sha256"] == nested["stderr_sha256"] == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_p19_static_hipporag_attestation_is_inherited_without_replay() -> None:
    design = _load("tatqa_p21_typed_evaluator_study_design_v1.json")
    inherited = design["inherited_source_free_asset_contract"]
    assert isinstance(inherited, dict)
    assert inherited == {
        "attestation_file_sha256": (
            "96479f597bbf6ae9f69998df375816db9d870634d787976513ccb5bbef173955"
        ),
        "attestation_receipt_sha256": (
            "f12863b59a83e19188ccbf35208cafdf2b7c857daf404749a58e7f7787a07618"
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
    attestation_path = MANIFESTS / inherited["attestation_relative_path"].split(
        "manifests/", 1
    )[1]
    raw = attestation_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == inherited["attestation_file_sha256"]
    value = json.loads(raw.decode("ascii"))
    body = dict(value)
    receipt = body.pop("receipt_sha256")
    assert receipt == inherited["attestation_receipt_sha256"]
    assert _semantic_hash(body) == receipt
    assert value["source_free_scope"][
        "formal_TAT_QA_source_or_rows_accessed"
    ] is False


def test_primary_and_promotion_remain_nonexpandable() -> None:
    design = _load("tatqa_p21_typed_evaluator_study_design_v1.json")
    claim = design["claim_contract"]
    assert isinstance(claim, dict)
    assert "p_at_most_0.10" in claim["A_hold_promotion"]
    primary = claim["joint_primary"]
    assert isinstance(primary, dict)
    assert set(primary) == {
        "condition_1",
        "condition_2",
        "condition_3",
        "condition_4",
        "operator",
        "primary_count",
    }
    for key in ("condition_3", "condition_4"):
        assert primary[key]["families"] == ["TABLE", "TEXT", "TABLE_TEXT"]
    lifecycle = design["lifecycle"]
    assert "additional_gate_or_runner_up_candidate" in lifecycle["forbidden"]
    freeze = design["implementation_freeze_contract"]
    assert freeze[
        "commit_and_self_hashed_manifest_required_before_any_formal_source_row_parse_or_selection_secret"
    ] is True
