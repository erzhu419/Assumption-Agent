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
    custody = _load("tatqa_p18_public_source_custody_v1.json")
    assert custody["schema"] == "tatqa_p18_public_source_custody_v1"
    assert custody["self_sha256"] == _self_hash(custody)
    access = custody["access_boundary"]
    assert isinstance(access, dict)
    assert access["dataset_payload_body_open_count"] == 0
    assert access["dataset_row_parse_count"] == 0
    assert access["test_payload_open_count"] == 0
    assert access["formal_marker_or_selection_secret_created"] is False


def test_design_binds_custody_and_has_one_fixed_lifecycle() -> None:
    custody = _load("tatqa_p18_public_source_custody_v1.json")
    design = _load("tatqa_p18_typed_evaluator_study_design_v1.json")
    assert design["schema"] == "tatqa_p18_typed_evaluator_study_design_v1"
    assert design["self_sha256"] == _self_hash(design)
    source = design["source_binding"]
    assert isinstance(source, dict)
    assert source["custody_self_sha256"] == custody["self_sha256"]
    assert source["formal_files_not_opened_before_design"] is True
    lifecycle = design["lifecycle"]
    assert isinstance(lifecycle, dict)
    assert lifecycle["allowed_order"] == [
        "implementation_freeze_runtime_fingerprint_and_public_synthetic_diagnostic",
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
    design = _load("tatqa_p18_typed_evaluator_study_design_v1.json")
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
    design = _load("tatqa_p18_typed_evaluator_study_design_v1.json")
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
    design = _load("tatqa_p18_typed_evaluator_study_design_v1.json")
    execution = design["execution_contract"]
    assert isinstance(execution, dict)
    assert execution["api_or_online_evaluator_calls"] == 0
    assert execution["external_network_during_action_or_scoring"] == 0
    assert execution["retry_replay_resample_provider_switch"] == 0
    assert execution["configured_HippoRAG_process_concurrency"] == 8
    assert "dedicated" in execution["HippoRAG_executor"]
    assert "eagerly_submit_every" in execution["action_submission"]
    assert execution["scoring"] == "local_offline_exact_only"
