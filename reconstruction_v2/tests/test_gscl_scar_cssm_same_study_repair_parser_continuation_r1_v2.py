"""Source-free tests for the SCAR parser-continuation boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from assumption_agent import gscl_scar_cssm_repair_contract_v2 as contract
from assumption_agent.benchmarks import (
    gscl_scar_cssm_same_study_repair_parser_continuation_r1_v2 as subject,
)


def test_gold_bijection_accepts_order_only_change_but_prediction_contract_does_not() -> None:
    left = ("l0", "l1", "l2")
    right = ("r0", "r1", "r2")
    reordered = (("l2", "r2"), ("l0", "r0"), ("l1", "r1"))
    subject._validate_gold_bijection(
        reordered,
        left_ids=left,
        right_ids=right,
        issue_id="SCAR_REPAIR_GOLD_INVALID",
    )
    with pytest.raises(subject.SameStudyRepairDevelopmentError):
        subject._validate_full_bijection(
            reordered,
            left_ids=left,
            right_ids=right,
            issue_id="SCAR_REPAIR_S0_INVALID",
        )


@pytest.mark.parametrize(
    "invalid",
    (
        (("l0", "r0"), ("l1", "r1")),
        (("l0", "r0"), ("l0", "r1"), ("l2", "r2")),
        (("l0", "r0"), ("l1", "r1"), ("l2", "outside")),
    ),
)
def test_gold_bijection_rejects_missing_duplicate_or_wrong_endpoint(
    invalid: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises(subject.SameStudyRepairDevelopmentError) as caught:
        subject._validate_gold_bijection(
            invalid,
            left_ids=("l0", "l1", "l2"),
            right_ids=("r0", "r1", "r2"),
            issue_id="SCAR_REPAIR_GOLD_INVALID",
        )
    assert caught.value.issue_id == "SCAR_REPAIR_GOLD_INVALID"


def _one_item_private_fixture() -> tuple[dict[str, Any], dict[str, Any], str]:
    token = "item"
    left = ("l0", "l1")
    right = ("r0", "r1")
    baseline = (("l0", "r0"), ("l1", "r1"))
    swap = (("r0", "l0"), ("r1", "l1"))
    alternative = {
        "flat_structural_score": 1,
        "injective_verified": True,
        "length2_composition_verified": True,
        "length2_path_matched": 1,
        "length2_path_total": 1,
        "operator_id": "ori_keep.pol_keep.slots_identity",
        "origins": ["structure_kbest"],
        "proposal_hash": "a" * 64,
        "semantic_score": 1,
        "target_indices": [1, 0],
        "typed_incidence_matched": 1,
        "typed_incidence_total": 1,
        "typed_incidence_verified": True,
    }
    prediction = {
        "diagnostics": {
            "base": {
                "left_binder": {},
                "right_binder": {},
                "structural_diagnostics_available": True,
            }
        },
        "execution": {
            "document_call_count": 2,
            "error_code": None,
            "structural_status": "EXECUTED_WITHOUT_TYPED_FAILURE",
        },
        "item_token": token,
        "private_mechanism_receipts": {
            "availability": "COMPLETE",
            "sides": {
                "left": {"slot_graph": {"slots": {"side": "left"}}},
                "right": {"slot_graph": {"slots": {"side": "right"}}},
            },
            "variants": {
                "base": {
                    "semantic_mapping": {},
                    "structural_mapping": {"proposals": [alternative]},
                    "target_color_shuffle_mapping": {},
                }
            },
        },
        "proposal_pools": {},
        "variants": {
            "base": {
                "arms": {
                    "semantic_only": {"pairs": [list(row) for row in baseline]},
                    "full_with_length2_composition": {
                        "disposition": "ANSWER",
                        "error_code": None,
                        "pairs": [list(row) for row in baseline],
                    },
                }
            },
            "system_swap": {
                "arms": {
                    "semantic_only": {"pairs": [list(row) for row in swap]},
                    "full_with_length2_composition": {
                        "disposition": "ANSWER",
                        "error_code": None,
                        "pairs": [list(row) for row in swap],
                    },
                }
            },
        },
    }
    label = {
        "gold_pairs": {
            "base": [["l1", "r1"], ["l0", "r0"]],
            "system_swap": [["r1", "l1"], ["r0", "l0"]],
        },
        "item_token": token,
        "strata": {
            "arity": 2,
            "cohort": "primary_unique_slot",
            "domain_relation": "cross",
        },
    }
    return {"items": [prediction]}, {"items": [label]}, token


def test_private_qualification_has_no_effect_computation_or_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction_pack, label_pack, token = _one_item_private_fixture()
    output_root = tmp_path / "must_not_exist"
    archive_root = tmp_path / "archive"
    prediction_path = archive_root / "control" / "prediction_pack.private.json"
    label_path = tmp_path / "label.private.json"
    binding = {
        "append_only_runtime_contract": {
            "append_only_output_root": str(output_root)
        },
        "old_remote_roots_read_only": {
            "prepared_label_pack": str(label_path),
            "private_result_archive_root": str(archive_root),
        },
    }
    roots = {"formal_result_self_sha256": "1" * 64}
    monkeypatch.setattr(subject, "PRIMARY_ITEM_COUNT", 1)
    monkeypatch.setattr(subject, "AMBIGUOUS_ITEM_COUNT", 0)
    monkeypatch.setattr(subject, "TOTAL_ITEM_COUNT", 1)
    monkeypatch.setattr(subject, "_EXPECTED_ARITY", {2: 1})
    monkeypatch.setattr(subject, "_EXPECTED_DOMAIN_RELATION", {"cross": 1})
    monkeypatch.setattr(
        subject, "_validate_frozen_manifests", lambda **_: (binding, dict(roots))
    )
    monkeypatch.setattr(
        subject, "_validate_parser_continuation_authority", lambda *_, **__: {}
    )
    monkeypatch.setattr(
        subject, "_validate_static_implementation_closure", lambda *_, **__: {}
    )
    reads: list[Path] = []

    def read_once(path: Path, *, issue_id: str) -> tuple[dict[str, Any], str]:
        reads.append(path)
        return (
            prediction_pack if path == prediction_path else label_pack,
            "2" * 64,
        )

    monkeypatch.setattr(subject, "_read_json_once", read_once)
    monkeypatch.setattr(subject, "_validate_pack_roots", lambda *_: ("3" * 64, "4" * 64))
    monkeypatch.setattr(subject, "_validate_input_implementation_closure", lambda *_, **__: None)
    monkeypatch.setattr(subject, "_prediction_index", lambda _: {token: prediction_pack["items"][0]})
    monkeypatch.setattr(
        subject,
        "_validate_label_item",
        lambda _: (
            token,
            {
                "arity": 2,
                "cohort": "primary_unique_slot",
                "domain_relation": "cross",
            },
            label_pack["items"][0]["gold_pairs"],
        ),
    )
    monkeypatch.setattr(
        subject,
        "_validate_slots",
        lambda value, **_: ("l0", "l1") if value["side"] == "left" else ("r0", "r1"),
    )
    monkeypatch.setattr(
        subject,
        "_proposal_for_choice",
        lambda *_, **__: {"semantic_score": 2, "target_indices": [0, 1]},
    )
    monkeypatch.setattr(subject, "_binder_row", lambda _: {})
    monkeypatch.setattr(subject, "_validate_proposal", lambda value, **_: value)
    monkeypatch.setattr(
        subject.contract,
        "extract_archived_features",
        lambda _: tuple(float(index) for index in range(16)),
    )
    monkeypatch.setattr(
        subject.mechanisms,
        "build_null_package_mean",
        lambda *_: (
            SimpleNamespace(
                proposal_hash="a" * 64,
                f04_flat_structural_score_per_slot=0,
                f05_typed_incidence_match_rate=0,
                f06_typed_incidence_total_per_slot=0,
                f07_zero_incidence_support=1,
            ),
        ),
    )

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("effect or attempt path entered")

    monkeypatch.setattr(subject.contract, "pair_f1", forbidden)
    monkeypatch.setattr(subject, "_assign_stratified_folds", forbidden)
    monkeypatch.setattr(subject, "_nested_crossfit", forbidden)
    monkeypatch.setattr(subject, "_claim_single_attempt", forbidden)
    monkeypatch.setattr(subject, "_atomic_write_new", forbidden)

    result = subject.qualify_private_input_schema_only(
        prediction_pack_path=prediction_path,
        label_pack_path=label_path,
        formal_result_path=tmp_path / "formal.json",
        arm_spec_path=tmp_path / "arm.json",
        analysis_spec_path=tmp_path / "analysis.json",
        oracle_spec_path=tmp_path / "oracle.json",
        binding_path=tmp_path / "binding.json",
        continuation_amendment_path=tmp_path / "amendment.json",
    )
    assert reads == [prediction_path, label_path]
    assert result["status"] == "PASS"
    assert result["qualification_kind"] == (
        "PRIVATE_SCHEMA_QUALIFICATION_NOT_EFFECT_MEASUREMENT"
    )
    assert result["primary_item_count"] == 1
    assert result["ambiguous_item_count"] == 0
    assert result["internal_target_construction_occurred"] is False
    assert result["access_counts"]["fit"] == 0
    assert result["access_counts"]["score"] == 0
    assert not output_root.exists()


def test_qualification_cli_rejects_output_root_with_canonical_fail_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_root = tmp_path / "forbidden"
    exit_code = subject.main(
        [
            "--prediction-pack",
            str(tmp_path / "prediction.json"),
            "--label-pack",
            str(tmp_path / "label.json"),
            "--formal-result",
            str(tmp_path / "formal.json"),
            "--arm-spec",
            str(tmp_path / "arm.json"),
            "--analysis-spec",
            str(tmp_path / "analysis.json"),
            "--oracle-spec",
            str(tmp_path / "oracle.json"),
            "--binding",
            str(tmp_path / "binding.json"),
            "--continuation-amendment",
            str(tmp_path / "amendment.json"),
            "--qualify-private-schema-only",
            "--output-root",
            str(output_root),
        ]
    )
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert exit_code == 2
    assert captured.err == ""
    assert receipt["status"] == "FAIL"
    assert receipt["issue_id"] == "SCAR_REPAIR_QUALIFICATION_OUTPUT_ROOT_FORBIDDEN"
    assert receipt["access_counts"]["prediction_pack"] == 0
    assert receipt["access_counts"]["label_pack"] == 0
    assert not output_root.exists()
