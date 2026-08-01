from __future__ import annotations

import json

import pytest

from assumption_agent.benchmarks import (
    gscl_phase0_offline_qualification_v1 as qualification,
)
from assumption_agent.generalized_structural_correspondence_v1 import (
    strict_content_hash,
)


EXPECTED_CHECKS = {
    "raw_story_to_qwen_contract_test_stub",
    "concrete_frozen_scorers_and_four_arms",
    "supervisor_internal_factory_source_free_adapter",
    "frozen_scorer_authority_negative_checks",
    "qwen_formal_authority_negative_checks",
}


def test_narrative_receipt_is_double_run_exact_nonformal_and_nonready() -> None:
    first = qualification.run_narrative_source_free_qualification()
    second = qualification.run_narrative_source_free_qualification()

    assert first == second
    assert first["same_process_byte_exact_replay"] is True
    assert first["same_iterative_harness_lineage"] == (
        qualification.EXTENDED_QUALIFICATION_VERSION
    )
    assert first["formal_result"] is False
    assert first["efficacy_evidence"] is False
    assert first["new_formal_study"] is False
    assert first["effect_gate_added"] is False
    assert first["public_intrinsic_measurement"] is False
    assert first["public_intrinsic_freeze_ready"] is False
    assert first["collect_all"] is True
    assert set(first["checks"]) == EXPECTED_CHECKS
    assert set(first["declared_capability_surface"].values()) == {
        False
    }


def test_narrative_receipt_passes_or_exposes_every_blocker_fail_closed() -> None:
    receipt = qualification.run_narrative_source_free_qualification()
    failed_checks = {
        check_id
        for check_id, row in receipt["checks"].items()
        if row["status"] == "FAIL"
    }

    if failed_checks:
        assert (
            receipt["status"]
            == "FAIL_GSCL_NARRATIVE_SOURCE_FREE_QUALIFICATION"
        )
        assert receipt["issue_ids"]
        assert all(
            any(issue.startswith(f"{check_id}.") for issue in receipt["issue_ids"])
            for check_id in failed_checks
        )
    else:
        assert (
            receipt["status"]
            == "PASS_GSCL_NARRATIVE_SOURCE_FREE_QUALIFICATION"
        )
        assert receipt["issue_ids"] == []


def test_narrative_receipt_hashes_recompute_and_omit_item_content() -> None:
    receipt = qualification.run_narrative_source_free_qualification()
    without_self = dict(receipt)
    self_hash = without_self.pop("self_hash")
    encoded = json.dumps(receipt, sort_keys=True)

    assert strict_content_hash(without_self) == self_hash
    assert (
        strict_content_hash(receipt["issue_ids"])
        == receipt["issue_commitment"]
    )
    for forbidden in (
        "Aster",
        "Birch",
        "Cedar",
        "Dune",
        "Ember",
        "Fjord",
        "opaque_item_id",
        "predicted_ordinal",
        "query_narrative",
        "first_choice",
        "second_choice",
        "by_arm",
    ):
        assert forbidden not in encoded


def test_narrative_matrix_collects_all_after_independent_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def check(check_id: str, *, fail: bool = False):
        def run() -> dict[str, object]:
            calls.append(check_id)
            if fail:
                raise RuntimeError("synthetic_failure")
            return {"status": "PASS"}

        return run

    monkeypatch.setattr(
        qualification,
        "_qualify_narrative_extractor_contract",
        check("extractor"),
    )
    monkeypatch.setattr(
        qualification,
        "_qualify_narrative_scorers_and_arms",
        check("scorers", fail=True),
    )
    monkeypatch.setattr(
        qualification,
        "_qualify_narrative_internal_factory_adapter",
        check("factory"),
    )
    monkeypatch.setattr(
        qualification,
        "_qualify_frozen_scorer_authority_negatives",
        check("scorer_negatives"),
    )
    monkeypatch.setattr(
        qualification,
        "_qualify_qwen_formal_authority_negatives",
        check("qwen_negatives"),
    )

    rows, issues = qualification._run_narrative_source_free_matrix()

    assert calls == [
        "extractor",
        "scorers",
        "factory",
        "scorer_negatives",
        "qwen_negatives",
    ]
    assert set(rows) == EXPECTED_CHECKS
    assert rows["concrete_frozen_scorers_and_four_arms"]["status"] == "FAIL"
    assert len(issues) == 1
    assert issues[0].startswith(
        "concrete_frozen_scorers_and_four_arms.RuntimeError."
    )
