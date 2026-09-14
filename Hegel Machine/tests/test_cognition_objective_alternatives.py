"""A shared challenger competes with the incumbent and supported alternatives."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.objective_alternatives import (
    prepare_interaction_prediction, supported_interaction,
)
from hegel_machine.cognition.validation_continuation import evidence_summary


def research():
    return {"candidate": {"degree": 2, "coefficients": [1., 0., 0., 0., 0.],
        "training_end": 12, "interaction_challenger": {
            "degree": 2, "coefficients": [1., 0., 0., 0., 0., 2.],
            "interactions": [[0, 1]], "training_rows": 13}}}


def validation(count=4, *, base=3., incumbent=4., interaction=1.):
    return [{"global_index": 13 + index, "features": {"x": index / 10., "y": .5},
             "candidate_error": base, "previous_error": incumbent,
             "interaction_error": interaction} for index in range(count)]


def test_prepared_cross_term_prediction_is_paid_without_mutating_candidate():
    spec = TaskSpec("shared", "Shared future evidence", ("x", "y"), (), (),
                    feature_bounds={"x": (0., 1.), "y": (0., 1.)})
    state = research()
    before = deepcopy(state)
    budget = SearchBudget(1)
    assert prepare_interaction_prediction(spec, state, {"x": .25, "y": .5}, budget) == {
        "interaction_mean": 1.25}
    assert budget.spent == 1 and state == before
    assert prepare_interaction_prediction(spec, state, {"x": .5, "y": .5}, budget) == {}
    unused = SearchBudget(1)
    assert prepare_interaction_prediction(spec, {"candidate": None}, {}, unused) == {}
    assert unused.spent == 0


@pytest.mark.parametrize("base,incumbent,expected", [(3., 4., True), (.5, 4., False), (3., .5, False)])
def test_challenger_must_beat_both_explanations_on_the_same_future_rows(base, incumbent, expected):
    state, rows = research(), validation(base=base, incumbent=incumbent)
    before = deepcopy((state, rows))
    result = supported_interaction(state, rows, 17, base_evidence=evidence_summary(rows))
    assert result["supported"] is expected
    assert result["comparison"]["indices"] == [13, 14, 15, 16]
    assert result["comparison"]["comparison_rows"] == 8
    assert result["model"]["interactions"] == [[0, 1]]
    assert result["model"]["training_end"] == 12
    assert all(row["candidate_error"] == 1. for row in result["validation"])
    assert (state, rows) == before
    result["model"]["coefficients"][0] = -100
    result["validation"][0]["features"]["x"] = -100
    assert (state, rows) == before


def test_missing_advance_predictions_do_not_supply_extra_support_after_json_restore():
    state = json.loads(json.dumps(research()))
    rows = validation(3)
    # A later GP rebase may supply a base comparison, but cannot fill the
    # challenger's missing forecast retrospectively, even if that label is easy.
    rows.append({"global_index": 16, "candidate_error": 100., "previous_error": 100.})
    assert supported_interaction(state, rows, 17, base_evidence=evidence_summary(rows)) is None
    rows.extend(validation(5)[-1:])
    result = supported_interaction(state, rows, 18, base_evidence=evidence_summary(rows))
    assert result["supported"]
    assert result["comparison"]["indices"] == [13, 14, 15, 17]
    assert len(result["validation"]) == 4


def test_positive_inconclusive_future_gain_does_not_promote_interactions():
    rows = validation()
    for row, gain in zip(rows, [4., 0., 0., 0.]):
        row["candidate_error"] = 1. + gain
        row["previous_error"] = 1. + gain
    result = supported_interaction(research(), rows, 17, base_evidence=evidence_summary(rows))
    assert not result["supported"]
    assert result["comparison"]["against_base"]["mean_gain"] > 0
    assert result["comparison"]["against_base"]["decision"] == "inconclusive"


def test_saved_future_evidence_promotes_the_only_supported_explanation():
    path = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_joint_identification_20260911/revision3/runs/"
        "goal_completion_2_center_interaction_change_hegel_joint_identification.json")
    state = json.loads(path.read_text())["objective_research"]["scope_research"]
    rows = state["validation"]
    base_evidence = evidence_summary(rows)
    before = deepcopy((state, base_evidence))
    result = supported_interaction(state, rows, 64, base_evidence=base_evidence)
    assert len(rows) == 11 and base_evidence["decision"] == "rejected"
    assert result["comparison"]["against_incumbent"]["decision"] == "supported"
    assert result["comparison"]["against_base"]["decision"] == "inconclusive"
    assert result["comparison"]["against_incumbent"]["candidate_mse"] == pytest.approx(.007133998)
    assert result["comparison"]["comparison_rows"] == 22
    assert result["comparison"]["base_evidence"] == base_evidence
    assert result["supported"]
    assert (state, base_evidence) == before


def test_two_supported_explanations_keep_base_without_resolved_head_to_head_advantage():
    rows = validation(incumbent=12.)
    for row, gain in zip(rows, [4., 0., 0., 0.]):
        row["candidate_error"] = 1. + gain
    base_evidence = evidence_summary(rows)
    assert base_evidence["decision"] == "supported"
    result = supported_interaction(research(), rows, 17, base_evidence=base_evidence)
    assert result["comparison"]["against_incumbent"]["decision"] == "supported"
    assert result["comparison"]["against_base"]["decision"] == "inconclusive"
    assert not result["supported"]
