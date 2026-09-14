"""Objective replacement decisions use distinct, prospective paid labels."""

import json

import pytest

from hegel_machine.cognition.objective_replacement import (
    ObjectiveReplacement, paired_replacement_evidence)


def add_rows(controller, count=4, *, start=10, incumbent=2., replacement=0.,
             goal=None, target=0.):
    for index in range(start, start + count):
        features = {"x": index / 100.}
        request = controller.prepare(features, incumbent, replacement,
                                     stage=index, candidate=goal)
        assert request is not None and "target" not in request
        controller.observe(features, target, global_index=index)


def test_replacement_waits_for_both_scope_confirmation_and_four_distinct_labels():
    controller = ObjectiveReplacement()
    controller.propose(7, 10)
    add_rows(controller, 3)
    assert controller.decide(stage=13)["replace"] is False
    add_rows(controller, 1, start=13)
    assert controller.decide(stage=14)["replace"] is False
    controller.confirm(7, 14)
    decision = controller.decide(stage=14)
    assert decision["replace"] is True
    assert decision["evidence"]["count"] == 4
    assert decision["evidence"]["mean_gain"] == 4.
    assert controller.state["scope_start"] == 7


@pytest.mark.parametrize("incumbent,replacement,expected", [
    (0., 2., "rejected"), (1., 1., "rejected")])
def test_losing_replacement_keeps_its_evidence_without_installing(incumbent,
                                                               replacement, expected):
    controller = ObjectiveReplacement()
    controller.confirm(7, 10)
    add_rows(controller, incumbent=incumbent, replacement=replacement)
    decision = controller.decide(stage=14)
    assert decision["replace"] is False
    assert decision["evidence"]["decision"] == expected
    assert len(controller.state["validation"]) == 4
    assert controller.state["confirmed"] is True


def test_positive_mean_with_wide_paired_errors_does_not_install():
    controller = ObjectiveReplacement()
    controller.confirm(7, 10)
    for index, incumbent in enumerate([0., 0., 0., 4.], 10):
        add_rows(controller, 1, start=index, incumbent=incumbent, replacement=0.)
    decision = controller.decide(stage=14)
    assert decision["evidence"]["mean_gain"] > 0
    assert decision["evidence"]["decision"] == "inconclusive"
    assert decision["replace"] is False


def test_same_scope_keeps_evidence_but_new_scope_allows_previously_compared_control():
    controller = ObjectiveReplacement()
    controller.confirm(7, 10)
    add_rows(controller, 1)
    saved = controller.to_dict()
    controller.propose(7, 11)
    assert controller.to_dict() == saved
    assert controller.prepare({"x": .1}, 2., 0., stage=11) is None
    assert len(controller.state["validation"]) == 1
    controller.propose(9, 11)
    assert controller.state["validation"] == []
    assert controller.state["confirmed"] is False
    assert len(controller.state["attempts"][0]["validation"]) == 1
    assert controller.prepare({"x": .1}, 2., 0., stage=11) is not None


def test_supported_goal_cannot_be_replaced_using_only_the_old_incumbent_advantage():
    controller = ObjectiveReplacement()
    controller.confirm(7, 10)
    add_rows(controller, goal={"tag": (4, 9), "mean": 0.}, replacement=1.)
    assert controller.decide(stage=14)["replace"] is True
    protected = controller.decide(stage=14, accepted_candidate=(4, 9))
    assert protected["evidence"]["decision"] == "supported"
    assert protected["candidate_evidence"]["decision"] == "rejected"
    assert protected["replace"] is False
    assert controller.state["comparison_rows"] == 12


def test_supported_goal_needs_four_matching_prospective_comparisons():
    controller = ObjectiveReplacement()
    controller.confirm(7, 10)
    add_rows(controller, 3, goal={"tag": (4, 9), "mean": 1.})
    add_rows(controller, 1, start=13)
    decision = controller.decide(stage=14, accepted_candidate=(4, 9))
    assert decision["evidence"]["decision"] == "supported"
    assert decision["candidate_evidence"]["count"] == 3
    assert decision["replace"] is False
    add_rows(controller, 1, start=14, goal={"tag": (4, 9), "mean": 1.})
    assert controller.decide(stage=15, accepted_candidate=(4, 9))["replace"] is True


def test_saved_pending_restores_means_and_matches_candidate_tags_after_json():
    controller = ObjectiveReplacement()
    controller.propose(7, 10)
    controller.prepare({"x": .1}, 2., 1., stage=10,
                       candidate={"tag": (4, 9), "mean": .5})
    saved = json.loads(json.dumps(controller.to_dict()))
    restored = ObjectiveReplacement(state=saved)
    assert restored.observe({"x": .2}, 0., global_index=10) is None
    assert restored.observe({"x": .1}, 0., global_index=11) is None
    row = restored.observe({"x": .1}, 0., global_index=10)
    assert (row["incumbent_error"], row["replacement_error"]) == (4., 1.)
    assert row["candidate"]["error"] == .25
    assert restored.state["pending"] is None
    assert restored.observe({"x": .1}, 0., global_index=10) is None
    assert restored.comparisons((4, 9)) == [{"features": {"x": .1},
        "global_index": 10, "target": 0., "candidate_error": .25, "previous_error": 1.}]
    assert restored.comparisons((4, 10)) == []
    assert saved["validation"] == []


def test_finished_attempt_is_archived_and_new_goal_comparisons_stay_separate():
    controller = ObjectiveReplacement()
    linked_state = controller.state
    controller.confirm(7, 10)
    add_rows(controller, 2, goal={"tag": (4, 9), "mean": 1.})
    add_rows(controller, 2, start=12, goal={"tag": (4, 11), "mean": .5})
    assert len(controller.comparisons((4, 9))) == 2
    assert len(controller.comparisons((4, 11))) == 2
    controller.decide(stage=14)
    controller.decide(stage=15)
    assert len(controller.state["decisions"]) == 1
    archived = controller.finish(15, "goal_head_installed")
    assert len(archived["validation"]) == 4
    assert controller.state["scope_start"] is None
    assert controller.state["validation"] == []
    assert controller.comparisons((4, 9)) == []
    assert controller.decide(stage=15)["replace"] is False
    assert controller.state is linked_state


def evidence_for(gains):
    return paired_replacement_evidence([
        {"previous_error": max(gain, 0.), "candidate_error": max(-gain, 0.)}
        for gain in gains])


def test_five_heterogeneous_wins_support_replacement_but_four_do_not():
    four = evidence_for([.01, .02, .03, 100.])
    assert four["mean_gain"] - four["radius"] < 0.
    assert four["decision"] == "inconclusive"
    assert four["sign_probability"] == 1 / 16
    assert four["support_basis"] is None
    five = evidence_for([.01, .02, .03, .04, 100.])
    assert five["mean_gain"] - five["radius"] < 0.
    assert five["decision"] == "supported"
    assert five["sign_probability"] == 1 / 32
    assert five["support_basis"] == "consistent_improvement"


def test_many_small_wins_cannot_hide_a_large_net_loss():
    evidence = evidence_for([.01] * 9 + [-100.])
    assert evidence["sign_probability"] <= .05
    assert evidence["mean_gain"] < 0.
    assert evidence["decision"] == "rejected"
    assert evidence["support_basis"] is None


def test_mixed_wins_and_losses_still_need_support():
    evidence = evidence_for([2., 2., 2., -1., -1.])
    assert evidence["mean_gain"] > 0.
    assert evidence["wins"] == 3 and evidence["losses"] == 2
    assert evidence["sign_probability"] == .5
    assert evidence["decision"] == "inconclusive"


def test_ties_do_not_add_sign_evidence_and_empty_comparison_is_unresolved():
    evidence = evidence_for([.01, .02, .03, 100.] + [0.] * 20)
    assert evidence["wins"] == 4 and evidence["ties"] == 20
    assert evidence["sign_probability"] == 1 / 16
    assert evidence["decision"] == "inconclusive"
    for gains in ([], [0.] * 4):
        empty = evidence_for(gains)
        assert empty["sign_probability"] == 1.
        assert empty["support_basis"] is None


def test_original_mean_margin_support_is_preserved_with_four_wins():
    evidence = evidence_for([1.] * 4)
    assert evidence["sign_probability"] > .05
    assert evidence["decision"] == "supported"
    assert evidence["support_basis"] == "original_mean_margin"


def test_consistent_improvement_rule_applies_to_both_replacement_opponents():
    controller = ObjectiveReplacement()
    controller.confirm(7, 10)
    for index, error in enumerate([.01, .02, .03, .04, 100.], 10):
        add_rows(controller, 1, start=index, incumbent=error ** .5,
                 goal={"tag": (4, 9), "mean": error ** .5})
    decision = controller.decide(stage=15, accepted_candidate=(4, 9))
    assert decision["replace"] is True
    assert decision["evidence"]["support_basis"] == "consistent_improvement"
    assert decision["candidate_evidence"]["support_basis"] == "consistent_improvement"
