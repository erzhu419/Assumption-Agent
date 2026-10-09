"""Causal and transfer checks for actual online inquiry, not log formatting."""

import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry import (
    InquiryLab, MECHANISMS, OnlineInquiryResearcher, public_candidates,
)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_recursive_and_full_action_flat_search_both_identify_the_law(mechanism):
    outcomes = []
    for mode in ("recursive", "flat"):
        lab = InquiryLab(seed=17, mechanism=mechanism)
        report = OnlineInquiryResearcher(mode=mode).run(lab)
        assert report["confirmed"]
        assert lab.blind_evaluate(report["program"], seed=919)["structural_equivalence"] is True
        assert lab.blind_evaluate(report["program"], seed=919)["mse"] <= 1e-10
        assert report["costs"]["observation_cost"] == lab.observation_cost <= 24
        outcomes.append(report)
    # A strong flat control may need fewer experiments. Recursion is not made
    # necessary by withholding an action or an expression from this control.
    assert outcomes[0]["online_subquestions"] == 2
    assert outcomes[1]["online_subquestions"] == 0
    assert len(public_candidates(lab.public_spec())) == 96


def test_live_questions_precede_acquisition_and_answers_resume_the_parent():
    lab = InquiryLab(seed=4, mechanism="absolute")
    report = OnlineInquiryResearcher().run(lab)
    events = report["events"]
    for node in report["nodes"]:
        decisions = [e for e in events if e["event"] == "experiment_selected"
                     and e["question_id"] == node["question_id"]]
        assert node["opened_at"] < decisions[0]["event_id"]
        resolutions = [e for e in events if e["event"] == "question_resolved"
                       and e["question_id"] == node["question_id"]]
        assert len(resolutions) == 1
        if node["parent_id"]:
            resume = next(e for e in events if e["event"] == "parent_resumed"
                          and e["consumes_answer"] == node["question_id"])
            assert node["resolved_at"] < resume["event_id"]
            decision = next(e for e in events if e["event"] == "experiment_selected"
                            and e["question_id"] == node["parent_id"]
                            and e["event_id"] > resume["event_id"])
            assert decision["answers_known"][node["question_id"]] == node["answer"]
    assert report["nodes"][2]["parent_id"] == report["nodes"][1]["question_id"]


def test_intervening_on_child_evidence_changes_parent_interpretation_and_test():
    class ReversedCalibrationReceipt(InquiryLab):
        def execute(self, action):
            result = super().execute(action)
            # Counterfactual intervention on the child's returned evidence only.
            # The process and subsequent sensor/response observations stay fixed.
            if action["kind"] == "calibrate":
                result["value"] *= -1
            return result

    reports = []
    for cls in (InquiryLab, ReversedCalibrationReceipt):
        lab = cls(seed=2, mechanism="linear", driver=0, orientation=1,
                  calibration_sign=1, calibration_offset=0)
        reports.append(OnlineInquiryResearcher().run(lab))
    genuine, intervened = reports
    mapping = lambda r: next(n["answer"] for n in r["nodes"] if n["goal"] == "coordinate_mapping")
    assert mapping(genuine)["orientation"] == 1
    assert mapping(intervened)["orientation"] == -1
    root_query = lambda r: next(e["action"] for e in r["events"]
                               if e["event"] == "experiment_selected"
                               and e["goal"] == "response_law")
    assert root_query(genuine) != root_query(intervened)
    assert genuine["confirmed"]
    assert not intervened["confirmed"]


def test_method_is_rewritten_before_observation_and_frozen_control_can_fallback():
    source = OnlineInquiryResearcher()
    source.run(InquiryLab(seed=0, mechanism="linear"))
    state = json.loads(json.dumps(source.to_dict()))
    reports = {}
    for mode in ("recursive", "frozen_method"):
        researcher = OnlineInquiryResearcher(mode=mode, state=state)
        lab = InquiryLab(seed=0, surface=1, mechanism="quadratic",
                         calibration_sign=-1, calibration_offset=1)
        report = researcher.run(lab)
        reports[mode] = report
        assert report["confirmed"]
        assert lab.blind_evaluate(report["program"])["structural_equivalence"] is True
        if mode == "frozen_method":
            assert researcher.to_dict()["methods"] == state["methods"]
    rewrites = [e for e in reports["recursive"]["events"] if e["event"] == "method_rewritten"]
    assert rewrites
    for rewrite in rewrites:
        assert rewrite["old_information_gain"] < rewrite["new_information_gain"]
        following = next(e for e in reports["recursive"]["events"]
                         if e["event"] == "experiment_selected" and e["event_id"] > rewrite["event_id"])
        assert following["action"] == rewrite["selected_action"]
        assert following["observations_before"] == rewrite["observations_before"]
    assert not reports["frozen_method"]["method_revisions"]
    assert reports["frozen_method"]["online_subquestions"] == reports["recursive"]["online_subquestions"]


def test_stored_hypothesis_is_called_after_surface_change_and_serializes_alone():
    source = OnlineInquiryResearcher()
    assert source.run(InquiryLab(seed=5, mechanism="quadratic"))["confirmed"]
    state = json.loads(json.dumps(source.to_dict()))
    target = OnlineInquiryResearcher(state=state)
    lab = InquiryLab(seed=5, surface=1, mechanism="quadratic", calibration_sign=-1)
    report = target.run(lab)
    assert report["confirmed"] and report["direct_call"]
    assert report["program"]["op"] == "call"
    assert len(target.library) == len(source.library)
    program = Program.from_dict(json.loads(json.dumps(report["program"])))
    assert lab.blind_evaluate(program)["structural_equivalence"] is True
    assert source.to_dict() == state


def test_research_consumes_only_public_lab_interface_and_never_evaluates():
    lab = InquiryLab(seed=8, mechanism="threshold")

    class PublicInterface:
        public_spec = lab.public_spec
        available_actions = lab.available_actions
        execute = lab.execute

    result = OnlineInquiryResearcher().run(PublicInterface())
    assert result["confirmed"]
    paid_before = lab.observation_cost
    assert lab.blind_evaluate(result["program"])["correct"]
    assert lab.observation_cost == paid_before


def test_insufficient_budget_never_reports_an_unfinished_test_as_confirmed():
    lab = InquiryLab(seed=3, mechanism="threshold")
    report = OnlineInquiryResearcher(observation_budget=3).run(lab)
    assert report["costs"]["observation_cost"] <= 3
    assert not report["confirmed"]
