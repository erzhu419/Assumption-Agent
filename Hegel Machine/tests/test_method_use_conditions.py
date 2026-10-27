"""Method feedback changes research admission, never a row's predictor choice."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_conditions import method_profile, method_use_decision
from hegel_machine.cognition.models import Observation, TaskSpec


METHOD = {"id": "saved-coordinate-method", "version": 2}
BASIS = {"binding": {"role_0": "u"}, "paid_cutoff": {"rows": 4},
         "fit_split": {"training_records": [0], "validation_records": [1]}}


def profile(*, longer_history=False):
    rows = [Observation({"u": value}, 2 * value, ()) for value in (-1., -.5, .5, 1.)]
    if longer_history:
        rows[-1] = Observation({"u": 1.}, 2., ({"u": -.5}, {"u": .5}))
    spec = TaskSpec("public-task", "Public controls only", ("u",), (), (),
                    feature_bounds={"u": (-1., 1.)})
    return method_profile(spec, rows, Program("feature", name="u"),
        residual_loss=.1, baseline_loss=.2, minimum_improvement=1e-10)


def feedback(*, before=.1, after=.1, status="completed", **changes):
    result = {"method_ref": deepcopy(METHOD), "profile": profile(),
        "evidence_basis": deepcopy(BASIS), "status": status,
        "feedback_ref": {"file": "paid_method_returns.json", "json_path": ["returns", 0]},
        "working_comparison": {"complete": True, "same_rows": True,
            "before_loss": before, "after_loss": after, "data_role": "exposed_development"}}
    result.update(changes)
    return result


def decide(outcomes, *, target=None, basis=None, method=None):
    return method_use_decision(method or METHOD, target or profile(), outcomes,
        evidence_basis=BASIS if basis is None else basis)


@pytest.mark.parametrize("after", [.1, .15])
def test_completed_unhelpful_visit_defers_only_its_same_paid_binding(after):
    record = feedback(after=after)
    result = decide([record])
    assert result["admission"] == "defer"
    assert result["next_action"] == "defer_until_new_evidence"
    assert result["status"] == "complete_unhelpful"
    assert result["deciding_feedback_ref"] == record["feedback_ref"]
    assert not result["scope"]["independent_support"]
    assert not result["scope"]["world_support_updated"]


@pytest.mark.parametrize("change", ["paid_evidence", "binding", "public_profile"])
def test_actual_input_change_reopens_bounded_research(change):
    basis, target = deepcopy(BASIS), profile()
    if change == "paid_evidence":
        basis["paid_cutoff"]["rows"] += 1
    elif change == "binding":
        basis["binding"]["role_0"] = "calibrated_u"
    else:
        target = profile(longer_history=True)
    result = decide([feedback()], target=target, basis=basis)
    assert result["admission"] == "bounded_exploration"
    assert result["next_action"] == "distinguish_method_use"
    assert result["question"]["parent_use"]["kind"] == "select_research_action"
    assert "ordinary_research" in result["question"]["alternatives"]


def test_local_positive_return_preserves_a_trial_without_claiming_independent_support():
    record = feedback(after=.03)
    original = deepcopy(record)
    result = decide([record])
    assert result["admission"] == "admit"
    assert result["next_action"] == "try_or_continue_method"
    assert result["status"] == "local_improvement"
    changed = deepcopy(BASIS)
    changed["paid_cutoff"]["rows"] += 1
    assert decide([record], basis=changed)["admission"] == "admit"
    assert result["scope"]["data_role"] == "exposed_development"
    assert not result["independent_support"]
    assert "selector" not in result
    assert record == original


@pytest.mark.parametrize("status", ["incomplete", "execution_error", "administrative_censored",
                                     "python_execution_budget_exhausted"])
def test_interruption_or_resource_failure_is_not_a_negative_method_example(status):
    record = feedback(after=100., status=status)
    result = decide([record])
    assert result["admission"] == "bounded_exploration"
    assert result["status"] == "unresolved_method_use"
    assert result["feedback_refs"] == []
    assert result["unresolved_feedback_refs"] == [record["feedback_ref"]]


@pytest.mark.parametrize("change", ["incomplete_comparison", "different_rows", "wrong_role",
                                     "administrative_censoring", "execution_error", "no_loss"])
def test_only_complete_uncensored_common_working_losses_can_limit_admission(change):
    record = feedback(after=.8)
    if change == "incomplete_comparison":
        record["working_comparison"]["complete"] = False
    elif change == "different_rows":
        record["working_comparison"]["same_rows"] = False
    elif change == "wrong_role":
        record["working_comparison"]["data_role"] = "fresh_confirmation"
    elif change == "no_loss":
        record["working_comparison"]["before_loss"] = None
    else:
        record[change] = True
    assert decide([record])["admission"] == "bounded_exploration"


@pytest.mark.parametrize("method", [{"id": "different-method", "version": 2},
                                    {"id": METHOD["id"], "version": 3}])
def test_a_method_or_version_does_not_inherit_another_versions_negative(method):
    result = decide([feedback()], method=method)
    assert result["admission"] == "bounded_exploration"
    assert result["feedback_refs"] == result["unresolved_feedback_refs"] == []


def test_later_unknown_work_does_not_erase_a_measured_result_or_refute_all_methods():
    negative = feedback()
    unresolved = feedback(status="incomplete")
    assert decide([negative, unresolved])["admission"] == "defer"
    positive = feedback(after=.02)
    positive["feedback_ref"]["json_path"][-1] = 1
    assert decide([negative, positive])["admission"] == "admit"
    assert decide([positive, negative])["admission"] == "defer"


def test_missing_evidence_basis_or_feedback_reference_remains_a_method_level_question():
    assert method_use_decision(METHOD, profile(), [feedback()])["admission"] == "bounded_exploration"
    assert method_use_decision(METHOD, profile(), [feedback(after=.02)])["admission"] == "bounded_exploration"
    result = decide([feedback(feedback_ref=None)])
    assert result["status"] == "unresolved_method_use"
    assert result["question"]["answer_type"] == "research_action"
    assert "declared budget" in result["question"]["required_output"]


def test_mixed_returns_without_a_current_paid_match_require_a_method_use_question():
    positive, negative = feedback(after=.02), feedback(after=.1)
    negative["evidence_basis"]["paid_cutoff"]["rows"] = 5
    negative["feedback_ref"]["json_path"][-1] = 1
    new_basis = deepcopy(BASIS)
    new_basis["paid_cutoff"]["rows"] = 6
    result = decide([positive, negative], basis=new_basis)
    assert result["admission"] == "bounded_exploration"
    assert result["reason"] == "mixed_local_returns_require_distinguishing_research_conditions"
    assert result["question"]["evidence_refs"] == [
        positive["feedback_ref"], negative["feedback_ref"]]


def paired(before=.1, after=.2):
    return {"before_loss": before, "after_loss": after, "reference_rows": 8,
        "same_rows": True, "data_role": "new_paid_readings", "execution_complete": True,
        "source_refs": [{"file": "paid_later_scores.json", "json_path": ["frozen_pair"]}]}


def test_paired_new_readings_revise_work_improvement_without_permanently_banning_method():
    record = feedback(after=.02, execution_complete=True, paired_future=paired())
    original = deepcopy(record)
    assert decide([feedback(after=.02)])["admission"] == "admit"
    result = decide([record])
    assert result["admission"] == "defer"
    assert result["deciding_return_basis"] == {"kind": "paired_future",
        "data_role": "new_paid_readings", "reference_rows": 8,
        "source_refs": record["paired_future"]["source_refs"]}
    changed = deepcopy(BASIS)
    changed["paid_cutoff"]["rows"] += 8
    assert decide([record], basis=changed)["admission"] == "bounded_exploration"
    assert not result["independent_support"] and not result["scope"]["world_support_updated"]
    assert record == original


def test_paired_improvement_can_revise_a_working_negative_as_a_bounded_trial():
    record = feedback(after=.2, execution_complete=True, paired_future=paired(after=.01))
    result = decide([record])
    assert result["admission"] == "admit"
    assert result["next_action"] == "try_or_continue_method"
    assert result["deciding_return_basis"]["kind"] == "paired_future"
    assert not result["independent_support"]


@pytest.mark.parametrize("invalid", ["different_rows", "working_rows", "missing_refs", "zero_rows",
    "nonfinite", "pair_incomplete", "method_incomplete", "pair_censored"])
def test_incomparable_later_pair_keeps_only_the_prior_local_working_return(invalid):
    record = feedback(after=.02, execution_complete=True, paired_future=paired())
    future = record["paired_future"]
    if invalid == "different_rows":
        future["same_rows"] = False
    elif invalid == "working_rows":
        future["data_role"] = "exposed_development"
    elif invalid == "missing_refs":
        future["source_refs"] = []
    elif invalid == "zero_rows":
        future["reference_rows"] = 0
    elif invalid == "nonfinite":
        future["after_loss"] = float("nan")
    elif invalid == "pair_incomplete":
        future["execution_complete"] = False
    elif invalid == "method_incomplete":
        record["execution_complete"] = False
    else:
        future["administrative_censoring"] = {"kind": "capacity_pause"}
    result = decide([record])
    if invalid == "method_incomplete":
        assert result["admission"] == "bounded_exploration"
    else:
        assert result["admission"] == "admit"
        assert result["deciding_return_basis"]["kind"] == "working_comparison"
    assert not result["independent_support"]


def test_single_predictor_confirmation_cannot_replace_a_paired_method_return():
    record = feedback(after=.02, execution_complete=True,
        confirmation={"passed": False, "mse": 100., "rows": 8})
    result = decide([record])
    assert result["admission"] == "admit"
    assert result["deciding_return_basis"]["kind"] == "working_comparison"
