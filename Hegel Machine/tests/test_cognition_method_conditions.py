from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.method_conditions import method_applicability, method_profile, same_research_condition
from hegel_machine.cognition.models import Candidate, Observation, TaskSpec


def spec(name="task", features=("u",), **kwargs):
    return TaskSpec(name, "Predict the observed response", features, (), (),
                    feature_bounds={feature: (-1., 1.) for feature in features}, **kwargs)


def rows(values=(-.8, -.3, .3, .8), *, name="u", history=False, binary=False):
    result, past = [], []
    for value in values:
        target = float(value > 0) if binary else value * value
        features = {name: value}
        result.append(Observation(features, target, tuple(past) if history else ()))
        past.append({**features, "__target__": target})
    return result


def profile(*, observations=None, program=None, task=None, trace=(), residual=1., baseline=2., reference=None):
    return method_profile(task or spec(), rows() if observations is None else observations,
                          program or Program("feature", name="u"), trace,
                          residual_loss=residual, baseline_loss=baseline, reference_loss=reference)


def test_profile_reads_existing_evidence_and_preserves_program_without_predicting(monkeypatch):
    def unexpected_prediction(*args, **kwargs):
        raise AssertionError("Profiling must reuse the caller's measured losses")

    monkeypatch.setattr(Program, "predict", unexpected_prediction)
    program = Program("mul", children=(Program("const", value=2.), Program("feature", name="u")))
    value = method_profile(spec(), rows(history=True), Candidate("fit", program, "refit"),
                           residual_loss=.4, baseline_loss=2.)
    restored = json.loads(json.dumps(value))
    assert restored["observation_count"] == 4
    assert restored["representation"]["program"] == program.to_dict()
    assert restored["residual"]["normalized"] == pytest.approx(.2)
    assert restored["data"]["max_history"] == 3
    assert restored["data"]["coverage"][0]["bins"] == [0, 1, 2, 3]


def test_task_names_private_context_and_variable_renaming_do_not_determine_applicability():
    source = profile(task=replace(spec("first"), context={"hidden_family": "a"}))
    target = profile(task=replace(spec("other", ("different",)), context={"hidden_family": "b"}),
                     observations=rows(name="different"), program=Program("feature", name="different"))
    assert method_applicability(source, target, "refit")["status"] == "supported"
    assert same_research_condition(source, target, "refit")
    assert "hidden_family" not in json.dumps(target)
    assert "task_id" not in target


@pytest.mark.parametrize("program,expected", [
    (Program("feature", name="u"), "raw"),
    (Program("lag", name="u", lag=2), "history"),
    (Program("accumulate", value=.8, children=(Program("feature", name="u"), Program("const", value=0.))), "continuous"),
    (Program("window_sum", lag=3, children=(Program("feature", name="u"),)), "structure"),
])
def test_current_executable_representation_is_classified(program, expected):
    assert profile(program=program)["representation"]["kind"] == expected


def test_call_bodies_reveal_continuous_state_instead_of_looking_raw():
    body = Program("accumulate", value=.6, children=(Program("arg", name="input"), Program("const", value=0.)))
    definition = Definition("latent", ("input",), body)
    call = Program("call", name="latent", children=(Program("feature", name="u"),), definition=definition)
    value = profile(program=call)
    assert value["representation"]["kind"] == "continuous"
    assert value["representation"]["uses_calls"]
    assert value["representation"]["continuous_coordinates"] == 1


def test_appending_rows_inside_observed_support_does_not_reopen_a_failed_condition():
    initial = profile(observations=rows((-.4, -.2, .2, .4)))
    extra = rows((-.35, -.25, .25, .35))
    repeated = profile(observations=[*rows((-.4, -.2, .2, .4)), *extra], residual=1.03)
    assert repeated["observation_count"] > initial["observation_count"]
    assert same_research_condition(initial, repeated, "refit")
    changed_coverage = profile(observations=[*extra, *rows((-.9, .9))])
    assert not same_research_condition(initial, changed_coverage, "refit")
    assert not same_research_condition(initial, profile(residual=.2), "refit")
    changed_program = profile(program=Program("mul", children=(Program("feature", name="u"), Program("feature", name="u"))))
    assert not same_research_condition(initial, changed_program, "refit")


def test_old_method_without_profile_can_receive_a_trial_but_is_not_supported_transfer():
    decision = method_applicability(None, profile(), "refit")
    assert decision["status"] == "explore"
    assert decision["can_try"] and not decision["transfer_supported"]


def test_state_actions_require_actual_history_and_observed_response_support():
    historical = profile(observations=rows((-.8, -.2, .3, .7), history=True))
    assert method_applicability(historical, profile(), "continuous_state")["status"] == "unsupported"
    assert method_applicability(historical, historical, "continuous_state")["status"] == "supported"
    binary = profile(observations=rows(history=True, binary=True))
    assert method_applicability(binary, binary, "state_transition")["status"] == "supported"
    assert method_applicability(binary, historical, "state_transition")["status"] == "unsupported"
    assert method_applicability(None, binary, "state_transition")["status"] == "explore"


def test_calibration_requires_public_risk_objective_and_an_unmeasured_residual_stays_open():
    ordinary = profile()
    risk = profile(task=spec(decision_costs=(9., 1.)))
    assert method_applicability(risk, ordinary, "calibration")["status"] == "unsupported"
    assert method_applicability(risk, risk, "calibration")["status"] == "supported"
    unmeasured = method_profile(spec(), rows(), Program("feature", name="u"))
    assert method_applicability(ordinary, unmeasured, "refit")["status"] == "explore"


def test_paid_control_sequences_describe_coverage_without_readding_missing_predictor_features():
    target = spec(features=("aggregate",))
    target = replace(target, feature_bounds={"left": (-1., 1.), "right": (-1., 1.)})
    observations = rows(name="aggregate")
    trace = [{"action": "observe", "details": {"sequence": [
        {"left": -.9, "right": .9}, {"left": .9, "right": -.9}]}}]
    value = profile(task=target, observations=observations,
                    program=Program("feature", name="aggregate"), trace=trace)
    assert all(field["bins"] == [0, 3] for field in value["data"]["coverage"])
    assert all(set(row.features) == {"aggregate"} for row in observations)


def test_negative_or_neutral_feedback_blocks_same_attempt_but_new_coverage_allows_exploration():
    source = profile(observations=rows((-.4, -.2, .2, .4)))
    for gain in (-.3, 0.):
        trace = [{"action": "representation", "gain": gain, "profile": source}]
        unchanged = profile(observations=rows((-.35, -.25, .25, .35)), trace=trace)
        assert method_applicability(source, unchanged, "representation")["status"] == "unsupported"
        changed = profile(observations=rows(), trace=trace)
        assert method_applicability(source, changed, "representation")["status"] == "explore"
        assert "recent_actions" not in changed["recent_actions"][0]["condition"]
    unscoped = profile(trace=[{"action": "representation", "gain": -.3}])
    assert method_applicability(source, unscoped, "representation")["status"] == "explore"


def test_new_counterexample_reopens_an_otherwise_adequate_fit():
    source = profile()
    adequate = profile(residual=0.)
    counterexample = profile(residual=0., reference=.2)
    assert method_applicability(source, adequate, "refit")["status"] == "unsupported"
    assert method_applicability(source, counterexample, "refit")["can_try"]


def test_representation_method_can_introduce_history_before_target_has_adopted_it():
    observations = rows((-.8, -.2, .3, .7), history=True)
    source = profile(observations=observations, program=Program("lag", name="u", lag=2))
    target = profile(observations=observations)
    assert method_applicability(source, target, "representation")["status"] == "supported"
    assert method_applicability(source, target, "refit")["status"] == "explore"
    assert method_applicability(source, profile(), "representation")["status"] == "explore"


def test_unfinished_fitting_is_not_a_measured_failure_of_an_experiment():
    initial = profile(observations=rows(history=True, binary=True))
    pending = {"action": "transition", "gain": 0., "profile": initial, "execution_complete": False}
    incomplete = profile(observations=rows(history=True, binary=True), trace=[pending])
    assert incomplete["recent_actions"] == []
    assert method_applicability(initial, incomplete, "transition")["status"] == "supported"
    complete = profile(observations=rows(history=True, binary=True), trace=[{**pending, "execution_complete": True}])
    assert method_applicability(initial, complete, "transition")["status"] == "unsupported"
