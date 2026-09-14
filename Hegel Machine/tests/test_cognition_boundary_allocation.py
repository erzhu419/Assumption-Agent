"""A currently bracketed boundary changes the actual research allocation."""

from dataclasses import replace

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.probing import transition_probe


def bracketed_experiment():
    spec = TaskSpec("new_control", "Predict a controlled response", ("x",), (), (),
                    feature_bounds={"x": (0., 1.)}, max_probe_length=8)
    rows, history = [], []
    for x in (0., .2, .8, 1., .1, .9):
        target = float(x >= .37)
        rows.append(Observation({"x": x}, target, tuple(history)))
        history.append({"x": x, "__target__": target})
    program = Program("ge", children=(Program("feature", name="x"), Program("const", value=.5)))
    probe, details = transition_probe(spec, [Candidate("current", program, "refit")], rows,
        budget=SearchBudget(200), remaining=8, round_number=0)
    assert probe is not None and details["boundary_support"]
    return spec, rows, ActionOption("transition", probe.cost, probe.cost, (probe, details))


def test_observed_gap_gets_an_experiment_despite_successful_generic_coverage_history():
    spec, rows, targeted = bracketed_experiment()
    policy = ResearchActionPolicy(rows, spec, {})
    policy.set_condition("adequate")
    policy.update("representative", before_loss=1., after_loss=0.,
                  observation_cost=1, compute_cost=1)
    ordinary = ActionOption("representative", 1, 1)
    selected = policy.choose([ordinary, targeted], observation_remaining=16, compute_remaining=200)
    assert selected is targeted
    assert policy.last_decision["reason"] == "resolve_observed_boundary"
    assert policy.last_decision["unresolved_boundary_width"] > .1


def test_resolved_or_unaffordable_boundary_does_not_force_another_experiment():
    spec, rows, targeted = bracketed_experiment()
    policy = ResearchActionPolicy(rows, spec, {})
    ordinary = ActionOption("representative", 1, 1)
    assert policy.choose([ordinary, targeted], observation_remaining=1, compute_remaining=200) is ordinary
    probe, details = targeted.payload
    narrowed = {**details, "boundary_support": [
        {**item, "relative_width": .001} for item in details["boundary_support"]]}
    resolved = replace(targeted, payload=(probe, narrowed))
    assert policy.choose([ordinary, resolved], observation_remaining=16, compute_remaining=200) is ordinary
    assert policy.last_decision["reason"] == "untried_action"
