"""History contrasts buy real evidence and retain a separate decision comparison."""

from dataclasses import replace
import json

import pytest

from hegel_machine.cognition import decision_probing
from hegel_machine.cognition.decision_probing import plan_decision_probes
from hegel_machine.cognition.evaluation import predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.uncertainty import select_decision_plan


class ResponseApparatus:
    """Actual resettable public probes; no hidden evaluation API is used."""

    def __init__(self, feedback):
        self.feedback = feedback
        self.spec = TaskSpec("anonymous_apparatus", "Respond to public controls", ("x",), (), (),
            feature_bounds={"x": (-1., 1.)}, max_probe_length=16,
            risk_quantile=.9, decision_costs=(.9, .1))
        self.paid_rows = 0

    @property
    def evaluation_metadata(self):
        raise AssertionError("not learning evidence")

    def evaluation_rows(self):
        raise AssertionError("not learning evidence")

    def observe(self, probe):
        result, history = [], []
        for controls in probe.sequence:
            target = controls["x"] + (self.feedback * history[-1]["__target__"] if history else 0.)
            result.append(Observation(dict(controls), target, tuple(history)))
            history.append({**controls, "__target__": target})
        self.paid_rows += len(result)
        return result


def policies(feedback=.5):
    x = Program("feature", name="x")
    dynamic = Program("if", children=(Program("available", lag=1),
        Program("add", children=(x, Program("mul", children=(Program("const", value=feedback),
                                                Program("lag", name="__target__", lag=1))))), x))
    return [(Candidate("static", x, "robustification"), {"kind": "nominal"}),
            (Candidate("feedback", dynamic, "continuous_state"), {"kind": "nominal"})]


def acquired(environment):
    return environment.observe(Probe("initial_a", ({"x": 0.},) * 4)) + environment.observe(
        Probe("initial_b", ({"x": 0.},) * 4))


def plan(environment, pairs=None, **kwargs):
    rows = kwargs.pop("observations", None)
    if rows is None:
        rows = acquired(environment)
    return plan_decision_probes(environment.spec, pairs or policies(), rows, 0.,
        kwargs.pop("budget", SearchBudget(500)), remaining=kwargs.pop("remaining", 40),
        round_number=3, **kwargs)


@pytest.mark.parametrize("response_feedback", [0., .5])
def test_equal_initial_predictions_are_split_by_paid_history_then_actual_representative_trials(response_feedback):
    environment = ResponseApparatus(response_feedback)
    rows, pairs = acquired(environment), policies()
    assert all(predict(pairs[0][0].program, row, 0.) == predict(pairs[1][0].program, row, 0.) for row in rows)
    result = plan(environment, observations=rows)
    assert result["observation_cost"] == 24
    assert result["simulation_compute_cost"] <= 500
    assert result["design"]["candidate_disagreement"] > 0.
    assert "gain" not in result and "expected_gain" not in result["design"]
    targeted = [environment.observe(probe) for probe in result["targeted_probes"]]
    assert targeted[0][-1].features == targeted[1][-1].features
    assert targeted[0][-1].history != targeted[1][-1].history
    right = pairs[int(response_feedback > 0)][0].program
    wrong = pairs[int(response_feedback == 0)][0].program
    assert all(abs(predict(right, block[-1], 0.) - block[-1].target) < 1e-12 for block in targeted)
    assert any(abs(predict(wrong, block[-1], 0.) - block[-1].target) > .1 for block in targeted)

    # The final choice uses new full-domain trials, never the targeted extrema.
    fresh = [row for probe in result["reference_probes"] for row in environment.observe(probe)]
    comparison_budget = SearchBudget(result["comparison_compute_cost"])
    selected, _, outcome = select_decision_plan(wrong, {"kind": "nominal"},
        right, {"kind": "nominal"}, fresh, fallback=0., costs=environment.spec.decision_costs,
        budget=comparison_budget)
    assert outcome["adopted"] and selected == right
    assert outcome["trajectories"] == 2 and outcome["gain"] > 0.
    assert environment.paid_rows == len(rows) + result["observation_cost"]
    assert comparison_budget.spent == result["comparison_compute_cost"]


def test_all_empirical_and_simulated_program_executions_are_priced(monkeypatch):
    counts = {"generator": 0, "policy": 0}
    original_predict = decision_probing.predict
    original_action = decision_probing.action_and_interval

    def measured_predict(*args, **kwargs):
        counts["generator"] += 1
        return original_predict(*args, **kwargs)

    def measured_action(*args, **kwargs):
        counts["policy"] += 1
        return original_action(*args, **kwargs)

    monkeypatch.setattr(decision_probing, "predict", measured_predict)
    monkeypatch.setattr(decision_probing, "action_and_interval", measured_action)
    budget = SearchBudget(500)
    result = plan(ResponseApparatus(.5), budget=budget)
    assert result and counts["generator"] and counts["policy"]
    assert sum(counts.values()) == budget.spent == result["simulation_compute_cost"]


def test_observation_quote_keeps_both_reference_trials_and_failure_never_calls_apparatus():
    environment = ResponseApparatus(.5)
    rows = acquired(environment)
    budget = SearchBudget(500)
    assert plan(environment, observations=rows, budget=budget, remaining=23) == {}
    assert budget.spent == 0 and environment.paid_rows == 8
    assert plan(environment, observations=rows, budget=SearchBudget(1)) == {}
    assert environment.paid_rows == 8


def test_future_response_replays_observed_innovations_and_keeps_actions_out_of_history():
    path = [Observation({"x": 0., "sensor": 3.}, 2.)]
    probe = Probe("public_control", ({"x": -1.}, {"x": 0.}))
    generator = Candidate("quantile", Program("const", value=10.), "robustification")
    reader = Candidate("reader", Program("lag", name="__target__", lag=1), "continuous_state")
    calibrated = {"kind": "global_interval", "residual_interval": [-10., 10.],
                  "action_rule": "empirical_cost_quantile", "action_offset": 100.}
    actions = decision_probing._terminal_actions(probe, generator, [-8.], path,
        [(reader, {"kind": "nominal"}), (generator, calibrated)], 0., (.9, .1))
    assert actions == [2., 110.]  # Neither 10 (quantile) nor 110 (action) became the prior response.


def test_no_history_or_identical_history_response_does_not_buy_an_experiment():
    environment = ResponseApparatus(.5)
    static = policies()[0]
    other = (Candidate("different_constant", Program("const", value=5.), "refit"), static[1])
    assert plan(environment, [static, other]) == {}
    dynamic = policies()[1]
    shifted = (dynamic[0], {"kind": "global_interval", "residual_interval": [-1., 1.],
        "action_rule": "empirical_cost_quantile", "action_offset": .8})
    assert plan(environment, [dynamic, shifted]) == {}


def test_recalled_preparation_is_executable_and_paid_and_condition_ignores_parameter_drift():
    environment = ResponseApparatus(.5)
    recalled = [Probe("experiment_old_low", ({"x": -1.},) * 4),
                Probe("experiment_old_high", ({"x": 1.},) * 4)]
    first = plan(environment, recalled_probes=recalled)
    assert {item["source_probe_id"] for item in first["design"]["preparations"]} == {
        "experiment_old_low", "experiment_old_high"}
    changed = plan(environment, policies(.6), recalled_probes=recalled)
    assert first["design"]["condition"] == changed["design"]["condition"]
    assert "anonymous_apparatus" not in json.dumps(first["design"]["condition"])
    assert json.loads(json.dumps(first["design"])) == first["design"]
    # Replaying the same pair cannot masquerade as a newly acquired contrast.
    replay = plan(environment, recalled_probes=recalled, used=first["targeted_probes"])
    assert not replay or not any(probe.sequence in [used.sequence for used in first["targeted_probes"]]
                                 for probe in replay["targeted_probes"])


def test_delayed_hypothesis_prices_a_preparation_and_reference_long_enough_to_execute_it():
    environment = ResponseApparatus(.5)
    pairs = policies()
    delayed = Program("if", children=(Program("available", lag=4),
        Program("lag", name="__target__", lag=4), Program("feature", name="x")))
    pairs[1] = Candidate("long_history", delayed, "representation"), {"kind": "nominal"}
    environment.spec = replace(environment.spec, max_probe_length=32)
    result = plan(environment, pairs, remaining=60, budget=SearchBudget(800))
    assert result
    assert [probe.cost for probe in result["reference_probes"]] == [20, 20]
    assert all(probe.cost > 4 for probe in result["targeted_probes"])
    assert result["observation_cost"] <= 60
