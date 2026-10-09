"""A purchased weak background changes the next actual duration experiment."""

from dataclasses import replace
from random import Random

from hegel_machine.cognition.condition_response_actions import research_observed_duration_experiments
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent

from test_cognition_observed_temporal_identifiability import response_program, selected_response, spec


class ActualTemporalPurchases:
    def __init__(self):
        self.random = Random(990371)
        self.program = response_program()
        self.observation_cost = self.measurement_cost = 0
        self.probes = []

    def acquire(self, channels, purpose, *, probe):
        assert channels == ("meter",)
        assert all(set(point) == {"drive", "bias"} for point in probe.sequence)
        rows, history = [], []
        for point in probe.sequence:
            features = {**point, "meter": .2 * point["bias"] + self.random.uniform(-1., 1.)}
            current = Observation(features, 0., tuple(history))
            target = self.program.predict(current)
            rows.append(replace(current, target=target))
            history.append({**features, "__target__": target})
        self.observation_cost += len(rows)
        self.measurement_cost += len(rows)
        self.probes.append(probe)
        return rows


def test_weak_paid_background_is_rewritten_before_full_response_refit():
    purchases = ActualTemporalPurchases()
    generator = Random(990372)
    training = []
    for index in range(2):
        probe = Probe(f"old_paid:{index}", tuple({name: generator.uniform(-1., 1.)
            for name in ("drive", "bias")} for _ in range(32)))
        training.extend(purchases.acquire(("meter",), "old_paid", probe=probe))
    actions = []
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7, max_candidates=8))
    budget = SearchBudget(200000)
    first = {"probe": Probe("paid_weak_background", tuple({"drive": 0., "bias": 0.}
        for _ in range(24))), "condition_field": "meter",
        "fixed_control_background": {"drive": 0., "bias": 0.}}
    rows, evidence = research_observed_duration_experiments(agent, spec(), selected_response(),
        training, [], budget, actions, acquire=purchases.acquire, channels=("meter",),
        observation_allowance=48, first_plan=first)
    experiments = [item for item in actions if item["action"] == "distinguish_paid_condition_duration"]
    assert len(experiments) == 2 and len(rows) == len(training) + 48
    assert not experiments[0]["response_supported"] and not experiments[0]["evidence"]
    assert any(item["status"] == "unidentified_condition_duration"
               for item in experiments[0]["competing_structures"])
    assert experiments[1]["response_supported"]
    assert experiments[0]["fixed_control_background"] != experiments[1]["fixed_control_background"]
    assert evidence and evidence[0]["width_candidates"] == [5]
    assert evidence[0]["placement_candidates"] == ["outside_window"]
    assert evidence[0]["loss"] < 1e-12
    assert purchases.observation_cost == purchases.measurement_cost == 112
    assert 0 < budget.spent <= 200000 - 80000


def test_an_unfunded_refit_does_not_buy_a_tail_duration_experiment():
    purchases = ActualTemporalPurchases()
    probe = Probe("old_paid", tuple({"drive": .8, "bias": 0.} for _ in range(24)))
    training = purchases.acquire(("meter",), "old_paid", probe=probe)
    budget = SearchBudget(10000)
    rows, evidence = research_observed_duration_experiments(UnifiedResearchAgent(), spec(),
        selected_response(), training, [], budget, [], acquire=purchases.acquire,
        channels=("meter",), observation_allowance=24)
    assert rows == training and not evidence
    assert purchases.observation_cost == purchases.measurement_cost == 24
    assert budget.spent == 0
