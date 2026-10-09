"""A failed local temporal fit can propose a complete response experiment."""

from dataclasses import replace
from random import Random

from hegel_machine.cognition.condition_response_actions import (
    _observed_duration_nominations, _observed_response_claims,
    research_observed_duration_experiments)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.observed_condition_duration import propose_observed_condition_duration
from hegel_machine.cognition.unified_research import UnifiedResearchAgent

from test_cognition_observed_temporal_identifiability import feature, spec


# Actual reading support from the retained W7 regression. The experiment can
# set its two commands; these readings are purchased, not assigned commands.
READINGS = (-.39250971031013243, -.8316118417450409, .1353187380052368,
    .660584751988956, -.8057262101597567, -.7091059782971274, -1.1097435223916092,
    .5384306021440199, .10357260659460393, .5030740225868873, -.35064914414547216,
    .5502568236147825, -.6283326153840365, -.16678207120203353, .23314221722066947,
    -.48018363426875704, -.5944577978693697, -1.1227252495857039,
    .04314605478279271, -.7512754062745832, .6420422217243429,
    -.06493353320645306, -.1466337524331362, -.2312338936890263)


def physical_response(*, conditional=True):
    first, second = [Program("add", children=(Program("mul", children=(
        Program("const", value=sign * 2.), feature("drive"))),
        Program("mul", children=(Program("const", value=.25), feature("meter")))))
        for sign in (1., -1.)]
    response = (Program("if", children=(Program("ge", children=(feature("meter"),
        Program("const", value=-.02))), first, second)) if conditional else first)
    return Program("window_sum", lag=7, children=(response,))


class MeasuredTemporalResponse:
    def __init__(self, *, conditional=True):
        self.program = physical_response(conditional=conditional)
        self.random = Random(9915908)
        self.observations = self.measurements = 0

    def acquire(self, channels, purpose, *, probe):
        assert channels == ("meter",)
        rows, history = [], []
        for index, point in enumerate(probe.sequence):
            assert set(point) == {"drive", "bias"}
            assert all(-1. <= value <= 1. for value in point.values())
            reading = (READINGS[index] if probe.probe_id == "purchased_background"
                       else .2 * point["bias"] + self.random.uniform(-1., 1.))
            row = Observation({**point, "meter": reading}, 0., tuple(history))
            target = self.program.predict(row)
            rows.append(replace(row, target=target))
            history.append({**row.features, "__target__": target})
        self.observations += len(rows)
        self.measurements += len(rows)
        return rows

    def varied(self, label, seed):
        random = Random(seed)
        probe = Probe(label, tuple({name: random.uniform(-1., 1.)
            for name in ("drive", "bias")} for _ in range(32)))
        return self.acquire(("meter",), "varied_response", probe=probe)


def test_unresolved_nomination_refits_a_complete_response_with_a_101k_budget():
    lab = MeasuredTemporalResponse()
    training = [*lab.varied("training0", 61), *lab.varied("training1", 62)]
    validation = lab.varied("unfitted_validation", 63)
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7, max_candidates=8))
    selected = {"candidate": Candidate("retained", feature("drive"), "research")}
    plan = {"probe": Probe("purchased_background", tuple({"drive": 1., "bias": 0.}
        for _ in range(24))), "condition_field": "meter",
        "fixed_control_background": {"drive": 1., "bias": 0.}}
    actions, budget = [], SearchBudget(101000)
    rows, evidence = research_observed_duration_experiments(agent, spec(), selected,
        training, [], budget, actions, acquire=lab.acquire, channels=("meter",),
        observation_allowance=24, validation_rows=len(validation), first_plan=plan)
    assert not evidence
    experiment = next(item for item in actions if item["action"] == "distinguish_paid_condition_duration")
    assert not experiment["response_supported"]
    assert {trial["width"] for trial in experiment["competing_structures"][0]["parameter_trials"]} == set(range(1, 25))
    nominations = _observed_duration_nominations(actions, 1e-7)
    assert nominations[0]["width"] == 7 and nominations[0]["placement"] == "inside_window"
    assert nominations[0]["status"] == "unresolved_duration" and nominations[0]["loss"] > 1e-7
    assert all("parameter_trials" not in item for item in nominations)
    claims = _observed_response_claims(agent, spec(), rows, selected, budget, actions, validation, evidence)
    best = min(claims, key=lambda item: max(item["training_loss"], item["validation_loss"]))
    assert max(best["training_loss"], best["validation_loss"]) < 1e-7
    step = best["research_program"]["selection_steps"][-1]
    assert step["width"] == 7 and step["placement"] == "inside_window"
    assert not step["experiment_evidence"] and step["structure_nominations"]
    assert lab.observations == lab.measurements == 120
    assert len(rows) == 88 and budget.spent <= 101000


def test_actual_unresolved_structure_can_lose_to_a_smooth_deletion():
    previous = MeasuredTemporalResponse()
    background = Probe("purchased_background", tuple({"drive": 1., "bias": 0.}
        for _ in range(24)))
    local = propose_observed_condition_duration(spec(), previous.acquire(("meter",),
        "old_background", probe=background), SearchBudget(40000))
    actions = [{"action": "distinguish_paid_condition_duration", "competing_structures":
        [item["observed_condition_evidence"] for item in local]}]
    lab = MeasuredTemporalResponse(conditional=False)
    training = [*lab.varied("new_training0", 61), *lab.varied("new_training1", 62)]
    validation = lab.varied("new_validation", 63)
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7, max_candidates=8))
    claims = _observed_response_claims(agent, spec(), training, None,
        SearchBudget(80000), actions, validation)
    best = min(claims, key=lambda item: (0. if max(item["training_loss"], item["validation_loss"]) < 1e-7
                                       else max(item["training_loss"], item["validation_loss"]),
                                       item["candidate"].program.size))
    assert max(best["training_loss"], best["validation_loss"]) < 1e-7
    assert best["candidate"].candidate_id == "experimental_plain:7"
    assert "if(" not in best["candidate"].program.describe()
