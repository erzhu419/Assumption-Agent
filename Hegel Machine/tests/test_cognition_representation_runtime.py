"""Continued research funds a complete historical representation and comparison."""

from random import Random

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.workspace import ResearchWorkspace


class DelayedMeasurements:
    def __init__(self):
        self.spec = TaskSpec("measured_delay", "Predict the measured response", ("drive", "disturbance"),
                            (), (), feature_bounds={"drive": (-2., 2.), "disturbance": (-2., 2.)},
                            max_probe_length=32)

    def observe(self, probe):
        history, rows = [], []
        for point in probe.sequence:
            x, z = point["drive"], point["disturbance"]
            target = .2 + .4*x - .3*x*x + .8*x*z
            if len(history) >= 3:
                target += .65 * history[-3]["drive"]
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        return rows


def test_resumed_86_row_problem_reaches_input_lag_after_output_lag_with_same_budget():
    environment = DelayedMeasurements()
    rng = Random(925)
    probes = [Probe(f"acquired_{index}", tuple({"drive": rng.uniform(-2., 2.), "disturbance": rng.uniform(-2., 2.)}
                    for _ in range(length))) for index, length in enumerate((14, 18, 24, 30))]
    observations = [row for probe in probes for row in environment.observe(probe)]
    assert len(observations) == 86
    x, z = Program("feature", name="drive"), Program("feature", name="disturbance")
    def mul(coefficient, term):
        return Program("mul", children=(Program("const", value=coefficient), term))
    current = Program("add", children=(Program("const", value=.2),
        Program("add", children=(mul(.4, x), Program("add", children=(
            mul(-.3, Program("mul", children=(x, x))), mul(.8, Program("mul", children=(x, z)))))))))
    wrong = Program("if", children=(Program("available", lag=3),
        Program("add", children=(current, mul(.2, Program("lag", name="__target__", lag=3)))), current))
    fallback = initial_prediction(observations, .5)
    incumbent = Candidate("old_output_lag", wrong, "composition")
    parent = Candidate("current_only", current, "composition")
    champion = Candidate("earlier_constant", Program("const", value=fallback), "refit")
    workspace = ResearchWorkspace(environment.spec.task_id, observations=observations, reference=observations[:52],
        queried=probes, incumbent=incumbent, parent=parent, evidence_champion=champion,
        bank=[incumbent, parent, champion], fallback=fallback,
        initial_loss=loss(champion.program, observations, fallback, .5), rounds_completed=4)
    config = ResearchConfig(seed=42, compute_budget=24000, observation_budget=48, use_memory=False)
    result = CognitiveAgent(config).run(environment, research_state=workspace.to_dict())
    fresh = [row for index in range(4) for row in environment.observe(Probe(f"new_{index}", tuple(
        {"drive": rng.uniform(-2., 2.), "disturbance": rng.uniform(-2., 2.)} for _ in range(32))))]
    assert loss(wrong, fresh, fallback, .5) > .1
    assert loss(result.program, fresh, result.fallback, .5) < 1e-10
    assert ("drive", 3) in result.program.observed_variables()
    assert ("__target__", 3) not in result.program.observed_variables()
    assert result.compute_cost <= 24000
    assert result.observation_cost <= 48
    assert any(action.action == "composition" and action.round == 4 for action in result.actions)
    first_revision = next(action for action in result.actions if action.action == "revise")
    assert first_revision.round == 4
    assert ("drive", 3) in Program.from_dict(first_revision.details["after_program"]).observed_variables()
    assert result.research_state["evidence_champion"] is not None
