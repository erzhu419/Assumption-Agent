"""Ordinary prospective learning executes and judges recursive knowledge changes."""

from dataclasses import replace
import json

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.workspace import ResearchWorkspace


def op(name, *children):
    return Program(name, children=children)


def feature(name):
    return Program("feature", name=name)


def config(**changes):
    return replace(ResearchConfig(compute_budget=22000, observation_budget=8, max_rounds=1,
        strategy="fixed", use_abstractions=False, active_probes=False, state_search=False,
        joint_planning=False, use_warm_start=False, transition_probes=False,
        learn_research_policy=False), **changes)


class Response:
    def __init__(self, name, names, function, probes, sensor=None):
        self.function, self.sensor = function, sensor
        self.spec = TaskSpec(name, "Predict the measured response", tuple(names), (), tuple(probes), max_probe_length=8)

    def observe(self, probe):
        rows, history = [], []
        for control in probe.sequence:
            features = dict(control)
            if self.sensor:
                features.update(self.sensor(features))
            target = self.function(features)
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
        return rows


def learned(library, name, program, rows):
    spec = TaskSpec(name, "Learn from actual component observations", tuple(rows[0].features), (), ())
    identifier, _ = library.learn(Candidate(name, program, "composition"), spec, rows)
    return next(atom for atom in library.callable_atoms(spec, limit=32)
                if atom.definition.source_concept_id == identifier and atom.name.endswith(":abstract:0:definition"))


def resume(task, program, rows, *, reference=(), graph=None):
    workspace = ResearchWorkspace(task.spec.task_id, observations=list(rows), reference=list(reference),
        incumbent=Candidate("existing", program, "composition"), initial_loss=100., fallback=0.,
        graph=graph.to_dict() if graph else {})
    return json.loads(json.dumps(workspace.to_dict()))


def test_controller_repairs_the_failed_component_and_retains_the_supported_old_domain():
    library = KnowledgeLibrary()
    x, z = feature("x"), feature("z")
    square = learned(library, "quadratic", op("mul", x, x),
        [Observation({"x": value}, value**2) for value in (-2., -1., 0., 1., 2.)])
    linear = learned(library, "linear", op("mul", Program("const", value=2.), z),
        [Observation({"z": value}, 2*value) for value in (-2., -1., 0., 1., 2.)])
    program = op("add", square, linear)
    points = tuple({"x": a, "z": b, "regime": regime}
                   for regime in (-1., 1.) for a in (-2., -1., 1., 2.) for b in (-2., -1., 1., 2.))
    fresh = Probe("fresh", tuple({"x": a, "z": b, "regime": regime}
        for regime in (-.5, .5) for a, b in ((-1.5, -.7), (-.2, .3), (.6, -1.1), (1.7, 1.4))))
    task = Response("changed_components", ("x", "z", "regime"),
        lambda row: row["x"]**2 + (2 if row["regime"] < 0 else 3)*row["z"], (fresh,))
    rows = task.observe(Probe("acquired", points))
    support = [row for row in rows if row.features["regime"] < 0]
    library.learn(Candidate("composite", program, "composition"), task.spec, support)
    result = CognitiveAgent(config(), library).run(task, research_state=resume(task, program, rows, reference=support))
    assert any(action.action == "repair_mechanism" and action.compute_cost > 0 for action in result.actions)
    assert any(action.action == "revise" and action.details["operation"] == "mechanism_repair" for action in result.actions)
    assert result.program.children[0] == square
    assert loss(result.program, support + task.observe(fresh), 0., .5) < 1e-10
    assert result.compute_cost <= config().compute_budget
    assert result.observation_cost == fresh.cost


def test_controller_adopts_real_call_sharing_then_reopens_it_on_a_distinguishing_trajectory():
    library = KnowledgeLibrary()
    x = feature("x")
    square = op("mul", x, x)
    difference = op("mul", x, op("mul", op("sub", square, Program("const", value=1.)),
                                               op("sub", square, Program("const", value=4.))))
    source = (-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2.)
    delta = lambda value: value*(value**2-1)*(value**2-4)
    first = learned(library, "nonlinear", op("add", square, difference),
        [Observation({"x": value}, value**2+delta(value)) for value in source])
    second = learned(library, "quadratic", square,
        [Observation({"x": value}, value**2) for value in source])
    original = op("add", first, second)
    equivalent = Probe("agreement", tuple({"x": value} for value in (-2., -1., 0., 1., 2., -2., 0., 2.)))
    task = Response("empirical_merge", ("x",), lambda row: 2*row["x"]**2+delta(row["x"]), (equivalent,))
    first_result = CognitiveAgent(config(), library).run(task,
        research_state=resume(task, original, task.observe(equivalent)))
    comparisons = [action.details for action in first_result.actions if action.action == "compare"]
    assert any(row["execution_gain"] and row["execution_gain"]["before"] > row["execution_gain"]["after"] for row in comparisons)
    assert first_result.program.op == "if"
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert restored.evolution.merges[0]["status"] == "adopted"
    distinguishing = Probe("new_domain", tuple({"x": value} for value in (.2, .3, .4, .5, .6, .7, .8, .9)))
    next_task = Response("empirical_merge", ("x",), task.function, (distinguishing,))
    saved = ResearchWorkspace.from_dict(first_result.research_state)
    saved.observations.extend(next_task.observe(distinguishing))
    second_result = CognitiveAgent(config(), restored).run(next_task, research_state=saved.to_dict())
    assert any(action.action == "split_concepts" and action.details["tested"] for action in second_result.actions)
    assert loss(second_result.program, next_task.observe(distinguishing), 0., .5) < 1e-10
    assert restored.evolution.merges[0]["status"] == "split_adopted"
    assert any(action.action == "split_adopted" for action in second_result.actions)
    assert not restored.without([first.definition.source_concept_id]).evolution.merges


def test_controller_delivers_actual_sensor_measurements_and_uses_dependent_answers():
    probes = tuple(Probe(f"fresh_{index}", tuple({"x": value} for value in (-1.4, -.9, -.6, -.1, .3, .7, 1.1, 1.6)))
                   for index in range(2))
    task = Response("recursive_measurement", ("x", "sensor"), lambda row: row["sensor"]**2, probes,
                    sensor=lambda row: {"sensor": row["x"]**3})
    graph = ResearchGraph()
    graph.add_question("measurement", "Read the acquired sensor", required_measurements=("sensor",),
        expression=feature("sensor"), observations=[Observation({"x": value}, value**6) for value in (-1., 0., 1.)])
    graph.add_question("response", "Test a squared sensor explanation", depends_on=("measurement",),
        expression=op("mul", Program("arg", name="measurement"), Program("arg", name="measurement")))
    initial = op("mul", feature("x"), feature("x"))
    rows = task.observe(Probe("acquired", tuple({"x": value} for value in (-1.5, -.8, -.3, .2, .6, 1.2))))
    cfg = config(observation_budget=16, max_rounds=2, use_memory=False)
    result = CognitiveAgent(cfg).run(task, research_state=resume(task, initial, rows, graph=graph))
    assert any(action.action == "supply_research_measurements" for action in result.actions)
    assert any(action.action == "execute_research_graph" and "answer:response" in action.details["conclusions"] for action in result.actions)
    assert any(action.action == "revise" and action.details["operation"] == "research_answer" for action in result.actions)
    assert loss(result.program, task.observe(probes[0]), 0., .5) < 1e-10
    assert result.compute_cost <= cfg.compute_budget and result.observation_cost <= cfg.observation_budget
    # The same recursive hypothesis is a candidate, so fresh contradictory
    # measurements can keep the superior incumbent instead of forcing adoption.
    changed = Response(task.spec.task_id, task.spec.feature_names, lambda row: row["x"]**2, probes, sensor=task.sensor)
    graph.supply_measurements(rows)
    kept = CognitiveAgent(config(use_memory=False)).run(changed,
        research_state=resume(changed, initial, rows, reference=rows, graph=graph))
    assert kept.program == initial


def test_controller_creates_a_residual_subproblem_from_a_learned_computation():
    library = KnowledgeLibrary()
    x, power = feature("x"), feature("x")
    for _ in range(4):
        power = op("mul", power, x)
    known = learned(library, "power", power,
        [Observation({"x": value}, value**5) for value in (-2., -1., 0., 1., 2.)])
    fresh = Probe("fresh", tuple({"x": a, "z": b} for a, b in
        ((-1.8, -1.4), (-1.2, .4), (-.2, 1.1), (.3, -.8), (.7, 1.2), (1.3, -.6), (1.6, 1.5), (1.9, -1.1))))
    task = Response("remaining_mechanism", ("x", "z"), lambda row: row["x"]**5+row["z"]**2, (fresh,))
    rows = task.observe(Probe("acquired", tuple({"x": a, "z": b}
        for a in (-2., -1., 1., 2.) for b in (-2., -1., 1., 2.))))
    result = CognitiveAgent(config(strategy="adaptive", compute_budget=24000), library).run(task,
        research_state=resume(task, known, rows))
    assert any(action.action == "create_subproblem" for action in result.actions)
    assert any(action.action == "revise" and action.details["operation"] == "research_answer" for action in result.actions)
    assert result.program.children[0] == known
    assert loss(result.program, task.observe(fresh), 0., .5) < 1e-10
