"""Programs learned by the agent become executable coordinates in later tasks."""

from dataclasses import replace
import json
from random import Random

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.sparse_search import sparse_candidates
from hegel_machine.cognition.workspace import ResearchWorkspace


class MeasuredComposition:
    def __init__(self, task_id, level):
        self.level = level
        rng = Random(91)
        initial = Probe("initial", tuple({"x": rng.uniform(-1., 1.), "z": rng.uniform(-1., 1.)}
                                         for _ in range(16)))
        self.spec = TaskSpec(task_id, "Predict the measured response", ("x", "z"),
                             (initial,), (), feature_bounds={"x": (-1., 1.), "z": (-1., 1.)})
        self.queries = []

    def observe(self, probe):
        self.queries.append(probe)
        observations, history = [], []
        for point in probe.sequence:
            measured = point["x"] ** 2 + point["x"] * point["z"]
            target = measured if self.level == 1 else measured ** 2 + measured
            observations.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        return observations


def test_actual_learning_defines_a_coordinate_used_by_next_task_and_resumed_program():
    config = ResearchConfig(observation_budget=80, compute_budget=30000, seed=231)
    library = KnowledgeLibrary()
    source = CognitiveAgent(config, library).run(MeasuredComposition("source", 1))
    assert source.concepts_added
    # Only acquired source evidence enters the library; the target response is
    # exposed to the controller exclusively through its ordinary observe API.
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    target = MeasuredComposition("composition", 2)
    result = CognitiveAgent(config, restored).run(target)
    rng = Random(77)
    fresh = target.observe(Probe("evaluation", tuple(
        {"x": rng.uniform(-1., 1.), "z": rng.uniform(-1., 1.)} for _ in range(100))))
    assert loss(result.program, fresh, result.fallback, .5) < 1e-12
    assert result.program.to_dict().get("definitions")
    assert result.program.size == result.program.expanded().size
    assert any(action.action == "composition" and action.details["callable_atoms"]
               for action in result.actions if "callable_atoms" in action.details)
    assert result.compute_cost <= config.compute_budget

    saved = json.loads(json.dumps(result.research_state))
    workspace = ResearchWorkspace.from_dict(saved)
    assert workspace.incumbent.program == result.program
    before = len(target.queries)
    resumed = CognitiveAgent(replace(config, observation_budget=8), restored).run(
        target, research_state=saved)
    assert loss(resumed.program, fresh, resumed.fallback, .5) < 1e-12
    assert resumed.observation_cost <= 8
    assert all(probe.probe_id != "initial" for probe in target.queries[before:])


def test_learned_coordinates_do_not_consume_fit_budget_when_existing_coordinates_explain_the_rows():
    config = ResearchConfig(observation_budget=48, compute_budget=24000, seed=231)
    library = KnowledgeLibrary()
    environment = MeasuredComposition("source", 1)
    source = CognitiveAgent(config, library).run(environment)
    atoms = library.callable_atoms(environment.spec)
    assert atoms
    raw_budget, concept_budget = SearchBudget(2000), SearchBudget(2000)
    raw = sparse_candidates(source.observations, budget=raw_budget, max_history=0)
    with_concepts = sparse_candidates(source.observations, budget=concept_budget, max_history=0,
                                     additional_terms=atoms)
    assert concept_budget.spent == raw_budget.spent
    assert min(loss(item.program, source.observations, 0., .5) for item in with_concepts) < 1e-12
    assert {item.program for item in with_concepts} == {item.program for item in raw}


def test_residual_search_selects_calls_from_evidence_and_funds_their_executable_fit():
    source = MeasuredComposition("source", 1)
    source_rows = source.observe(source.spec.initial_probes[0])
    library = KnowledgeLibrary()
    discovered = sparse_candidates(source_rows, budget=SearchBudget(1800), max_history=0)
    best = min(discovered, key=lambda candidate: loss(candidate.program, source_rows, 0., .5))
    library.learn(best, source.spec, source_rows)
    target = MeasuredComposition("target", 2)
    rng = Random(37)
    rows = target.observe(Probe("fit", tuple({"x": rng.uniform(-1., 1.), "z": rng.uniform(-1., 1.)}
                                             for _ in range(40))))
    selected_baselines = []

    def select(work, baseline):
        selected_baselines.append(baseline)
        return library.select_callable_atoms(target.spec, rows, work, baseline_program=baseline)

    budget = SearchBudget(2600)
    candidates = sparse_candidates(rows, budget=budget, max_history=0, select_additional_terms=select)
    fresh = target.observe(Probe("fresh", tuple({"x": rng.uniform(-1., 1.), "z": rng.uniform(-1., 1.)}
                                                for _ in range(80))))
    assert selected_baselines and selected_baselines[0] is not None
    assert library.last_callable_selection["status"] == "useful_calls_selected"
    assert any(candidate.program.to_dict().get("definitions") and
               loss(candidate.program, fresh, 0., .5) < 1e-12 for candidate in candidates)
    assert budget.spent <= budget.limit

    selected_baselines.clear()
    sparse_candidates(source_rows, budget=SearchBudget(2600), max_history=0, select_additional_terms=select)
    assert not selected_baselines
