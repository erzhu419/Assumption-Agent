"""Learned individual observations can supply a newly changed aggregate."""

import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.research_programs import propose_research_programs, replay_research_program


@pytest.fixture(scope="module")
def failures():
    return json.loads((Path(__file__).parent / "fixtures/cognition_v066_object_call_lift.json").read_text())


def _inputs(case):
    spec = TaskSpec("anonymous_process", "", tuple(case["feature_names"]), (), (),
                    context={"object_schema": json.dumps(case["schema"])})
    batches = []
    for recorded in case["batches"]:
        rows, history = [], []
        for item in recorded:
            rows.append(Observation(item["features"], item["target"], tuple(history)))
            history.append({**item["features"], "__target__": item["target"]})
        batches.append(rows)
    return spec, [Program.from_dict(call) for call in case["calls"]], batches


@pytest.mark.parametrize("case_index", [0, 1])
def test_paid_changed_aggregate_and_window_reuse_the_learned_individual_observation(failures, case_index):
    case = failures["cases"][case_index]
    spec, calls, batches = _inputs(case)
    frozen = [call.to_dict() for call in calls]
    budget = SearchBudget(32000)
    candidates = propose_research_programs(spec, batches[0] + batches[1], budget, calls)
    winner = min(candidates, key=lambda item: loss(item["candidate"].program, batches[2], 0., .5))
    assert case["original_terminal_mse"] > .02
    assert winner["training_loss"] < 1e-12
    assert loss(winner["candidate"].program, batches[2], 0., .5) < 1e-12
    assert loss(winner["candidate"].program, batches[3], 0., .5) < 1e-12
    steps = winner["research_program"]["selection_steps"]
    assert steps[0]["operation"] == "lift_object_observation"
    assert steps[0]["instances"] == len(case["schema"]["objects"])
    assert steps[-1]["coordinate_operation"] == "window_3_current_and_event"
    assert case["source_concept"] in winner["candidate"].parents
    assert budget.spent < 10000
    assert [call.to_dict() for call in calls] == frozen
    restored = replay_research_program(json.loads(json.dumps(winner["research_program"])),
        batches[0] + batches[1], SearchBudget(3000))
    assert restored is not None
    assert loss(restored["candidate"].program, batches[3], 0., .5) < 1e-12


def test_lifted_dependencies_replay_under_new_measurement_names(failures):
    spec, calls, batches = _inputs(failures["cases"][1])
    candidates = propose_research_programs(spec, batches[0] + batches[1], SearchBudget(32000), calls)
    best = min(candidates, key=lambda item: loss(item["candidate"].program, batches[2], 0., .5))
    mapping = {name: f"relabelled_{index}" for index, name in enumerate(reversed(spec.feature_names))}

    def renamed(rows):
        return [Observation({mapping[name]: value for name, value in row.features.items()}, row.target,
            tuple({mapping.get(name, name): value for name, value in point.items()} for point in row.history))
            for row in rows]

    replayed = replay_research_program(json.loads(json.dumps(best["research_program"])),
        renamed(batches[0] + batches[1]), SearchBudget(3000), feature_mapping=mapping)
    assert replayed is not None
    assert replayed["candidate"].program.required_features() <= set(mapping.values())
    assert loss(replayed["candidate"].program, renamed(batches[3]), 0., .5) < 1e-12


def test_lifting_graph_answers_match_their_recorded_training_losses(failures):
    spec, calls, batches = _inputs(failures["cases"][1])
    training = batches[0] + batches[1]
    candidates = propose_research_programs(spec, training, SearchBudget(32000), calls)
    selected = min(candidates, key=lambda item: item["training_loss"])
    graph = ResearchGraph.from_dict(selected["research_program"]["graph"])
    fit_nodes = [node for key, node in graph.nodes.items() if key.startswith("fit:")]
    assert len(fit_nodes) == 2
    for node in fit_nodes:
        assert loss(node.answer, training, 0., .5) == pytest.approx(node.loss, rel=1e-10, abs=1e-12)
