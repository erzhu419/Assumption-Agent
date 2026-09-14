"""Reuse measured mechanisms instead of relearning their interaction boundary."""

import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import propose_research_programs, replay_research_program


@pytest.fixture(scope="module")
def paid_failure():
    return json.loads((Path(__file__).parent /
                       "fixtures/cognition_v065_temporal_role_elimination.json").read_text())


def _inputs(fixture, position):
    spec = TaskSpec("anonymous_process", "", tuple(fixture["feature_names"]), (), (),
                    context={"object_schema": json.dumps(fixture["schema"])})
    batches = []
    for recorded in fixture["cases"][position]["batches"]:
        history, rows = [], []
        for item in recorded:
            rows.append(Observation(item["features"], item["target"], tuple(history)))
            history.append({**item["features"], "__target__": item["target"]})
        batches.append(rows)
    return spec, [Program.from_dict(call) for call in fixture["calls"]], batches


@pytest.mark.parametrize("position,operator", [(0, "event"), (1, "current_and_event"),
                                               (2, "window_3_current_and_event")])
def test_paid_event_and_composition_failures_reuse_the_existing_mechanism(paid_failure, position, operator):
    spec, calls, batches = _inputs(paid_failure, position)
    frozen = [call.to_dict() for call in calls]
    budget = SearchBudget(32000)
    candidates = propose_research_programs(spec, batches[0] + batches[1], budget, calls)
    winner = min(candidates, key=lambda entry: loss(entry["candidate"].program, batches[2], 0., .5))
    assert winner["training_loss"] < 1e-12
    assert loss(winner["candidate"].program, batches[2], 0., .5) < 1e-12
    # Includes the independent 48-row confirmation that rejected the old event
    # radius, and the recorded failing total/event and window successors.
    assert loss(winner["candidate"].program, batches[3], 0., .5) < 1e-12
    step = winner["research_program"]["selection_steps"][-1]
    assert step["operation"] in {"derive_temporal_mechanism", "compose_lifted_observation"}
    assert step["coordinate_operation"] == operator
    assert "concept_0001" in winner["candidate"].parents
    assert [call.to_dict() for call in calls] == frozen
    assert budget.spent < 10000
    restored = replay_research_program(json.loads(json.dumps(winner["research_program"])),
        batches[0] + batches[1], SearchBudget(3000))
    assert restored is not None
    assert loss(restored["candidate"].program, batches[3], 0., .5) < 1e-12


def test_selected_temporal_dependencies_replay_after_rename_and_contribution_change(paid_failure):
    spec, calls, batches = _inputs(paid_failure, 1)
    candidates = propose_research_programs(spec, batches[0] + batches[1], SearchBudget(32000), calls)
    winner = min(candidates, key=lambda entry: loss(entry["candidate"].program, batches[2], 0., .5))
    total = next(call for call in calls if call.definition.source_concept_id == "concept_0002")
    mapping = {name: f"new_sensor_{index}" for index, name in enumerate(reversed(spec.feature_names))}

    def changed(rows):
        return [Observation({mapping[key]: value for key, value in row.features.items()},
            -.6 * row.target + 1.8 * total.predict(row),
            tuple({mapping.get(key, key): value for key, value in point.items()} for point in row.history))
            for row in rows]

    training = changed(batches[0] + batches[1])
    confirmation = changed(batches[3])
    replayed = replay_research_program(json.loads(json.dumps(winner["research_program"])),
        training, SearchBudget(3000), feature_mapping=mapping)
    assert replayed is not None
    assert loss(replayed["candidate"].program, confirmation, 0., .5) < 1e-12
    assert replayed["candidate"].program.required_features() <= set(mapping.values())
