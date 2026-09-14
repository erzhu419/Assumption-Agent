"""Local mechanism changes need coupled measured corrections to retained calls."""

import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.research_programs import propose_research_programs, replay_research_program
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab


@pytest.fixture(scope="module")
def failed_case():
    return json.loads((Path(__file__).parent / "fixtures/cognition_v064_joint_role_residual.json").read_text())


def observations(batch):
    history, rows = [], []
    for item in batch:
        rows.append(Observation(item["features"], item["target"], tuple(history)))
        history.append({**item["features"], "__target__": item["target"]})
    return rows


def calls_for(fixture, spec):
    current = json.loads(spec.context["object_schema"])
    mapping = {field: current["fields"][group][role]
               for group, fields in fixture["schema"]["fields"].items() for role, field in fields.items()}
    return [Program.from_dict(call).remap_features(mapping) for call in fixture["calls"]]


def test_previous_paid_failure_is_explained_by_joint_role_residual_without_new_acquisition(failed_case):
    batches = [observations(batch) for batch in failed_case["batches"]]
    with UnifiedObjectLab(apparatus_seed=failed_case["apparatus_seed"]) as lab:
        spec = lab.spec_for(("fine",))
    calls = calls_for(failed_case, spec)
    budget = SearchBudget(32000)
    candidates = propose_research_programs(spec, batches[0] + batches[1], budget, calls)
    best = min(candidates, key=lambda item: loss(item["candidate"].program, batches[2], 0., .5))
    assert best["training_loss"] < 1e-10
    assert loss(best["candidate"].program, batches[2], 0., .5) < 1e-10
    assert loss(best["candidate"].program, batches[3], 0., .5) < 1e-10
    step = best["research_program"]["selection_steps"][-1]
    assert step["operation"] == "joint_role_residual"
    assert step["joint_terms"] > 3
    assert step["parent_loss"] > 1e-4
    assert best["candidate"].parents
    assert budget.spent < 5000


@pytest.mark.parametrize("variant,readout", [
    ("sink", "total_event"), ("sink", "window_total_event"),
    ("nonlocal", "total_event"), ("nonlocal", "window_total_event"),
])
def test_same_public_role_grammar_repairs_local_and_global_changes_and_their_windows(failed_case, variant, readout):
    with UnifiedObjectLab(apparatus_seed=failed_case["apparatus_seed"], variant=variant,
                          readout=readout, actuator_curvature=.47, rename_prefix="changed_") as lab:
        spec = lab.spec_for(("fine",))
        calls = calls_for(failed_case, spec)
        training = [row for seed in (183, 287)
                    for row in lab.observe(lab.random_probe(seed, length=24), ("fine",))]
        validation = lab.observe(lab.random_probe(887, length=24), ("fine",))
        future = [row for seed, length in ((992, 1), (993, 3), (994, 64))
                  for row in lab.observe(lab.random_probe(seed, length=length), ("fine",))]
        budget = SearchBudget(32000)
        proposed = propose_research_programs(spec, training, budget, calls)
        best = min(proposed, key=lambda item: loss(item["candidate"].program, validation, 0., .5))
        assert loss(best["candidate"].program, future, 0., .5) < 1e-7
        assert best["research_program"]["selection_steps"][-1]["operation"] == "joint_role_residual"
        assert budget.spent <= 32000
        # Replay the selected measured dependencies on another paid trajectory;
        # startup and short trajectories remain part of the comparison.
        restored = json.loads(json.dumps(best["research_program"]))
        replay_budget = SearchBudget(3000)
        replayed = replay_research_program(restored, training, replay_budget)
        assert replayed is not None
        assert loss(replayed["candidate"].program, future, 0., .5) < 1e-7
        assert replayed["candidate"].parents


def test_unfunded_joint_role_search_preserves_the_old_calls(failed_case):
    with UnifiedObjectLab(apparatus_seed=failed_case["apparatus_seed"]) as lab:
        spec = lab.spec_for(("fine",))
    calls = calls_for(failed_case, spec)
    saved = [call.to_dict() for call in calls]
    budget = SearchBudget(20)
    assert propose_research_programs(spec, observations(failed_case["batches"][0]), budget, calls) == []
    assert [call.to_dict() for call in calls] == saved
    assert budget.spent == 0
