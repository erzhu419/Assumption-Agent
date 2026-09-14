import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Definition, EvaluationContext, Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.structural_discovery import (
    StructuralDiscovery, boundary_probe, confirm_structure, discover_structure,
    distinguishing_probes, project_batch,
)
from hegel_machine.cognition.structure_lab import StructureBatch, StructureLab


def operators(program):
    return {program.op} | set().union(*(operators(child) for child in program.children))


def test_object_event_window_operators_execute_through_calls_and_renaming():
    body = Program("window_sum", lag=3, children=(Program("arg", name="quantity"),))
    definition = Definition("three_event_total", ("quantity",), body)
    quantity = Program("object_sum", children=(Program("feature", name="a"), Program("feature", name="b")))
    call = Program("call", name=definition.definition_id, children=(quantity,), definition=definition)
    row = Observation({"a": 7., "b": 8.}, 0., ({"a": 1., "b": 2., "__target__": 90.},
        {"a": 3., "b": 4., "__target__": 91.}, {"a": 5., "b": 6., "__target__": 92.}))
    restored = Program.from_dict(json.loads(json.dumps(call.to_dict())))
    context = EvaluationContext(row)
    assert restored.predict(row, context=context) == 33.
    assert call.expanded().predict(row) == 33.
    assert context.primitive_steps > 0
    renamed = restored.remap_features({"a": "left", "b": "right"})
    changed = Observation({"left": 7., "right": 8.}, 0., tuple(
        {"left": past["a"], "right": past["b"], "__target__": past["__target__"]} for past in row.history))
    assert renamed.predict(changed) == 33.
    event = Program("event_delta", lag=1, children=(call,))
    assert event.predict(row) == 12.
    assert event.predict(Observation(row.features, 0.)) == 0.
    assert call.predict(Observation(row.features, 0.)) == 15.
    assert event.minimum_history_depth == 0


@pytest.mark.parametrize("readout,required,dimension", [
    ("total", "object_sum", 2), ("focus", "neighbor_sum", 3),
    ("window", "window_sum", 1), ("event", "event_delta", 1),
])
def test_observed_structures_reduce_dimensions_and_predict_new_trajectories(readout, required, dimension):
    lab = StructureLab(readout=readout)
    batch = lab.acquire(lab.random_probe(89, length=24))
    budget = SearchBudget(28000)
    found = discover_structure(batch.observations, batch.schema, budget=budget, transitions=batch.transitions)
    restored = StructuralDiscovery.from_dict(json.loads(json.dumps(found.to_dict())))
    prediction = restored.candidates[0].program
    fresh = lab.evaluation_rows(seed=165, length=48)
    assert loss(prediction, fresh, 0., .5) < 1e-8
    assert required in operators(prediction)
    assert restored.fits[0]["effective_dimension"] == dimension
    assert dimension < restored.fits[0]["raw_dimension"]
    assert 0 < restored.compute_cost == budget.spent <= budget.limit
    assert prediction.to_dict() == found.candidates[0].program.to_dict()


@pytest.mark.parametrize("variant,refuted", [
    ("exchange", set()), ("anchored", {"conservation", "translation", "permutation"}),
    ("nonlocal", {"conservation", "locality"}),
])
def test_paid_interventions_support_or_refute_structural_proposals(variant, refuted):
    lab = StructureLab(variant=variant)
    training_probe = lab.random_probe(89, length=24)
    training = lab.acquire(training_probe)
    measured_pairs = [(Observation(point, row.target), row)
                      for point, row in zip(training_probe.sequence, training.observations)]
    budget = SearchBudget(30000)
    found = discover_structure(training.observations, training.schema, budget=budget, transitions=training.transitions)
    experiments = [{"kind": plan["kind"], "baseline": lab.acquire(plan["baseline"]),
                    "batch": lab.acquire(plan["probe"])}
                   for plan in distinguishing_probes(found, lab.spec, length=6,
                                                      paired_observations=measured_pairs)]
    before = budget.spent
    hypotheses = confirm_structure(found, experiments, budget=budget)
    by_kind = {item["kind"]: item for item in hypotheses}
    assert budget.spent > before
    assert {kind for kind, item in by_kind.items() if item["status"] == "refuted"} == refuted
    for kind in refuted:
        assert by_kind[kind]["counterexamples"]
    if not refuted:
        assert all(item["status"] == "supported" for item in hypotheses)


def test_aggregation_changes_observations_and_billed_measurements():
    lab = StructureLab(readout="total")
    probe = lab.random_probe(181, length=16)
    fine = lab.acquire(probe)
    coarse_schema = lab.schema_for(scale=4)
    projected = project_batch(fine, coarse_schema)
    coarse = lab.acquire(probe, scale=4)
    assert coarse.observations == projected.observations
    assert coarse.transitions == projected.transitions
    assert coarse.measurement_cost * 4 == fine.measurement_cost
    assert projected.measurement_cost == projected.observation_cost == 0
    assert len(coarse.observations[0].features) * 4 == len(fine.observations[0].features)
    restored = StructureBatch.from_dict(json.loads(json.dumps(coarse.to_dict())))
    assert restored == coarse
    found = discover_structure(restored.observations, restored.schema, budget=SearchBudget(5000))
    fresh = lab.evaluation_rows(scale=4, seed=501, length=24)
    assert loss(found.candidates[0].program, fresh, 0., .5) < 1e-8


def test_ambiguous_local_graph_is_revised_by_a_boundary_intervention():
    lab = StructureLab()
    batch = lab.acquire(lab.random_probe(89, length=24))
    found = discover_structure(batch.observations, batch.schema, budget=SearchBudget(28000))
    initial_width = found.radius_interval[1] - found.radius_interval[0]
    rows = list(batch.observations)
    corrected_errors = []
    for index in range(3):
        left, right = found.radius_interval
        probe = boundary_probe(found, lab.spec, index=index)
        counterexamples = lab.acquire(probe).observations
        previous_error = loss(found.candidates[0].program, counterexamples, 0., .5)
        rows.extend(counterexamples)
        found = discover_structure(rows, batch.schema, budget=SearchBudget(16000),
            radius_candidates=[left + (right - left) * part for part in (.25, .5, .75)])
        corrected_errors.append(previous_error - loss(found.candidates[0].program, counterexamples, 0., .5))
    assert found.radius_interval[1] - found.radius_interval[0] <= initial_width / 8 + 1e-10
    assert loss(found.candidates[0].program, rows, 0., .5) < 1e-8
    assert max(corrected_errors) > .001
