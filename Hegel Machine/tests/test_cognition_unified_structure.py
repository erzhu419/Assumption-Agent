import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.structure_lab import StructureLab
from hegel_machine.cognition.unified_structure import (
    assess_structural_experiments, propose_structures, structural_experiments,
    structural_probe,
)


@pytest.mark.parametrize("readout,family,operator", [
    ("total", "conserved_total", "object_sum"),
    ("focus", "local_exchange", "neighbor_sum"),
    ("window", "window_exchange", "window_sum"),
    ("event", "event_change", "event_delta"),
])
def test_public_schema_proposals_predict_fresh_trajectories(readout, family, operator):
    lab = StructureLab(readout=readout, seed=71)
    training = lab.acquire(lab.random_probe(89, length=24))
    budget = SearchBudget(80000)
    entries = propose_structures(lab.spec, training.observations, budget,
                                  transitions=training.transitions)
    selected = entries[0]
    assert selected["kind"] == f"structural:{family}"
    assert operator in selected["candidate"].program.describe()
    assert 0 < budget.spent <= budget.limit
    fresh = lab.acquire(lab.random_probe(165, length=48)).observations
    assert loss(selected["candidate"].program, fresh, 0., .5) < 1e-8
    assert selected["structure"]["candidates"][0]["candidate_id"] == selected["candidate"].candidate_id
    assert selected["structure"]["fits"][0]["family"] == family
    assert len(selected["structure"]["candidates"]) == 1


def test_prospective_winner_owns_its_boundary_experiment():
    lab = StructureLab(seed=71)
    training = lab.acquire(lab.random_probe(89, length=4)).observations
    validation = lab.acquire(lab.random_probe(129, length=48)).observations
    entries = propose_structures(lab.spec, training, SearchBudget(80000))
    selected = min(entries, key=lambda entry: loss(entry["candidate"].program, validation, 0., .5))
    assert selected is not entries[0]
    assert selected["structure"]["radius_interval"] != entries[0]["structure"]["radius_interval"]
    probe = structural_probe(selected, lab.spec, 7, 6)
    assert probe.probe_id == "structure_boundary_7"
    low, high = selected["structure"]["radius_interval"]
    assert probe.sequence[0]["x_o1"] - probe.sequence[0]["x_o0"] == pytest.approx((low + high) / 2)
    assert len(probe.sequence) == 6


def test_serialized_parent_narrows_live_radius_using_new_measurements():
    lab = StructureLab(readout="window", seed=71)
    rows = lab.acquire(lab.random_probe(89, length=24)).observations
    budget = SearchBudget(100000)
    selected = propose_structures(lab.spec, rows, budget)[0]
    original = selected["structure"]["radius_interval"]
    for index in range(3):
        parent = {"structure": json.loads(json.dumps(selected["structure"]))}
        probe = structural_probe(parent, lab.spec, index, 6)
        rows.extend(lab.acquire(probe).observations)
        selected = propose_structures(lab.spec, rows, budget, parent=parent)[0]
        assert selected["structure_search"] == "boundary_revision"
        assert selected["training_loss"] < 1e-8
    low, high = selected["structure"]["radius_interval"]
    assert high - low <= (original[1] - original[0]) / 8 + 1e-12
    fresh = lab.acquire(lab.random_probe(907, length=48)).observations
    assert loss(selected["candidate"].program, fresh, 0., .5) < 1e-8
    assert budget.spent <= budget.limit


def test_old_family_reopens_when_new_observations_refute_it():
    previous = StructureLab(readout="focus", seed=71)
    old_rows = previous.acquire(previous.random_probe(89, length=24)).observations
    parent = propose_structures(previous.spec, old_rows, SearchBudget(80000))[0]
    changed = StructureLab(readout="window", seed=71)
    new_rows = changed.acquire(changed.random_probe(89, length=24)).observations
    entries = propose_structures(changed.spec, new_rows, SearchBudget(80000), parent=parent)
    assert entries[0]["structure_search"] == "full"
    assert entries[0]["kind"] == "structural:window_exchange"
    assert entries[0]["training_loss"] < 1e-8


def test_no_structural_evidence_adds_no_search_or_boundary():
    budget = SearchBudget(80000)
    spec = TaskSpec("scalar", "A scalar response.", ("u",), (), ())
    assert propose_structures(spec, [], budget) == []
    assert structural_probe({}, spec, 0, 6) is None
    assert structural_experiments({}, spec, 0) == []
    assert assess_structural_experiments({}, [], budget) == []
    assert budget.spent == 0


@pytest.mark.parametrize("variant,refuted", [
    ("exchange", set()),
    ("anchored", {"conservation", "translation", "permutation"}),
    ("nonlocal", {"conservation", "locality"}),
])
def test_paired_outcomes_and_transitions_refute_proposed_laws(variant, refuted):
    source = StructureLab(seed=71)
    training_probe = source.random_probe(89, length=24)
    training = source.acquire(training_probe)
    measured_pairs = [(Observation(point, row.target), row)
                      for point, row in zip(training_probe.sequence, training.observations)]
    entry = propose_structures(source.spec, training.observations, SearchBudget(80000),
                               transitions=training.transitions)[0]
    lab = StructureLab(variant=variant, seed=71)
    plans = structural_experiments(entry, lab.spec, 2, paired_observations=measured_pairs)
    other_plans = structural_experiments(entry, lab.spec, 3, paired_observations=measured_pairs)
    assert {plan[key].probe_id for plan in plans for key in ("baseline", "probe")}.isdisjoint(
        {plan[key].probe_id for plan in other_plans for key in ("baseline", "probe")})
    pairs = [{"kind": plan["kind"], "baseline": lab.acquire(plan["baseline"]),
              "batch": lab.acquire(plan["probe"])} for plan in plans]
    budget = SearchBudget(10000)
    hypotheses = assess_structural_experiments(entry, pairs, budget)
    assert {item["kind"] for item in hypotheses if item["status"] == "refuted"} == refuted
    assert all(item["counterexamples"] for item in hypotheses if item["kind"] in refuted)
    assert all(item["status"] == "supported" for item in hypotheses if item["kind"] not in refuted)
    assert 0 < budget.spent <= budget.limit
    assert all(item["status"] == "proposed" for item in entry["structure"]["hypotheses"])
