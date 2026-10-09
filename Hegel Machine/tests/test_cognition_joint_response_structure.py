"""Discover response combinations and their edge range from public measurements."""

import json
from itertools import combinations

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Probe, SearchBudget
from hegel_machine.cognition.structural_discovery import StructuralDiscovery, discover_structure
from hegel_machine.cognition.structure_lab import StructureLab
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_structure import propose_structures
from hegel_machine.cognition.unified_structure_transfer import transfer_structure


def _operators(program):
    return {program.op}.union(*(_operators(child) for child in program.children))


def _reset_trajectories(lab, seeds, length=24):
    return [row for seed in seeds
            for row in lab.acquire(lab.random_probe(seed, length=length)).observations]


@pytest.mark.parametrize("readout,temporal", [
    ("total_event", {"event_delta"}),
    ("focus_window", {"window_sum"}),
    ("window_total_event", {"window_sum", "event_delta"}),
])
def test_empty_discovery_jointly_identifies_response_and_relation_range(readout, temporal):
    lab = StructureLab(readout=readout, seed=71)
    paid = _reset_trajectories(lab, (89, 90, 91))
    budget = SearchBudget(200000)
    # Only public object roles and acquired measurements reach the proposer.
    # There is no parent, response label, transition law, or supplied radius.
    found = discover_structure(paid, lab.object_schema, budget=budget)
    restored = StructuralDiscovery.from_dict(json.loads(json.dumps(found.to_dict())))
    program = restored.candidates[0].program
    fresh = _reset_trajectories(lab, (1801, 1802), length=48)
    assert restored.fits[0]["training_loss"] < 1e-8
    assert loss(program, fresh, 0., .5) < 1e-8
    assert {"neighbor_sum", *temporal}.issubset(_operators(program))
    assert restored.radius_interval[0] < lab._radius <= restored.radius_interval[1]
    assert 0 < found.compute_cost == budget.spent <= budget.limit
    assert restored.candidates[0].program == found.candidates[0].program


@pytest.mark.parametrize("readout", ["total_event", "focus_window", "window_total_event"])
def test_joint_response_family_revises_old_radius_without_losing_temporal_law(readout):
    source = StructureLab(readout=readout, seed=71)
    source_rows = _reset_trajectories(source, (89, 90, 91))
    parent = propose_structures(source.spec, source_rows, SearchBudget(200000))[0]
    # Parent metadata is stored and later read by a different research episode.
    parent = {"structure": json.loads(json.dumps(parent["structure"]))}
    old_interval = tuple(parent["structure"]["radius_interval"])
    changed = StructureLab(readout=readout, seed=99)
    paid = _reset_trajectories(changed, (317, 318, 319))
    budget = SearchBudget(180000)
    revised = min(propose_structures(changed.spec, paid, budget, parent=parent),
                  key=lambda entry: entry["training_loss"])
    interval = revised["structure"]["radius_interval"]
    fresh = _reset_trajectories(changed, (1901, 1902), length=48)
    assert revised["structure_search"] == "family_revision"
    assert revised["structure"]["fits"][0]["family"] == parent["structure"]["fits"][0]["family"]
    assert interval[1] < old_interval[0]
    assert interval[0] < changed._radius <= interval[1]
    assert loss(revised["candidate"].program, fresh, 0., .5) < 1e-8
    assert tuple(parent["structure"]["radius_interval"]) == old_interval
    assert 0 < budget.spent <= budget.limit


def _history_only_endings(lab, distances, quantities):
    rows = []
    for distance in distances:
        for quantity in quantities:
            previous = {"q_o0": quantity, "q_o1": -.3, "x_o0": 0., "x_o1": distance,
                        "u_o0": 0., "u_o1": 0., "focus": 0.}
            current = {"q_o0": 0., "q_o1": 0., "x_o0": 0., "x_o1": 1.9,
                       "u_o0": 0., "u_o1": 0., "focus": 0.}
            batch = lab.acquire(Probe(f"history_{distance}_{quantity}", (previous, current)))
            rows.append(batch.observations[-1])
    return rows


def test_history_only_edges_bound_radius_instead_of_current_uninformative_frames():
    lab = StructureLab(objects=2, readout="event", seed=71)
    paid = _history_only_endings(lab, (.4, .8, 1., 1.4), (-.7, .2, .8))
    # Every selected current frame has zero flow at every radius. Only its
    # observed predecessor reveals whether the response contains an edge.
    assert all(row.features["q_o0"] == row.features["q_o1"] for row in paid)
    found = discover_structure(paid, lab.object_schema, budget=SearchBudget(40000))
    assert found.fits[0]["training_loss"] < 1e-8
    assert found.radius_interval == pytest.approx((.8, 1.))
    fresh = _history_only_endings(lab, (.65, .85, 1.05, 1.65), (-.6, .5))
    assert loss(found.candidates[0].program, fresh, 0., .5) < 1e-8


def test_learned_joint_family_rebinds_response_terms_to_changed_instruments():
    source = StructureLab(readout="window_total_event", seed=71)
    paid = _reset_trajectories(source, (89, 90, 91))
    learned = propose_structures(source.spec, paid, SearchBudget(200000))[0]
    prior = {"source_concept_id": "concept_0014",
             "structure": json.loads(json.dumps(learned["structure"]))}
    original = json.dumps(prior, sort_keys=True)
    target = UnifiedObjectLab(readout="window_total_event", seed=99, rename_prefix="new_")
    spec = target.spec_for(("fine",))
    training = [row for seed in (317, 318, 319)
                for row in target.acquire(target.random_probe(seed, length=24), ("fine",)).observations]
    budget = SearchBudget(60000)
    entries = transfer_structure(spec, training, budget, prior)
    assert entries
    selected = entries[0]
    fresh = [row for seed in (1901, 1902)
             for row in target.acquire(target.random_probe(seed, length=48), ("fine",)).observations]
    assert loss(selected["candidate"].program, fresh, 0., .5) < 1e-8
    assert selected["method_source_ids"] == ["concept_0014"]
    assert selected["candidate"].parents == ("concept_0014",)
    terms = [Program.from_dict(term)
             for term in selected["structure"]["fits"][0]["response_terms"]]
    assert terms
    assert all(name.startswith("measured_new_") for term in terms for name in term.required_features())
    assert selected["candidate"].program.required_features().issubset(spec.feature_names)
    assert selected["structure"]["radius_interval"][1] < prior["structure"]["radius_interval"][0]
    assert json.dumps(prior, sort_keys=True) == original
    assert 0 < budget.spent <= budget.limit


def test_radius_search_leaves_a_locally_best_inconsistent_relation():
    lab = StructureLab(readout="window_total_event", seed=71)
    paid = lab.acquire(lab.random_probe(105, length=16)).observations
    schema = lab.object_schema
    reductions = [Program("object_sum", children=tuple(
        Program("feature", name=schema.fields[item][role]) for item in schema.objects))
        for role in ("q", "u", "x")]
    response_terms = [Program("window_sum", lag=3, children=(term,))
                      for term in [*reductions, *(Program("event_delta", lag=1, children=(term,))
                                                   for term in reductions)]]
    distances = sorted({abs(row.features[schema.fields[left]["x"]] -
                            row.features[schema.fields[right]["x"]])
                        for left, right in combinations(schema.objects, 2) for row in paid
                        if (row.features[schema.fields[left]["f"]] - row.features[schema.fields[right]["f"]]) *
                           (row.features[schema.fields[left]["q"]] - row.features[schema.fields[right]["q"]])})
    boundaries = [(left + right) / 2 for left, right in zip(distances, distances[1:])]
    trap = min(range(len(boundaries)), key=lambda index: abs(boundaries[index] - 2.62))
    local_radii = boundaries[trap - 1:trap + 2]
    profile_budget = SearchBudget(1500)
    local = discover_structure(paid, schema, budget=profile_budget, response_terms=response_terms,
                               families=("joint_window_event",), radius_candidates=local_radii)
    losses = {radius: min(fit["training_loss"] for fit in local.fits if fit["radius"] == radius)
              for radius in local_radii}
    # This fixed paid trajectory has a genuine wrong local minimum: neither
    # adjacent graph cell improves it, although another range explains it.
    assert 1e-4 < losses[local_radii[1]] < min(losses[local_radii[0]], losses[local_radii[2]])
    budget = SearchBudget(5000)
    found = discover_structure(paid, schema, budget=budget, response_terms=response_terms,
                               families=("joint_window_event",))
    assert found.fits[0]["training_loss"] < 1e-8
    assert found.radius_interval[0] < lab._radius <= found.radius_interval[1]
    assert 0 < found.compute_cost == budget.spent <= budget.limit
