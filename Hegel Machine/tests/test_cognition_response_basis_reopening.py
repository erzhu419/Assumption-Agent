"""Current response evidence can replace a transferred temporal background."""

from copy import deepcopy
import json

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.structural_discovery import (
    StructuralDiscovery, confirm_structure, discover_structure, distinguishing_probes,
)
from hegel_machine.cognition.structure_lab import StructureLab


def paid_trajectories(laboratory, seeds=(89, 90), length=16):
    return [row for seed in seeds for row in laboratory.acquire(
        laboratory.random_probe(seed, length=length)).observations]


def old_window_drive(schema):
    total = Program("object_sum", children=tuple(
        Program("feature", name=schema.fields[obj]["u"]) for obj in schema.objects))
    return [Program("window_sum", lag=3, children=(total,))]


def test_measured_current_and_event_response_replaces_inherited_window_background():
    laboratory = StructureLab(readout="total_event", seed=71)
    training = paid_trajectories(laboratory)
    inherited = old_window_drive(laboratory.object_schema)
    original = deepcopy([term.to_dict() for term in inherited])
    budget = SearchBudget(45000)
    result = discover_structure(training, laboratory.object_schema, budget=budget,
                                response_terms=inherited)

    # Only measured fields and outcomes reached discovery. The old background
    # cannot express this response, although changing its radius can improve it.
    revision = next(item for item in result.hypotheses if item["kind"] == "response_basis_revision")
    assert revision["previous_loss"] > 1e-3
    assert result.fits[0]["training_loss"] < 1e-8
    fresh = laboratory.evaluation_rows(seed=1901, length=48)
    assert loss(result.candidates[0].program, fresh, 0., .5) < 1e-8
    assert result.fits[0]["response_terms"] != original
    assert [term.to_dict() for term in inherited] == original
    assert 0 < result.compute_cost == budget.spent <= budget.limit


def test_supported_inherited_basis_keeps_its_explanation_without_rediscovery():
    laboratory = StructureLab(readout="window_total_event", seed=71)
    training = paid_trajectories(laboratory)
    learned = discover_structure(training, laboratory.object_schema, budget=SearchBudget(120000))
    assert learned.fits[0]["training_loss"] < 1e-8
    previous = deepcopy(learned.to_dict())
    inherited = [Program.from_dict(term) for term in learned.fits[0]["response_terms"]]
    budget = SearchBudget(12000)
    # The previous paid search supplies this candidate range. This isolates
    # whether a currently adequate response basis is needlessly replaced.
    kept = discover_structure(training, laboratory.object_schema, budget=budget,
        response_terms=inherited, radius_candidates=[learned.fits[0]["radius"]])

    assert kept.fits[0]["training_loss"] < 1e-8
    assert kept.candidates[0].program == learned.candidates[0].program
    assert kept.fits[0]["response_terms"] == learned.fits[0]["response_terms"]
    assert not any(item["kind"] == "response_basis_revision" for item in kept.hypotheses)
    fresh = laboratory.evaluation_rows(seed=1901, length=48)
    assert loss(kept.candidates[0].program, fresh, 0., .5) < 1e-8
    assert learned.to_dict() == previous
    assert 0 < kept.compute_cost == budget.spent <= budget.limit


def test_explicit_family_revision_does_not_expand_the_requested_question():
    laboratory = StructureLab(readout="total_event", seed=71)
    training = paid_trajectories(laboratory)
    inherited = old_window_drive(laboratory.object_schema)
    budget = SearchBudget(45000)
    restricted = discover_structure(training, laboratory.object_schema, budget=budget,
        response_terms=inherited, families=("joint_window_event",))

    assert restricted.fits
    assert all(item["family"] == "joint_window_event" for item in restricted.fits)
    assert all(item["response_terms"] == [term.to_dict() for term in inherited]
               for item in restricted.fits)
    assert min(item["training_loss"] for item in restricted.fits) > 1e-7
    assert not any(item["kind"] == "response_basis_revision" for item in restricted.hypotheses)
    assert 0 < restricted.compute_cost == budget.spent <= budget.limit


def test_response_search_is_not_supported_by_unrelated_structural_confirmation():
    laboratory = StructureLab(readout="total_event", seed=71)
    training = paid_trajectories(laboratory, seeds=(89, 90, 91), length=24)
    budget = SearchBudget(45000)
    found = discover_structure(training, laboratory.object_schema, budget=budget,
                               response_terms=old_window_drive(laboratory.object_schema))
    result = StructuralDiscovery.from_dict(json.loads(json.dumps(found.to_dict())))
    revision = next(item for item in result.hypotheses if item["kind"] == "response_basis_revision")
    paid_error = loss(result.candidates[0].program, training, 0., .5)
    assert paid_error < revision["previous_loss"]
    assert abs(paid_error - revision["best_loss"]) < 1e-15
    original = deepcopy(result.to_dict())

    plans = distinguishing_probes(result, laboratory.spec, seed=1811, length=6)
    pairs = [{"kind": plan["kind"], "baseline": laboratory.acquire(plan["baseline"]),
              "batch": laboratory.acquire(plan["probe"])} for plan in plans]
    confirmation_budget = SearchBudget(1000)
    assessed = confirm_structure(result, pairs, budget=confirmation_budget)
    assert any(item["kind"] == "translation" and item["status"] == "supported"
               for item in assessed)
    # Translation evidence tests an invariant; it does not independently
    # confirm the response model selected from the earlier paid trajectories.
    current = next(item for item in assessed if item["kind"] == "response_basis_revision")
    assert current["status"] == "fit_improved"
    assert result.to_dict() == original
    assert 0 < found.compute_cost == budget.spent <= budget.limit
    assert 0 < confirmation_budget.spent <= confirmation_budget.limit
