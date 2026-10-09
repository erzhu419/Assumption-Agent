"""A paid counterexample can retain a better frozen answer without adoption."""

from copy import deepcopy

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.structural_discovery import discover_structure
from hegel_machine.cognition.structure_lab import StructureLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _fit, _terms


def test_paid_confirmation_rejects_overfit_and_retains_a_frozen_structural_answer():
    laboratory = StructureLab(readout="total_event", seed=71)
    training = [row for seed in (89, 90) for row in laboratory.acquire(
        laboratory.random_probe(seed, length=16)).observations]
    discovery = discover_structure(training, laboratory.object_schema, budget=SearchBudget(120000))
    structural = {"candidate": discovery.candidates[0], "kind": "structural:joint_event",
        "family_id": "T11", "training_loss": discovery.fits[0]["training_loss"],
        "validation_loss": discovery.fits[0]["training_loss"]}
    overfit = _fit(_terms(laboratory.spec.feature_names, 2, True), training,
                   SearchBudget(30000), "quadratic_history")
    simpler = _fit(_terms(laboratory.spec.feature_names), training,
                   SearchBudget(10000), "current_affine")
    overfit["validation_loss"] = overfit["training_loss"]
    simpler["validation_loss"] = simpler["training_loss"]
    assert overfit["training_loss"] < 1e-7 and structural["training_loss"] < 1e-7
    frozen = deepcopy([overfit, simpler, structural])

    # Both answers were frozen before this actual apparatus query. The large
    # quadratic fit explains its training rows but fails the new trajectory.
    confirmation = laboratory.acquire(laboratory.random_probe(1901, length=24)).observations
    incumbent_loss = loss(overfit["candidate"].program, confirmation, 0., .5)
    assert incumbent_loss > .1
    budget, actions = SearchBudget(2 * len(confirmation)), []
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))
    selected, measured_loss = agent._retain_confirmation_alternative(overfit,
        [overfit, simpler, structural, structural], confirmation, laboratory.spec,
        budget, 0., actions, incumbent_loss=incumbent_loss)

    assert selected["candidate"] == structural["candidate"]
    assert measured_loss == loss(selected["candidate"].program, confirmation, 0., .5)
    assert measured_loss < 1e-7
    assert selected["validation_loss"] == measured_loss
    assert selected["selection_basis"] == "all_acquired_evidence"
    assert budget.spent == 2 * len(confirmation)
    assert [overfit, simpler, structural] == frozen
    assert not agent.library.concepts

    # With no calculation left, a frozen alternative cannot be adopted just
    # because it was supported on some earlier sample.
    stopped, stopped_loss = agent._retain_confirmation_alternative(overfit,
        [structural], confirmation, laboratory.spec, SearchBudget(0), 0., [],
        incumbent_loss=incumbent_loss)
    assert stopped is overfit and stopped_loss == incumbent_loss
