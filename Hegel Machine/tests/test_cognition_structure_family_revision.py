"""A family with conflicting trajectories is revised before being abandoned."""

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.structure_lab import StructureLab
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_structure import propose_structures


def test_validation_leading_unresolved_family_can_use_all_paid_evidence():
    seed = 3050339151
    lab = UnifiedObjectLab(seed=seed, apparatus_seed=6131, readout="event")
    spec = lab.spec_for(("fine",))
    batches = [lab.acquire(lab.random_probe(seed + 1009 * serial, length=24), ("fine",))
               for serial in (5, 6, 7)]
    training = batches[0].observations + batches[1].observations
    validation = batches[2].observations
    transitions = batches[0].transitions + batches[1].transitions
    entries = propose_structures(spec, training, SearchBudget(28000), transitions=transitions, tolerance=1e-7)
    assessed = [{**item, "validation_loss": loss(item["candidate"].program, validation, 0., .5)}
                for item in entries]
    exact_families = {item["kind"] for item in assessed if item["training_loss"] < 1e-7}
    unresolved = [item for item in assessed
                  if item["training_loss"] > 1e-7 and item["kind"] in exact_families]
    # The improved search may already rank an exact fit first. Exercise a
    # real unresolved candidate rather than requiring the search to miss it.
    parent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))._best(unresolved)
    assert parent["training_loss"] > 1e-7
    assert parent["structure"]["radius_interval"] is None
    assert any(item["training_loss"] < 1e-7 for item in entries)
    budget = SearchBudget(20000)
    revised = propose_structures(spec, training + validation, budget, parent=parent,
                                 transitions=transitions + batches[2].transitions, tolerance=1e-7)[0]
    assert revised["structure_search"] == "family_revision"
    assert revised["kind"] == parent["kind"]
    assert revised["training_loss"] < 1e-7
    assert revised["structure"]["radius_interval"]
    assert budget.spent < 6000


def test_changed_response_reopens_other_structures_after_family_revision_fails():
    source = StructureLab(readout="event", seed=71)
    source_rows = source.acquire(source.random_probe(89, length=24)).observations
    parent = propose_structures(source.spec, source_rows, SearchBudget(30000))[0]
    changed = StructureLab(readout="focus", seed=71)
    rows = changed.acquire(changed.random_probe(89, length=24)).observations
    budget = SearchBudget(60000)
    entries = propose_structures(changed.spec, rows, budget, parent=parent)
    # Failed family attempts remain available for comparison; the API does
    # not promise to return the eventual winner before those attempts.
    found = min(entries, key=lambda entry: entry["training_loss"])
    assert any(entry["structure_search"] == "family_revision" and entry["training_loss"] > 1e-7
               for entry in entries)
    assert found["structure_search"] == "full"
    assert found["kind"] == "structural:local_exchange"
    assert found["training_loss"] < 1e-7
    fresh = changed.evaluation_rows(seed=165, length=48)
    assert loss(found["candidate"].program, fresh, 0., .5) < 1e-7
