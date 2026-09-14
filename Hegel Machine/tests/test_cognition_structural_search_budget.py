"""Spend local relation search on the explanation supported by paid samples."""

import json

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_structure import propose_structures


def test_leading_relation_refinement_reaches_the_previously_starved_measured_boundary():
    # Replay the two public random probes already acquired by the failing event
    # task. Its correct boundary was 13th in local order, but the old search
    # spent the radius budget on seven different models at every local point.
    task_seed = 3050339151
    lab = UnifiedObjectLab(seed=task_seed, apparatus_seed=6131, readout="event")
    spec = lab.spec_for(("fine",))
    batches = [lab.acquire(lab.random_probe(task_seed + 1009 * serial, length=24), ("fine",))
               for serial in (5, 6, 7)]
    training = batches[0].observations + batches[1].observations
    transitions = batches[0].transitions + batches[1].transitions
    budget = SearchBudget(28000)
    entries = propose_structures(spec, training, budget, transitions=transitions, tolerance=1e-7)
    selected = entries[0]
    assert selected["kind"] == "structural:event_change"
    assert selected["training_loss"] < 1e-7
    assert selected["structure"]["radius_interval"]
    assert budget.spent < 24000
    # The existing third probe still constrains the explanation; it is not
    # silently treated as a passing confirmation of the initial radius.
    combined = training + batches[2].observations
    refinement = SearchBudget(12000)
    revised = propose_structures(spec, combined, refinement, parent=selected,
                                 transitions=transitions + batches[2].transitions, tolerance=1e-7)[0]
    assert revised["structure_search"] == "boundary_revision"
    assert loss(revised["candidate"].program, combined, 0., .5) < 1e-7
    assert refinement.spent < 12000
    assert ObjectSchema.from_dict(revised["structure"]["schema"]) == ObjectSchema.from_dict(
        json.loads(spec.context["object_schema"]))
