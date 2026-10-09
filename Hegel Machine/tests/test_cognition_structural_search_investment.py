"""Paid trajectories determine which structural search deserves refinement."""

from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.structural_discovery import discover_structure
from hegel_machine.cognition.structure_lab import StructureLab
from hegel_machine.cognition.unified_structure import propose_structures


@pytest.mark.parametrize('readout', ['focus', 'event', 'window'])
def test_budgeted_search_resolves_an_edge_in_many_paid_distances(readout):
    lab = StructureLab(readout=readout)
    rows = [row for seed in (89, 90, 91)
            for row in lab.acquire(lab.random_probe(seed, length=24)).observations]
    budget = SearchBudget(10000)
    found = discover_structure(rows, lab.object_schema, budget=budget)
    assert found.fits[0]['training_loss'] < 1e-8
    assert loss(found.candidates[0].program, lab.evaluation_rows(seed=165, length=48), 0., .5) < 1e-8
    assert 0 < budget.spent <= budget.limit


def test_wrong_parent_family_leaves_budget_for_a_new_response():
    source = StructureLab(readout='event')
    rows = source.acquire(source.random_probe(89, length=24)).observations
    spec = replace(source.spec, feature_names=source.object_schema.feature_names,
                   context={'object_schema': json.dumps(source.object_schema.to_dict())})
    parent = min(propose_structures(spec, rows, SearchBudget(28000)),
                 key=lambda entry: entry['training_loss'])
    target = StructureLab(readout='focus')
    paid = target.acquire(target.random_probe(91, length=24)).observations
    budget = SearchBudget(16000)
    entries = propose_structures(spec, paid, budget, parent=parent)
    # Refuting the old event family must not discard its partial results or
    # consume the full budget before the current-response family can compete.
    assert any(entry['structure_search'] == 'family_revision' for entry in entries)
    changed = min(entries, key=lambda entry: entry['training_loss'])
    assert changed['kind'] == 'structural:local_exchange'
    assert loss(changed['candidate'].program, target.evaluation_rows(seed=165, length=48), 0., .5) < 1e-8
    assert len({entry['candidate'].candidate_id for entry in entries}) == len(entries)
    assert 0 < budget.spent <= budget.limit
