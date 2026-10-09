"""Paid independent responses can justify, but do not finish, new research."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.mechanism_revision import propose_response_update
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec


def pilot_evidence():
    feature = Program('feature', name='input')
    source = Definition('measured_response', (), feature, source_concept_id='learned_basis')
    program = Program('call', name=source.definition_id, definition=source)
    parent = {'candidate': Candidate('incumbent', program, 'research_program')}
    spec = TaskSpec('response', 'Predict measured response', ('input',), (), ())
    training = [Observation({'input': float(index)}, 1.2 * index)
                for index in [*range(-12, 0), *range(1, 13)]]
    original = [Observation(dict(row.features), 1.6 * row.features['input']) for row in training]
    repeated = [replace(row, target=1.45 * row.features['input']) for row in original]
    return spec, parent, training, original, repeated


def test_original_updates_acquired_basis_but_repeat_only_selects_research_opportunity():
    spec, parent, training, original, repeated = pilot_evidence()
    repeated = [replace(row, target=1.205 * row.features['input'] + 1.) for row in repeated]
    before = deepcopy(parent)
    # The same observation object is already in training; it is not another
    # paid draw. The independent repeat remains excluded from the fitted rows.
    training.append(original[0])
    budget = SearchBudget(4000)
    result = propose_response_update(spec, training, original, repeated, budget, parent=parent)
    candidate, event = result['selected'], result['event']
    assert candidate is not None
    assert candidate['kind'] == 'mechanism_full_refit'
    assert candidate['candidate'].program.predict(Observation({'input': 3.}, 0.)) == pytest.approx(4.2)
    assert candidate['candidate'].parents == ('learned_basis',)
    assert candidate['method_source_ids'] == ['learned_basis']
    assert candidate['mechanism_revision']['original_program'] == parent['candidate'].program.to_dict()
    assert candidate['mechanism_revision']['status'] == 'proposed'
    assert candidate['selection_basis'] == 'repeated_response_pilot'
    assert candidate['validation_loss'] == event['after_loss']
    assert event['training_rows'] == 48 and event['heldout_rows'] == 24
    assert 1e-7 < event['gain'] < event['margin'] and event['requires_fresh_confirmation']
    assert event['compute_cost'] == budget.spent > 4 * len(repeated)
    assert parent == before

    changed_repeat = [replace(row, target=2. * row.features['input']) for row in repeated]
    changed = propose_response_update(spec, training, original, changed_repeat,
                                     SearchBudget(4000), parent=parent)
    assert changed['selected']['candidate'].program == candidate['candidate'].program
    assert changed['event']['gain'] != event['gain']


def test_repeat_rejects_update_that_only_improves_its_fitting_acquisition():
    spec, parent, training, original, repeated = pilot_evidence()
    repeated = [replace(row, target=row.features['input']) for row in repeated]
    result = propose_response_update(spec, training, original, repeated, SearchBudget(4000), parent=parent)
    assert result['selected'] is None
    assert result['event']['gain'] < 0.
    assert result['event']['reason'] == 'no_positive_pilot_gain'
    assert not result['event']['requires_fresh_confirmation']


@pytest.mark.parametrize('limit', [0, 24, 96])
def test_unfunded_update_is_not_recommended_without_its_pilot_comparison(limit):
    spec, parent, training, original, repeated = pilot_evidence()
    budget = SearchBudget(limit)
    result = propose_response_update(spec, training, original, repeated, budget, parent=parent)
    assert result['selected'] is None
    assert result['event']['reason'] == 'insufficient_budget'
    assert result['event']['gain'] is None
    assert result['event']['compute_cost'] == budget.spent <= limit
