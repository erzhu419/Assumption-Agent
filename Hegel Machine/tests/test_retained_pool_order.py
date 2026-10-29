"""A restored full pool keeps the paid order without weakening score guards."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.finite_native_author import _completed_choice
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.registered_source_scene import preserve_completed_pool_order


def scene():
    order = ['strong.py', 'native.py', 'revision.py']
    report = {'operator': 'research_choice', 'arguments': {
        'operator': 'research_choice', 'models': order},
        'research_question_id': 'q', 'question_meaning_version': 1,
        'paid_rows': 2, 'models_requested': 3, 'models_completed': 3,
        'evaluations': [{'path': name} for name in order]}
    return SimpleNamespace(rows=[None, None], operator_results=[report]), report


def test_same_full_pool_restores_paid_order_and_reuses_the_ordinary_report(monkeypatch):
    agent, report = scene()
    reconstructed = ['strong.py', 'revision.py', 'native.py']
    question = {'id': 'q', 'meaning_version': 1}
    state = {'questions': [question], 'operator_results': agent.operator_results}
    monkeypatch.setattr(Program, 'predict', lambda *_: pytest.fail('repeat scientific prediction'))
    before = deepcopy(report)
    assert _completed_choice([], reconstructed, 2, question, state) is None
    retained = preserve_completed_pool_order(agent, reconstructed)
    assert retained == report['arguments']['models']
    assert set(retained) == set(reconstructed)
    assert _completed_choice([], retained, 2, question, state) is report
    assert report == before


@pytest.mark.parametrize('change', [
    'new_candidate', 'missing_candidate', 'different_candidate', 'wrong_rows',
    'comparison_error', 'comparison_truncated', 'incomplete', 'different_evaluation_order',
    'missing_evaluation', 'truncated_evaluation', 'duplicate_paid_path',
])
def test_unmatched_or_incomplete_pool_does_not_borrow_an_old_order(change):
    agent, report = scene()
    reconstructed = ['strong.py', 'revision.py', 'native.py']
    if change == 'new_candidate':
        reconstructed.append('new.py')
    elif change == 'missing_candidate':
        reconstructed.pop()
    elif change == 'different_candidate':
        reconstructed[-1] = 'different.py'
    elif change == 'wrong_rows':
        report['paid_rows'] = 1
    elif change == 'comparison_error':
        report['error'] = 'preserved interrupted comparison'
    elif change == 'comparison_truncated':
        report['truncated'] = True
    elif change == 'incomplete':
        report['models_completed'] = 2
    elif change == 'different_evaluation_order':
        report['evaluations'].reverse()
    elif change == 'missing_evaluation':
        report['evaluations'].pop()
    elif change == 'truncated_evaluation':
        report['evaluations'][1]['truncated'] = True
    else:
        report['arguments']['models'][-1] = 'native.py'
    before = deepcopy(report)
    assert preserve_completed_pool_order(agent, reconstructed) is reconstructed
    assert report == before


def test_complete_comparison_keeps_failed_predictors_in_the_full_pool():
    agent, report = scene()
    report['evaluations'][1].update(returncode=1, error='retained source failure')
    reconstructed = ['strong.py', 'revision.py', 'native.py']
    assert preserve_completed_pool_order(agent, reconstructed) == ['strong.py', 'native.py', 'revision.py']
    assert report['evaluations'][1]['error'] == 'retained source failure'
