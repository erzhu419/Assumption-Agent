"""Extra evidence-consumption work must fit the caller's real compute slice."""

from copy import deepcopy
from dataclasses import replace

import pytest

from test_executable_evidence_binding import return_local_checks, setup
from test_inquiry_revision import action


PLAIN = '''def bind_research(inputs): pass
def fit(rows): pass
def predict(row):
    with open('prediction_calls.jsonl', 'a') as stream:
        stream.write('1\\n')
    return row['features']['u']
'''


def bound_model(tmp_path):
    agent, lab = setup(tmp_path)
    record = return_local_checks(agent, lab, expected=2.)
    agent._execute('write', {'path': 'plain.py', 'code': PLAIN}, lab)
    agent._execute('run', {'operator': 'bind_research', 'model': 'plain.py',
        'inputs': {'contrast': {'revision_index': record['index'],
                                'check_ids': ['difference']}}}, lab)
    return agent, lab


@pytest.mark.parametrize('allowance', [4, 5, 6, 7, 8])
def test_paid_scoring_bounds_witnesses_and_relation_terms_without_clipping_cost(tmp_path, allowance):
    agent, lab = bound_model(tmp_path)
    before, paid = agent.costs['compute_cost'], lab.paid
    agent.config = replace(agent.config, compute_budget=before + allowance)

    result = agent._execute('evaluate', {'model': 'plain.py'}, lab)

    # Four candidate-row units are the existing attempt charge; the two
    # witness predictions and two diagnostic terms consume only their balance.
    assert agent.costs['compute_cost'] - before == allowance
    assert result['research_prediction_count'] == min(2, allowance - 4)
    assert lab.paid == paid
    calls = tmp_path / 'prediction_calls.jsonl'
    assert (len(calls.read_text().splitlines()) if calls.exists() else 0) == (
        allowance - 4 if allowance < 6 else 6)
    if allowance < 6:
        assert result['status'] == 'scoped_unknown' and result['truncated']
        assert result['reason'] == 'research_compute_budget_exhausted'
        assert 'mse' not in result and result['evaluated_model'] is None
        check = result['research_bindings'][0]['local_relations'][0]
        assert check['checks'] == []
        assert check['pending_check']['completed_witnesses'] == allowance - 4
        assert agent.research_candidates['plain.py']['status'] == 'evaluation_inconclusive_budget'
    else:
        assert result['mse'] == 0. and not result.get('truncated')
        assert result['evaluated_model']['source_model'] == 'plain.py'
        relations = result['paid_relation_diagnostic']
        assert relations['term_count'] == allowance - 6
        assert relations['truncated'] is (allowance < 8)
        assert relations['check_count'] == (1 if allowance == 8 else 0)
        if allowance == 7:
            assert relations['pending_check']['completed_terms'] == 1
            assert len(relations['pending_check']['terms']) == 1


def test_zero_research_balance_still_scores_an_unbound_model(tmp_path):
    agent, lab = setup(tmp_path)
    agent._execute('write', {'path': 'unbound.py',
        'code': "def predict(row): return row['features']['u']\n"}, lab)
    before = agent.costs['compute_cost']
    agent.config = replace(agent.config, compute_budget=before + len(agent.rows))

    result = agent._execute('evaluate', {'model': 'unbound.py'}, lab)

    assert result['mse'] == 0. and result['research_prediction_count'] == 0
    assert agent.costs['compute_cost'] - before == len(agent.rows)
    assert result['evaluated_model']['source_model'] == 'unbound.py'


def test_incomplete_incumbent_rescore_does_not_install_the_other_candidate(tmp_path):
    agent, lab = bound_model(tmp_path)
    agent._execute('write', {'path': 'plain.py', 'code':
        'def bind_research(inputs): pass\ndef predict(row): return 0.\n'}, lab)
    agent._execute('evaluate', {'model': 'plain.py'}, lab)
    incumbent = deepcopy(agent.evaluated_model)
    agent._execute('query', {'actions': [action(.75, .25)]}, lab)
    agent._execute('write', {'path': 'better.py',
        'code': "def predict(row): return row['features']['u']\n"}, lab)
    before = agent.costs['compute_cost']
    # New score: six rows and two optional relation terms. Remaining six units
    # start the incumbent's ordinary score, leaving no witness allowance.
    agent.config = replace(agent.config, compute_budget=before + 14)

    result = agent._execute('evaluate', {'model': 'better.py'}, lab)

    assert result['mse'] == 0.
    assert result['retained_comparison']['truncated']
    assert result['retention_decision']['status'] == 'deferred'
    assert agent.evaluated_model == incumbent
    assert agent.costs['compute_cost'] - before == 14


@pytest.mark.parametrize('allowance', [6, 7, 8])
def test_refit_validation_does_not_export_a_candidate_with_unfinished_binding(tmp_path, allowance):
    agent, lab = bound_model(tmp_path)
    agent._execute('query', {'actions': [action(.75, .25)]}, lab)
    source = (tmp_path / 'plain.py').read_text()
    before, paid = agent.costs['compute_cost'], lab.paid
    agent.config = replace(agent.config, compute_budget=before + allowance)

    result = agent._execute('run', {'operator': 'refit', 'model': 'plain.py',
        'validation_records': [2], 'path': 'refitted'}, lab)

    assert result['split']['train_rows'] == 4 and result['split']['holdout_rows'] == 2
    assert result['validation']['research_prediction_count'] == allowance - 6
    assert result['compute_cost'] == allowance
    assert agent.costs['compute_cost'] - before == allowance
    assert lab.paid == paid and (tmp_path / 'plain.py').read_text() == source
    # An unfinished isolated context is recovery state, not an admitted model.
    assert (tmp_path / result['tested_model']).exists()
    if allowance < 8:
        assert result['status'] == 'refit_inconclusive_budget' and result['path'] is None
        assert result['validation']['truncated'] and 'mse' not in result['validation']
        assert result['tested_model'] not in agent.candidate_paths
    else:
        assert result['status'] == 'refit_supported_on_validation'
        assert result['validation']['mse'] == 0. and result['path'] in agent.candidate_paths


def test_finish_does_not_buy_confirmation_after_incomplete_bound_input_validation(tmp_path):
    agent, lab = bound_model(tmp_path)
    before, paid = agent.costs['compute_cost'], lab.paid
    agent.config = replace(agent.config, compute_budget=before + 5)

    result = agent._execute('finish', {'model': 'plain.py'}, lab)

    assert result['truncated'] and 'fresh_confirmation' not in result
    assert lab.paid == paid and not agent.confirmations
    assert agent.costs['compute_cost'] - before == 5
