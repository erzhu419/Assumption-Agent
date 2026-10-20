"""Frozen local predictions must face later evidence without old-fit dilution."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import observable_research
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_common_archive_observable_research import MeasuredLab
from test_inquiry_revision import action


VIEW = '''def bind(spec, observations, memory):
    return {'rows': [dict(row, evidence=[index]) for record in observations
        for index, row in zip(record['row_indices'], record['rows'])]}
'''


def prepared(tmp_path, kind):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='hegel', source_task=False,
        config=InquiryLLMConfig(observation_budget=100, confirmation_rows=1,
            execution_seconds=30., compute_budget=10000, confirmation_mse_threshold=1e-6))
    lab = MeasuredLab()
    agent._execute('query', {'actions': [action(*([0.] * 64), channels=['meter'])]}, lab)
    if kind == 'observable':
        arguments = {'target': 'meter', 'model': 'local.py',
                     'code': "def predict(row): return 2*row['features']['u']+1.\n"}
        pool, key = agent.observable_models, 'meter'
        parent = '''def bind_models(models):
    global local
    local = models['local']
def predict(row): return 3*local(row)+row['features']['u']
'''
    else:
        agent._execute('write', {'path': 'view.py', 'code': VIEW}, lab)
        arguments = {'component': 'local_relation', 'view': 'view.py', 'model': 'local.py',
                     'code': "def predict(row): return 7*row['features']['u']+3.\n"}
        pool, key = agent.conditional_models, 'local_relation'
        parent = '''def bind_models(models):
    global local
    local = models['local']
def predict(row): return local(row)
'''
    agent._execute('evaluate', arguments, lab)
    frozen = pool[key]['model']
    agent._execute('evaluate', {'model': 'parent.py', 'code': parent,
                              'bindings': {'local': frozen}}, lab)
    return agent, lab, pool, key, arguments


@pytest.mark.parametrize('kind', ['observable', 'conditional'])
def test_later_counterexample_refutes_and_reopens_parent_in_one_scoring_run(tmp_path, monkeypatch, kind):
    agent, lab, pool, key, arguments = prepared(tmp_path, kind)
    previous = deepcopy(pool[key])
    evaluated = []
    original = observable_research.evaluate_python_model

    def score_once(*args, **kwargs):
        evaluated.append((args[1], kwargs.get('training_observations')))
        return original(*args, **kwargs)

    monkeypatch.setattr(observable_research, 'evaluate_python_model', score_once)
    compute, paid = agent.costs['compute_cost'], lab.paid
    lab.offset = 1.002
    result = agent._execute('query', {'actions': [action(2., channels=['meter'])]}, lab)
    entry = pool[key]
    assert len(evaluated) == 1 and evaluated[0] == (previous['model'], None)
    assert agent.costs['compute_cost'] == compute + 65 and lab.paid == paid + 1
    assert entry['mse'] < 1e-6  # The former all-row support test would pass.
    assert entry['later_readings']['rows'] == 1
    error = .002 if kind == 'observable' else .006
    assert entry['later_readings']['mse'] == pytest.approx(error ** 2)
    assert entry['later_readings']['max_abs_error'] == pytest.approx(error)
    assert entry['support'] == 'refuted_on_paid_rows'
    assert entry['model'] == previous['model']
    assert entry['training_observation_indices'] == previous['training_observation_indices']
    feedback_key = 'observable_reassessments' if kind == 'observable' else 'component_reassessments'
    assert result[feedback_key][0]['evaluation']['later_readings'] == entry['later_readings']
    agent._request(lab)
    assert agent.research_candidates['parent.py']['status'] == 'needs_rebuild'
    blocked = agent._execute('evaluate', {'model': 'parent.py'}, lab)
    assert blocked['status'] == 'dependency_unresolved'

    # Explicitly rescoring the same retained file cannot relabel its counterexample as fitting data.
    again = {name: value for name, value in arguments.items() if name not in {'code', 'view'}}
    agent._execute('evaluate', {**again, 'model': previous['model']}, lab)
    assert entry['support'] == 'refuted_on_paid_rows'
    assert entry['later_readings']['mse'] == pytest.approx(error ** 2)


@pytest.mark.parametrize('kind', ['observable', 'conditional'])
def test_revised_fit_starts_own_later_evidence_without_rewriting_old_version(tmp_path, kind):
    agent, lab, pool, key, arguments = prepared(tmp_path, kind)
    frozen = pool[key]['model']
    lab.offset = 1.002
    agent._execute('query', {'actions': [action(2., channels=['meter'])]}, lab)
    source = arguments['code'].replace('+1.', '+1.+(.002 if row["features"]["u"]>1 else 0.)')
    if kind == 'conditional':
        source = arguments['code'].replace('+3.', '+3.+(.006 if row["features"]["u"]>1 else 0.)')
    result = agent._execute('evaluate', {**arguments, 'model': 'revised.py', 'code': source}, lab)
    current = pool[key]
    assert result['mse'] < 1e-25 and current['model'] != frozen
    assert current['support'] == 'paid_fit' and 'later_readings' not in current
    assert current['training_observation_indices'] == list(range(65))
    agent._execute('query', {'actions': [action(3., channels=['meter'])]}, lab)
    assert current['support'] == 'supported_on_later_readings'
    assert current['later_readings']['rows'] == 1
    assert current['later_readings']['mse'] < 1e-25
    assert (tmp_path / frozen).is_file()


def test_conditional_contrast_citing_old_and_new_rows_counts_as_later_evidence(tmp_path):
    agent, lab, pool, key, arguments = prepared(tmp_path, 'conditional')
    # The new view's rows share an old reference but each still cites its own observed label.
    mixed_view = VIEW.replace('evidence=[index]', 'evidence=[0, index]')
    agent._execute('write', {'path': 'contrast_view.py', 'code': mixed_view}, lab)
    agent._execute('evaluate', {**arguments, 'component': 'contrast', 'view': 'contrast_view.py'}, lab)
    lab.offset = 1.002
    agent._execute('query', {'actions': [action(2., channels=['meter'])]}, lab)
    contrast = pool['contrast']
    assert contrast['mse'] < 1e-6 and contrast['support'] == 'refuted_on_paid_rows'
    assert contrast['later_readings']['rows'] == 1
    assert contrast['later_readings']['mse'] == pytest.approx(.006 ** 2)
