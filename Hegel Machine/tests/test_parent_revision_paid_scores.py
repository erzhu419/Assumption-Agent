"""A new parent duty reuses its unchanged prediction-task scores."""

from copy import deepcopy
from dataclasses import replace

import pytest

from test_literal_program_score_reuse import MODEL, scene
from test_python_paid_score_reuse import no_execution, scoring_costs


def revision_scene(tmp_path, monkeypatch):
    agent, lab, calls = scene(tmp_path, monkeypatch)
    source = next(q for q in agent.questions if q['id'] == agent._active_question_id)
    source['theory_target'] = {
        'role_signature': {'u': {'role': 'public_control', 'unit': '1', 'time': 'current'}},
        'output_unit': '1', 'context_ref': deepcopy(source.get('context_ref'))}
    reference = {'question_id': source['id'],
                 'meaning_version': source.get('meaning_version', 1)}
    duty = agent._open_question({'id': 'parent_revision:fixture',
        'question': 'Revise the retained parent on the same paid prediction task.',
        'kind': 'program_revision', 'answer_type': 'program',
        'query_semantics': {'mode': 'solution_relation'},
        'context_ref': deepcopy(source.get('context_ref')),
        'theory_target': {**deepcopy(source['theory_target']), 'parent_theory_id': 'parent195'},
        'native_research_ref': deepcopy(reference),
        'paid_score_question_ref': deepcopy(reference)})
    return agent, lab, calls, source, duty


def compare_duty(agent, lab, duty):
    return agent._execute('run', {'operator': 'research_choice', 'models': [MODEL],
        'research_question': {'id': duty['id']}}, lab)


def test_parent_duty_reuses_paid_score_but_keeps_its_actual_research_attribution(
        tmp_path, monkeypatch):
    agent, lab, calls, source, duty = revision_scene(tmp_path, monkeypatch)
    original = deepcopy(source)
    before = scoring_costs(agent)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model',
                        no_execution)
    compared = compare_duty(agent, lab, duty)
    assert compared['reused_score_count'] == 1 and compared['new_scored_count'] == 0
    assert compared['research_question_id'] == duty['id']
    assert scoring_costs(agent) == before and calls == [MODEL]
    assert source == original and agent._active_question_id == duty['id']


@pytest.mark.parametrize('change', ['rows', 'model_parameters', 'public_spec'])
def test_parent_score_task_reference_preserves_real_execution_invalidation(
        tmp_path, monkeypatch, change):
    agent, lab, calls, source, duty = revision_scene(tmp_path, monkeypatch)
    if change == 'rows':
        agent.rows[0] = replace(agent.rows[0], target=7.)
    elif change == 'model_parameters':
        agent._save('frozen/_evaluated_model/params.json', '{"unused_fixture_parameter":1}')
    else:
        previous = lab.public_spec
        lab.public_spec = lambda: {**previous(), 'prediction_contract': 'changed'}
    before = agent.costs['compute_cost']
    compared = compare_duty(agent, lab, duty)
    assert compared['reused_score_count'] == 0 and compared['new_scored_count'] == 1
    assert agent.costs['compute_cost'] - before == 97 and len(calls) == 2


@pytest.mark.parametrize('change', ['source_meaning', 'context', 'roles', 'source_reference'])
def test_parent_score_reference_requires_its_declared_same_prediction_task(
        tmp_path, monkeypatch, change):
    agent, lab, calls, source, duty = revision_scene(tmp_path, monkeypatch)
    if change == 'source_meaning':
        source['meaning_version'] += 1
    elif change == 'context':
        duty['context_ref'] = {'scene': 'different'}
    elif change == 'roles':
        duty['theory_target']['role_signature']['u']['unit'] = 'changed'
    else:
        duty['native_research_ref']['question_id'] = 'different'
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model',
                        no_execution)
    with pytest.raises(ValueError, match='paid-score task no longer matches'):
        compare_duty(agent, lab, duty)
    assert calls == [MODEL]
