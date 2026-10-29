"""A direct complete comparison retains the question that owns its paid scope."""

import json
from copy import deepcopy

import pytest

from hegel_machine.cognition.finite_native_author import _completed_choice, export_finite_native_author
from hegel_machine.cognition.paid_score_history import paid_comparison_scope_owner
from test_python_paid_score_reuse import MODELS, no_execution, scene


def test_direct_comparison_persists_question_scope_and_is_reused_without_scoring(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    question = next(q for q in agent.questions if q['id'] == 'q')
    question['meaning_version'] = 3
    returned = agent._execute('run', {'operator': 'research_choice', 'models': MODELS,
        'research_question': {'id': question['id']}}, lab)
    assert returned['models_completed'] == len(MODELS)
    assert not agent.method_results
    agent._prepare_program_state()
    saved = json.loads((agent.workspace / 'research_state.json').read_text())['operator_results'][-1]
    assert saved['research_question_id'] == returned['research_question_id'] == 'q'
    assert saved['question_meaning_version'] == returned['question_meaning_version'] == 3

    before, parent = deepcopy(agent.costs), deepcopy(agent.evaluated_model)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    actual = _completed_choice([], MODELS, len(agent.rows), question,
        {'questions': agent.questions, 'operator_results': agent.operator_results})
    assert actual is not None
    assert agent.costs == before and agent.evaluated_model == parent


@pytest.mark.parametrize('change', [None, 'default_meaning', 'meaning', 'rows', 'public_contract', 'missing_reference'])
def test_old_report_owner_comes_from_exact_paid_scopes_without_rewriting_history(
        tmp_path, monkeypatch, change):
    agent, lab = scene(tmp_path)
    question = next(q for q in agent.questions if q['id'] == 'q')
    agent._execute('run', {'operator': 'research_choice', 'models': MODELS,
        'research_question': {'id': 'q'}}, lab)
    report = deepcopy(agent.operator_results[-1])
    report.pop('research_question_id')
    report.pop('question_meaning_version')
    requested, rows, spec = deepcopy(question), list(agent.rows), lab.public_spec()
    if change == 'meaning':
        requested['meaning_version'] += 1
    elif change == 'default_meaning':
        requested.pop('meaning_version')
    elif change == 'rows':
        rows = rows[:-1]
    elif change == 'public_contract':
        spec = {**spec, 'prediction_contract': 'changed public interface'}
    elif change == 'missing_reference':
        report['evaluations'][0]['paid_score_reuse'].pop('scope_ref')
    before, costs = deepcopy(report), deepcopy(agent.costs)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    owner = paid_comparison_scope_owner(report, question=requested, rows=rows,
        public_spec=spec, workspace=agent.workspace)
    assert owner == ({'research_question_id': 'q', 'question_meaning_version': 1}
                     if change in (None, 'default_meaning') else None)
    assert report == before and agent.costs == costs


def test_standalone_author_reuses_old_direct_scope_without_package_context(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    question = next(q for q in agent.questions if q['id'] == 'q')
    agent._execute('run', {'operator': 'research_choice', 'models': MODELS,
        'research_question': {'id': 'q'}}, lab)
    report = deepcopy(agent.operator_results[-1])
    report.pop('research_question_id')
    report.pop('question_meaning_version')
    namespace = {'__name__': 'frozen_finite_author'}
    exec(export_finite_native_author(), namespace)
    before, costs = deepcopy(report), deepcopy(agent.costs)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    reused = namespace['_completed_choice']([], MODELS, len(agent.rows), question,
        {'questions': agent.questions, 'operator_results': [report]},
        paid_rows_view=agent.rows, public_spec=lab.public_spec(), workspace=agent.workspace)
    assert reused is not None and reused['research_question_id'] == question['id']
    assert report == before and agent.costs == costs
