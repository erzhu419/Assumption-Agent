"""A method that searches a pool must see paid evidence beyond its incumbent."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_conditions import _method_observables
from hegel_machine.cognition.method_feedback import begin_method_feedback, paid_pool_transition_evidence
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_python_paid_score_reuse import scene, choose, no_execution


def prepared_pool(tmp_path):
    agent, lab = scene(tmp_path)
    control = Program('feature', name='u')
    guard = Program('available', lag=1)
    best = Program('if', children=(guard, Program('const', value=1.), control))
    alternative = Program('if', children=(guard, control, Program('const', value=0.)))
    for name, program in [('best.py', best), ('alternative.py', alternative),
                          ('alias.py', alternative), ('unguarded.py', Program('const', value=0.))]:
        agent._save(name, export_temporal_program(program))
    agent._save('unscored.py', export_temporal_program(control))
    choose(agent, lab, ['best.py', 'alternative.py', 'alias.py', 'unguarded.py'])
    assert agent.evaluated_model['source_model'] == 'best.py'
    assert agent.operator_results[-1]['evaluations'][2]['paid_score_reuse']['status'] == 'reused'
    agent.knowledge['finite_native_author'] = {'predictor_paths': [
        'best.py', 'alternative.py', 'alias.py', 'unguarded.py', 'unscored.py']}
    candidate = {'kind': 'python_research_method'}
    agent.theory_consolidation.setdefault('candidates', {})['fixture'] = candidate
    return agent, lab, candidate


def test_equal_score_alternative_has_different_paid_repair_diagnostic_without_reexecution(tmp_path, monkeypatch):
    agent, lab, candidate = prepared_pool(tmp_path)
    costs, rows = deepcopy(agent.costs), deepcopy(agent.rows)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    monkeypatch.setattr(agent, '_score', no_execution)
    profile = begin_method_feedback(agent, lab, candidate=candidate)['profile']
    assert profile['residual']['transition']['mature_first_divergences'] == 1
    assert profile['residual']['transition']['startup_first_divergences'] == 0
    pool = profile['research']['candidate_pool']
    assert pool['models_requested'] == 5
    assert pool['known_executions'] == 3 and pool['unknown_prediction_models'] == 1
    assert pool['guarded_executions'] == 2 and pool['unknown_guard_executions'] == 1
    assert pool['min_mature_first_divergences'] == 0
    assert pool['startup_only_first_error_executions'] == 1
    alternative = next(item for item in pool['witnesses'] if 'alternative.py' in item['models'])
    assert alternative['models'] == ['alternative.py', 'alias.py']
    assert alternative['transition']['startup_first_divergences'] == 1
    assert alternative['transition']['mature_first_divergences'] == 0
    assert alternative['execution_ref'] and alternative['transition']['source_refs']
    assert pool['unknown_models'] == ['unscored.py']
    coordinates = _method_observables(profile)
    assert coordinates[('research', 'candidate_pool', 'startup_only_first_error_executions')] == 1
    assert not any('models' in path or 'witnesses' in path or 'source_refs' in path for path in coordinates)
    assert agent.costs == costs and agent.rows == rows
    agent._method_use_profile = profile
    monkeypatch.setattr(agent, '_python_research_method', lambda path: {
        'selection_plan': {'ordinary_method_path': 'ordinary.py'}})
    agent._save('method.py', 'def select_actions(public_spec, observations, memory): return {"action": "finish"}\n')
    memory = agent._program_memory('method.py', {})
    assert memory['method_candidate_pool'] == pool
    memory['method_candidate_pool']['witnesses'].clear()
    assert profile['research']['candidate_pool']['witnesses']
    assert agent.costs == costs and agent.rows == rows


@pytest.mark.parametrize('changed', ['rows', 'source'])
def test_changed_paid_scope_or_model_remains_unknown_in_pool(tmp_path, changed):
    agent, lab, _ = prepared_pool(tmp_path)
    if changed == 'rows':
        agent.rows[0] = replace(agent.rows[0], target=99.)
    else:
        agent._save('alternative.py', export_temporal_program(Program('const', value=9.)))
    pool = paid_pool_transition_evidence(agent, lab.public_spec(), models=['alternative.py'])
    assert pool['status'] == 'unknown' and pool['known_executions'] == 0
    assert pool['unknown_prediction_models'] == 1
    assert pool['min_mature_first_divergences'] is None
    assert pool['startup_only_first_error_executions'] is None


def test_ordinary_python_paid_score_remains_usable_without_a_literal_program_guard(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    agent._save('ordinary.py', 'def predict(row): return row["features"]["u"]\n')
    scored = choose(agent, lab, ['ordinary.py'])
    assert scored['evaluations'][0]['mse'] == 0.
    assert scored['evaluations'][0]['paid_score_reuse']['execution_basis_ref']
    before = deepcopy(agent.costs)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    monkeypatch.setattr(agent, '_score', no_execution)
    pool = paid_pool_transition_evidence(agent, lab.public_spec(), models=['ordinary.py'])
    # The ordinary scorer saved its score, not a complete closed Program
    # prediction column. Its unknown diagnostic cannot erase that valid score.
    assert pool['status'] == 'unknown' and pool['unknown_prediction_models'] == 1
    assert pool['min_mature_first_divergences'] is None
    assert agent._evaluated_summary()['mse'] == 0. and agent.costs == before
