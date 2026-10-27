"""Real Common fits/comparisons feed method decisions without extra science."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.finite_native_author import export_revision_candidates
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_feedback import _read
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.theory_consolidation import install_theory_library
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_research_program_consolidation import compile_sources
from test_theory_consolidation import agent_scene
from test_inquiry_revision import PublicLab


def scene(path, *, useful):
    learner, result = compile_sources(path / 'source')
    agent, lab = agent_scene(path / 'target')
    lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
        'feature_bounds': {'u': [-10., 10.]}, 'max_probe_length': 8}
    agent._save('shared/theory_library.json', (learner.workspace / result['library_path']).read_text())
    agent.theory_consolidation = install_theory_library(agent)
    method, = agent.theory_consolidation['candidates'].values()
    for record in range(2):
        rows = [{'features': {'u': i + record * .1}, 'history': [],
            'target': 2 * (i + record * .1) + 1 if useful else (i + record * .1) ** 2}
            for i in range(8)]
        agent.observations.append({'action': {'kind': 'trajectory',
            'sequence': [row['features'] for row in rows]}, 'rows': rows,
            'observation_index': len(agent.rows), 'observation_cost': len(rows), 'measurement_cost': 0,
            'row_indices': list(range(len(agent.rows), len(agent.rows) + len(rows)))})
        agent.rows.extend(Observation(row['features'], row['target']) for row in rows)
    agent.costs['observation_cost'] = len(agent.rows)
    agent._write_evidence()
    old = Program('const', value=0.) if useful else Program('mul', children=(
        Program('feature', name='u'), Program('feature', name='u')))
    agent._save('old.py', export_temporal_program(old, feature_names=('u',)))
    agent._execute('evaluate', {'model': 'old.py'}, lab)
    question = {'id': 'public_response', 'question': 'Predict the acquired public response',
        'parent_id': 'task', 'depends_on': [], 'theory_target': {'role_signature': {
            'u': deepcopy(method['role_signature']['role_0'])},
            'output_unit': method['output_unit'], 'context_ref': {'apparatus': 'public'},
            'context_family_ref': method['scope']['context_ref'],
            'research_program_fitting': {'validation_records': [1]}}}
    question = agent._open_question(question)
    return agent, lab, method, question


def use_and_export(agent, lab, question):
    route = agent._execute('run', {'operator': 'research_route', 'research_question': question,
        'allocation': {'compute': 3000}}, lab)
    owner = next(item for item in agent.questions if item['id'] == question['id'])
    exports = export_revision_candidates(agent, owner)
    models = ['old.py', *[item['model'] for owner in agent.questions
        for item in owner.get('revision_candidate_exports', [])]]
    assert len(models) > 1
    return route, models


def use_and_compare(agent, lab, question):
    route, models = use_and_export(agent, lab, question)
    comparison = agent._execute('run', {'operator': 'research_choice', 'models': models,
        'research_question': {'id': question['id']}}, lab)
    return route, comparison


def freeze_usage(agent, lab, method):
    return agent._execute('run', {'operator': 'theory_consolidation',
        'object_type': 'method_use_condition', 'candidate_ids': [method['id']],
        'mission': {'purpose': 'method_learning', 'theory_learning_allocation': {'compute': 10}},
        'allocation': {'compute': 10}}, lab)


def transfer_arguments(method, question, identifier):
    return {**deepcopy(question['theory_target']), 'candidate_id': method['id'],
        'theory_id': identifier, 'mapping': {'role_0': 'u'},
        'allocation': {'compute': 3000}, 'operator': 'theory_transfer',
        'research_question': {'id': question['id']}}


def assert_unscored_return(group):
    returns = group['returns']
    assert returns['origins'] and all(item['status'] == 'unscored' for item in returns['origins'])
    assert all(record['trials'] == 0 for context in returns['action_experience'].values()
               for record in context.values())


@pytest.mark.parametrize('useful', [False, True])
def test_common_paid_return_freezes_conditions_and_changes_same_scene_method_action(tmp_path, useful):
    agent, lab, method, question = scene(tmp_path, useful=useful)
    route, comparison = use_and_compare(agent, lab, question)
    group = agent.theory_consolidation['method_feedback'][method['id']]
    assert len(group['outcome_refs']) == 1
    outcome = _read(agent, group['outcome_refs'][0])
    assert outcome['status'] == 'completed'
    assert outcome['working_comparison']['complete']
    assert outcome['working_comparison']['same_rows']
    assert (outcome['working_comparison']['after_loss'] <
            outcome['working_comparison']['before_loss']) is useful
    assert not outcome['independent_confirmation']
    before = deepcopy(method)
    costs = deepcopy(agent.costs)
    learned = agent._execute('run', {'operator': 'theory_consolidation',
        'object_type': 'method_use_condition', 'candidate_ids': [method['id']],
        'mission': {'purpose': 'method_learning', 'theory_learning_allocation': {'compute': 10}},
        'allocation': {'compute': 10}}, lab)
    current = agent.theory_consolidation['candidates'][method['id']]
    assert current['version'] == before['version'] + 1
    assert _read(agent, current['previous_version_ref']) == before
    assert current['research_program_contract'] == before['research_program_contract']
    assert agent.costs['compute_cost'] == costs['compute_cost'] + 1
    # This is the ordinary production transfer consumer, not a test-only gate.
    args = {**deepcopy(question['theory_target']), 'candidate_id': method['id'],
        'theory_id': 'next_method_visit', 'mapping': {'role_0': 'u'},
        'allocation': {'compute': 3000}, 'operator': 'theory_transfer',
        'research_question': {'id': question['id']}}
    paid = agent.costs['compute_cost']
    result = agent._execute('run', args, lab)
    if not useful:
        assert result['status'] == 'method_use_deferred'
        assert result['method_use_decision']['next_action'] == 'defer_until_new_evidence'
        assert agent.costs['compute_cost'] == paid
    else:
        assert result['generated_theories']
    assert lab.paid == 0 and not agent.confirmations
    assert learned['support_updated'] is False


def test_common_restore_retains_unhelpful_use_and_only_new_paid_evidence_reopens_it(tmp_path):
    agent, lab, method, question = scene(tmp_path, useful=False)
    use_and_compare(agent, lab, question)
    freeze_usage(agent, lab, method)
    group = agent.theory_consolidation['method_feedback'][method['id']]
    reference, = group['outcome_refs']
    original = _read(agent, reference)
    original_method = deepcopy(agent.theory_consolidation['candidates'][method['id']])
    agent._prepare_program_state()
    previous = agent.result('researching')
    costs = deepcopy(agent.costs)
    restored, _ = agent_scene(tmp_path / 'restored')
    fresh_lab = PublicLab()
    fresh_lab.costs = {'observation_cost': 0}
    fresh_lab.public_spec = lab.public_spec
    receipt = restored.restore_practice_research(previous, agent.workspace, fresh_lab, confirmation_seed=17)
    assert receipt['new_observations'] == 0 and fresh_lab.paid == 0
    assert restored.costs == costs
    assert restored.theory_consolidation['candidates'][method['id']] == original_method
    current = next(item for item in restored.questions if item['id'] == question['id'])
    deferred = restored._execute('run', transfer_arguments(method, current, 'restored_same_paid_use'), fresh_lab)
    assert deferred['status'] == 'method_use_deferred'
    assert deferred['method_use_decision']['next_action'] == 'defer_until_new_evidence'
    assert restored.costs == costs
    assert len(restored.theory_consolidation['method_feedback'][method['id']]['outcome_refs']) == 1

    # This fixture acquires the same quadratic response as the retained scene.
    execute = fresh_lab.execute

    def acquire(action):
        record = execute(action)
        for row in record['rows']:
            row['target'] = row['features']['u'] ** 2
        record['observation_index'] = len(restored.rows)
        record['row_indices'] = list(range(len(restored.rows), len(restored.rows) + len(record['rows'])))
        return record

    fresh_lab.execute = acquire
    acquired = restored._execute('query', {'actions': [{'kind': 'trajectory',
        'sequence': [{'u': 8.2}, {'u': 9.2}]}], 'research_question': {'id': current['id']}}, fresh_lab)
    assert acquired['observations'] and fresh_lab.paid == 2
    paid = restored.costs['compute_cost']
    explored = restored._execute('run', transfer_arguments(method, current, 'restored_new_paid_use'), fresh_lab)
    assert explored['generated_theories'] and explored['new_compute_cost'] > 0
    assert restored.costs['compute_cost'] - paid == explored['new_compute_cost'] <= 3000
    assert restored.costs['observation_cost'] == costs['observation_cost'] + 2
    assert _read(agent, reference) == original
    assert original['working_comparison']['after_loss'] >= original['working_comparison']['before_loss']
    assert not restored.confirmations and restored.costs['model_calls'] == 0
    assert_unscored_return(group)
    assert_unscored_return(restored.theory_consolidation['method_feedback'][method['id']])


@pytest.mark.parametrize('failure', ['failed_candidate', 'truncated_comparison'])
def test_common_incomplete_comparison_preserves_paid_work_without_learning_a_negative(tmp_path, failure):
    agent, lab, method, question = scene(tmp_path, useful=False)
    _, models = use_and_export(agent, lab, question)
    if failure == 'failed_candidate':
        agent._save('incomplete.py', 'def predict(row): raise ValueError("fixture_failure")\n')
    else:
        agent._save('incomplete.py', 'import time\ndef predict(row):\n    time.sleep(.25)\n    return 0.\n')
        agent.config = replace(agent.config, execution_seconds=agent.costs['python_execution_seconds'] + .2)
    costs = deepcopy(agent.costs)
    result = agent._execute('run', {'operator': 'research_choice', 'models': [*models, 'incomplete.py'],
        'research_question': {'id': question['id']}}, lab)
    if failure == 'failed_candidate':
        assert any(item['returncode'] != 0 for item in result['evaluations'])
    else:
        assert result['truncated'] and result['status'] == 'comparison_incomplete_python_budget'
    assert result['evaluations']  # Completed scores survive an unsuccessful whole comparison.
    assert agent.costs['compute_cost'] >= costs['compute_cost']
    group = agent.theory_consolidation['method_feedback'][method['id']]
    outcome = _read(agent, group['outcome_refs'][0])
    assert outcome['status'] == 'pending_comparison' and not outcome.get('working_comparison')
    assert not outcome['independent_confirmation'] and outcome['fit_compute_cost'] > 0
    assert_unscored_return(group)
    assert lab.paid == 0 and not agent.confirmations and agent.costs['model_calls'] == 0
