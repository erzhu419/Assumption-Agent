"""Method returns use unchanged predictors on common newly paid rows only."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.code_agent import python_model_context
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_feedback import _read
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_common_method_feedback import scene, use_and_compare
from test_inquiry_revision import PublicLab


def prepared(path):
    agent, lab, method, question = scene(path, useful=True)
    use_and_compare(agent, lab, question)
    group = agent.theory_consolidation['method_feedback'][method['id']]
    reference, = group['outcome_refs']
    return agent, lab, method, question, reference, _read(agent, reference)


def acquire_new_rows(agent, lab, question):
    measured = PublicLab()
    measured.public_spec = lab.public_spec
    execute = measured.execute

    def acquire(action):
        record = execute(action)
        for row in record['rows']:
            row['target'] = 2 * row['features']['u'] + 1
        record['observation_index'] = len(agent.rows)
        record['row_indices'] = list(range(len(agent.rows), len(agent.rows) + len(record['rows'])))
        return record

    measured.execute = acquire
    result = agent._execute('query', {'actions': [{'kind': 'trajectory',
        'sequence': [{'u': 8.2}, {'u': 9.2}]}], 'research_question': {'id': question['id']}}, measured)
    assert result['observations'] and measured.paid == 2
    return measured


def compare_same_models(agent, lab, question, original):
    working = original['working_comparison']
    return agent._execute('run', {'operator': 'research_choice',
        'models': [working['before_model'], working['after_model']],
        'research_question': {'id': question['id']}}, lab)


def test_actual_common_future_pair_scores_only_new_rows_and_revises_one_original_trial(tmp_path):
    agent, lab, method, question, reference, original = prepared(tmp_path)
    fit_cost = original['fit_compute_cost']
    assert fit_cost > 0 and original['working_comparison']['rows'] == 16
    measured = acquire_new_rows(agent, lab, question)
    costs = deepcopy(agent.costs)
    result = compare_same_models(agent, measured, question, original)
    assert result['models_requested'] == result['models_completed'] and not result.get('truncated')
    record = _read(agent, reference)
    pair = record['paired_future']
    assert pair['same_rows'] and pair['data_role'] == 'new_paid_readings'
    assert pair['reference_rows'] == 2 and pair['response_row_range'] == [16, 17]
    expected_before = ((2 * 8.2 + 1) ** 2 + (2 * 9.2 + 1) ** 2) / 2
    assert pair['before_loss'] == pytest.approx(expected_before)
    assert pair['before_loss'] != pytest.approx(original['working_comparison']['before_loss'])
    assert pair['after_loss'] < 1e-16
    assert len(pair['source_refs']) == 2
    assert agent.costs['observation_cost'] == costs['observation_cost']
    assert agent.costs['compute_cost'] >= costs['compute_cost']
    group = agent.theory_consolidation['method_feedback'][method['id']]
    assert group['outcome_refs'] == [reference]
    returns = deepcopy(group['returns'])
    origin, = returns['origins']
    trial = returns['action_experience'][origin['context']]['replay_research_program']
    assert trial['trials'] == 1 and trial['unscored_attempts'] == 0
    assert trial['mean_compute_cost'] == fit_cost and trial['unscored_compute_cost'] == 0
    assert origin['continuation_outcome']['reference_rows'] == 2
    assert origin['continuation_outcome']['raw_gain'] == pytest.approx(pair['before_loss'] - pair['after_loss'])
    compare_same_models(agent, measured, question, original)
    repeated = agent.theory_consolidation['method_feedback'][method['id']]['returns']
    same_origin, = repeated['origins']
    assert repeated['action_experience'][same_origin['context']]['replay_research_program'] == trial
    assert same_origin['feedback_ref'] == origin['feedback_ref']
    assert same_origin['continuation_outcome'] == origin['continuation_outcome']
    assert _read(agent, reference)['fit_compute_cost'] == fit_cost
    assert not agent.confirmations and agent.costs['model_calls'] == 0


@pytest.mark.parametrize('changed', ['source', 'fitted_parameter', 'old_paid_prefix'])
def test_changed_model_or_paid_prefix_does_not_credit_the_original_method_return(tmp_path, changed):
    agent, lab, method, question, reference, original = prepared(tmp_path)
    if changed == 'old_paid_prefix':
        agent.observations[0]['rows'][0]['target'] += 1.
        agent.rows[0] = replace(agent.rows[0], target=agent.rows[0].target + 1.)
        agent._write_evidence()
    else:
        model = original['working_comparison']['after_model']
        directory, inner = python_model_context(agent.workspace, model)
        if changed == 'source':
            agent._save(model, (directory / inner).read_text() + '\n# Changed executable source.\n')
        else:
            exported = next(item for owner in agent.questions
                for item in owner.get('revision_candidate_exports', []) if item['model'] == model)
            program = Program.from_dict(agent.theory_index[exported['theory_ref']['id']]['program'])

            def change_parameter(node):
                return (replace(node, value=node.value + 1.) if node.op == 'const'
                    else replace(node, children=tuple(change_parameter(child) for child in node.children)))

            agent._save(model, export_temporal_program(change_parameter(program), feature_names=('u',)))
    measured = acquire_new_rows(agent, lab, question)
    result = compare_same_models(agent, measured, question, original)
    assert result['models_requested'] == result['models_completed']
    assert 'paired_future' not in _read(agent, reference)
    group = agent.theory_consolidation['method_feedback'][method['id']]
    origin, = group['returns']['origins']
    trial = group['returns']['action_experience'][origin['context']]['replay_research_program']
    assert origin['status'] == 'unscored' and trial['trials'] == 0
    assert trial['unscored_compute_cost'] == original['fit_compute_cost']
    assert group['outcome_refs'] == [reference]
