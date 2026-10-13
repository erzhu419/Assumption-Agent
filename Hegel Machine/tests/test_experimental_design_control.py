"""The frozen selector ablation changes only the actual experimental choice."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts.experimental_design_control import (
    ExperimentalDesignMethodAgent, RandomExperimentMethodAgent, VarianceExperimentMethodAgent,
)
from test_common_archive_observable_research import MeasuredLab
from test_common_archive_online_research import ConfirmationLab, READOUT
from test_inquiry_revision import action, write


def prepared(path, cls, *, target='response', seed=1):
    agent = cls(None, path, arm='hegel', source_task=False, confirmation_seed=seed,
        config=InquiryLLMConfig(observation_budget=100, confirmation_rows=3,
            compute_budget=2000, execution_seconds=30., max_model_calls=0, token_budget=0))
    lab = MeasuredLab() if target == 'meter' else ConfirmationLab()
    channels = ['meter'] if target == 'meter' else []
    agent._execute('query', {'actions': [action(0., channels=channels)]}, lab)
    slopes = (2, 3) if target == 'meter' else (1, 0)
    offset = 1 if target == 'meter' else 0
    for name, slope in zip(('a.py', 'b.py'), slopes):
        write(agent, name, f'''offset = 0.
def fit(rows):
    global offset
    assert len(rows) == 1 and rows[0]['target'] == {offset}.
    offset = rows[0]['target']
def predict(row): return {slope} * row['features']['u'] + offset
''', lab)
    write(agent, 'readout.py', READOUT, lab)
    arguments = {'operator': 'compare_experiments', 'target': target,
        'models': ['a.py', 'b.py'], 'experiments': [
            {'id': name, 'actions': [action(*values, channels=channels)]}
            for name, values in [('zero', (0.,)), ('medium', (1.,)),
                                 ('strong', (2.,)), ('different_cost', (.1, .1))]]}
    return agent, lab, arguments


@pytest.mark.parametrize('target', ['response', 'meter'])
def test_random_control_keeps_candidates_and_cost_then_reads_actual_selected_measurements(tmp_path, target):
    outputs = {}
    for name, cls in [('greedy', ExperimentalDesignMethodAgent),
                      ('random', RandomExperimentMethodAgent)]:
        agent, lab, arguments = prepared(tmp_path / name, cls, target=target)
        rng_state = agent.confirmation_rng.getstate()
        comparison = agent._execute('run', arguments, lab)['comparison']
        outputs[name] = (agent, lab, comparison)
        assert lab.paid == agent.costs['observation_cost'] == 1
        assert agent.confirmation_rng.getstate() == rng_state
    greedy_agent, _, greedy = outputs['greedy']
    agent, lab, comparison = outputs['random']
    assert greedy['selected_experiment']['id'] == 'strong'
    assert comparison['selected_experiment']['id'] == 'medium'
    assert comparison['candidate_contexts'] == greedy['candidate_contexts']
    assert comparison['eligible_models'] == greedy['eligible_models']
    assert [(row['model'], row['mse']) for row in comparison['prior_fit']] == [
        (row['model'], row['mse']) for row in greedy['prior_fit']]
    assert [row['predictions'] for row in comparison['forecasts']] == [
        row['predictions'] for row in greedy['forecasts']]
    assert agent.costs['compute_cost'] == greedy_agent.costs['compute_cost']
    control = comparison['experiment_selection_control']
    assert control['greedy_suggestion'] == greedy['selected_experiment']
    assert control['eligible_experiments'] == ['zero', 'medium', 'strong']
    assert comparison['query_arguments']['revision']['models'] == greedy['query_arguments']['revision']['models']
    assert comparison['query_arguments']['revision']['target'] == target
    state = json.loads((agent.workspace / 'research_state.json').read_text())
    assert state['experiment_comparisons'][-1] == json.loads(json.dumps(comparison))

    query = deepcopy(comparison['query_arguments'])
    query['revision']['readout'] = 'readout.py'
    record = agent._execute('query', query, lab)['revision']
    assert lab.paid == agent.costs['observation_cost'] == 2
    assert record['actions'] == [action(1., channels=['meter'] if target == 'meter' else [])]
    assert [forecast['prospective_mse'] for forecast in record['forecasts']] == [0., 1.]
    assert record['authored_answer']['answer']['surviving'] == [record['forecasts'][0]['model']]
    assert record['authored_answer']['answer']['rejected'] == [record['forecasts'][1]['model']]
    from scripts.run_anonymous_online_research import experimental_design_trace
    reported = experimental_design_trace(agent.result('researching'))['comparisons'][0]['choice_control']
    assert reported['greedy_suggestion']['id'] == 'strong'
    assert reported['selected_experiment'] == 'medium'
    assert reported['eligible_experiments'] == ['zero', 'medium', 'strong']


@pytest.mark.parametrize('second_model,expected', [
    ("def predict(row): return 999.\n", 'single_supported_candidate'),
    ("def predict(row): return row['features']['u']\n", 'observational_ambiguity'),
])
@pytest.mark.parametrize('cls', [RandomExperimentMethodAgent, VarianceExperimentMethodAgent])
def test_no_comparable_choice_and_unrelated_query_keep_existing_behavior(tmp_path, second_model, expected, cls):
    agent, lab, arguments = prepared(tmp_path, cls)
    write(agent, 'b.py', second_model, lab)
    comparison = agent._execute('run', arguments, lab)['comparison']
    assert comparison['diagnosis'] == expected
    assert comparison['selected_experiment'] is comparison['query_arguments'] is None
    assert 'experiment_selection_control' not in comparison
    assert lab.paid == 1
    feedback = agent._execute('query', {'actions': [action(.75)]}, lab)
    assert feedback['observations'][0]['rows'][0]['target'] == .75
    assert lab.paid == 2


def test_variance_control_replaces_task_risk_choice_without_refitting_or_extra_cost(tmp_path):
    outputs = {}
    for name, cls in [('task_risk', ExperimentalDesignMethodAgent),
                      ('variance', VarianceExperimentMethodAgent)]:
        agent, lab, _ = prepared(tmp_path / name, cls)
        models = []
        for relevant in (0, 1):
            for nuisance in (0, 1):
                path = f'candidate_{relevant}_{nuisance}.py'
                write(agent, path, f'''def fit(rows):
    assert len(rows) == 1 and rows[0]['target'] == 0
def predict(row):
    u = row['features']['u']
    if u == 1: return {100 * nuisance}.
    if u == 4: return {200 * nuisance}.
    if u in (2, 3): return {relevant}.
    return 0.
''', lab)
                models.append(path)
        arguments = {'operator': 'compare_experiments', 'models': models,
            'experiments': [
                {'id': 'nuisance', 'actions': [action(1)]},
                {'id': 'task_relevant', 'actions': [action(2)]},
                {'id': 'expensive_nuisance', 'actions': [action(4, 4)]}],
            'reference_actions': [action(3)]}
        before_rows = deepcopy(agent.rows)
        rng_state = agent.confirmation_rng.getstate()
        comparison = agent._execute('run', arguments, lab)['comparison']
        assert agent.rows == before_rows and lab.paid == 1
        assert agent.confirmation_rng.getstate() == rng_state
        outputs[name] = agent, lab, comparison

    original_agent, _, original = outputs['task_risk']
    agent, lab, changed = outputs['variance']
    assert original['selected_experiment']['id'] == 'task_relevant'
    assert changed['selected_experiment']['id'] == 'nuisance'
    assert changed['experiment_selection_control'] == {
        'policy': 'maximum_prediction_variance_same_observation_cost',
        'eligible_experiments': ['nuisance', 'task_relevant'],
        'greedy_suggestion': original['selected_experiment'],
        'selected_experiment': 'nuisance'}
    for key in ('candidate_contexts', 'eligible_models', 'prior_fit', 'forecasts',
                'comparisons', 'task_value', 'evidence_cutoff', 'compute_cost'):
        assert changed[key] == original[key]
    assert agent.costs['compute_cost'] == original_agent.costs['compute_cost']
    assert changed['query_arguments']['revision']['models'] == original['query_arguments']['revision']['models']
    state = json.loads((agent.workspace / 'research_state.json').read_text())
    assert state['experiment_comparisons'][-1] == json.loads(json.dumps(changed))

    revision = agent._execute('query', changed['query_arguments'], lab)['revision']
    assert revision['actions'] == [action(1)]
    assert lab.paid == 2
    assert [f['predictions'] for f in revision['forecasts']] == [[0.], [100.], [0.], [100.]]
    assert [f['prospective_mse'] for f in revision['forecasts']] == [1., 9801., 1., 9801.]


def test_sequential_design_can_pass_twelve_invocations_without_reopening_confirmation(tmp_path):
    agent, lab, _ = prepared(tmp_path, ExperimentalDesignMethodAgent)
    write(agent, 'final.py', "def predict(row): return row['features']['u']\n", lab)
    agent.bind_method('''import json
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    if len(steps) < 13:
        return {'action':'run','arguments':{'code':'pass'}}
    return {'model':'final.py'}
''')
    result = agent.execute_method(lab, max_steps=12)
    assert result['status'] == 'completed'
    assert len(result['method_results']) == 14
    assert result['fresh_confirmation']['passed']
    assert len(agent.confirmations) == 1
    assert result['costs']['observation_cost'] == lab.paid == 4
