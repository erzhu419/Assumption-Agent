"""Same-scene method practice keeps paid evidence and advances its existing stream."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random

from hegel_machine.cognition.anonymous_method_learning import SingleSubmissionMethodAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.python_method_learning import PythonArchiveMethodLearner
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_native_condition_inquiry import buy_fixture, execute, prepared
from test_native_method_practice_resume import PracticeLab


def test_restored_practices_charge_only_new_work_and_do_not_reseed_confirmation(tmp_path):
    old = prepared(tmp_path / 'old')
    lab = PracticeLab()
    buy_fixture(old, lab)
    old._save('zero.py', 'def predict(row):\n    return 0.0\n')
    execute(old, lab, 'finish', {'model': 'zero.py'})
    previous = old.result('model_call_budget_exhausted')
    assert previous['fresh_confirmation']['passed'] is False
    (old.workspace / 'research_context.json').write_text(json.dumps(old._request(lab)))
    previous_path = old.workspace / 'training.json'
    previous_path.write_text(json.dumps(previous))
    original = previous_path.read_text()
    previous = json.loads(original)
    prior_rng = old.confirmation_rng.getstate()
    method = "def select_actions(spec, records, memory):\n    return {'model': 'zero.py'}\n"
    trials = []

    def prepare_trial(case, workspace, source, limits):
        origin = previous_path if not trials else trials[-1]['agent'].workspace / 'result.json'
        saved = json.loads(origin.read_text())
        total = replace(limits,
            observation_budget=limits.observation_budget + saved['costs']['observation_cost'],
            compute_budget=limits.compute_budget + saved['costs']['compute_cost'],
            execution_seconds=limits.execution_seconds + saved['costs']['python_execution_seconds'])
        agent = SingleSubmissionMethodAgent(None, workspace, arm='hegel',
            source_task=False, knowledge=deepcopy(old.knowledge), config=total,
            confirmation_seed=case['confirmation_seed'])
        current_lab = PracticeLab()
        agent.restore_practice_research(saved, origin.parent, current_lab,
            confirmation_seed=case['confirmation_seed'])
        agent.bind_method(source)
        payload = {'agent': agent, 'laboratory': current_lab, 'inherited_candidates': [],
            'prior_research_cost': deepcopy(saved['costs']), 'resume_origin': str(origin),
            'prior_method_count': len(saved['method_results']),
            'prior_event_count': len(saved['events'])}
        trials.append(payload)
        return payload

    author = PythonArchiveMethodLearner(None, tmp_path / 'author',
        knowledge={'files': {METHOD_PATH: method}},
        cases={'same_scene': {'confirmation_seed': 1709}}, lab_factory=lambda _: PracticeLab(),
        config=InquiryLLMConfig(observation_budget=500, compute_budget=100000,
            execution_seconds=120.), prepare_trial=prepare_trial,
        practice_limits=InquiryLLMConfig(observation_budget=200, confirmation_rows=16,
            compute_budget=20000, execution_seconds=30., max_model_calls=0, token_budget=0))

    expected_rng = Random()
    expected_rng.setstate(prior_rng)
    for index in range(2):
        expected_actions = PracticeLab().confirmation_actions(16, expected_rng)
        feedback = author._practice('same_scene')
        payload = trials[-1]
        assert payload['laboratory'].actions == expected_actions
        assert feedback['status'] == 'confirmation_failed'
        assert feedback['continuation_cost']['observation_cost'] == 16
        assert feedback['continuation_cost']['model_calls'] == 0
        assert feedback['costs']['observation_cost'] == previous['costs']['observation_cost'] + 16 * (index + 1)
        summary = feedback['execution_summary']
        assert summary['step_count'] == summary['confirmation_count'] == 1
        assert len(feedback['steps']) == 1
        assert summary['research_steps'][0]['json_path'] == ['method_results', index]
        saved = json.loads((payload['agent'].workspace / 'result.json').read_text())
        assert saved['paid_observations'][:len(previous['paid_observations'])] == previous['paid_observations']
        assert saved['events'][:len(previous['events'])] == previous['events']
        assert saved['practice_start']['events'] == payload['prior_event_count']
        assert saved['practice_start']['method_results'] == index

    assert author.costs['observation_cost'] == 32
    view = author._paid_practice_view(author.practice_trials[-1])
    assert view['continuation_cost']['observation_cost'] == 16
    assert view['prior_research_cost']['observation_cost'] == previous['costs']['observation_cost'] + 16
    for key in ('measurement_cost', 'compute_cost', 'python_execution_seconds'):
        assert author.costs[key] == sum(trial['continuation_cost'][key] for trial in author.practice_trials)
    assert previous_path.read_text() == original


def test_restored_summary_does_not_turn_old_confirmation_into_new_practice_support():
    from hegel_machine.cognition.temporal_paired_practice import practice_execution_summary

    result = {'status': 'method_stopped', 'practice_start': {'method_results': 1, 'events': 1},
        'method_results': [
            {'step': 0, 'proposal': {'model': 'old.py'}, 'feedback': {
                'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 16}}},
            {'step': 1, 'proposal': {'stop': 'incomplete'}, 'feedback': {'status': 'method_stopped'}}],
        'events': [{'feedback': {'error': 'old error'}}, {'feedback': {'status': 'method_stopped'}}]}
    summary = practice_execution_summary(result, 'trials/trial_001')
    assert summary['step_count'] == 1
    assert summary['confirmation_count'] == summary['submission_count'] == 0
    assert summary['research_steps'][0]['json_path'] == ['method_results', 1]
    assert 'latest_confirmation' not in summary
    assert 'last_execution_message' not in summary
