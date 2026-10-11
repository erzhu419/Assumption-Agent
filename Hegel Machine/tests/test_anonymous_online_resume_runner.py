"""Exercise actual runner continuation; scripted physics checks are not discovery."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from scripts.run_anonymous_online_research import (
    continuation_protocol, representation_laboratory, representation_protocol, run_online,
)
from test_anonymous_method_learning import memoryless_source


def test_continuation_selects_every_token_truncation_without_reopening_other_outcomes(tmp_path):
    plan = representation_protocol(tmp_path / 'archive')
    statuses = ['completed', 'token_budget_exhausted', 'token_budget_exhausted', 'provider_failed',
                'model_call_budget_exhausted', 'token_budget_exhausted', 'completed', 'completed']
    trials = [{'case': case['id'], 'arm': arm, 'status': status}
              for (case, arm), status in zip(
                  [(case, arm) for case in plan['cases'] for arm in plan['arms']], statuses)]
    (tmp_path / 'protocol.json').write_text(json.dumps(plan))
    (tmp_path / 'summary.json').write_text(json.dumps({'status': 'completed', 'trials': trials}))
    resumed = continuation_protocol(tmp_path)
    assert {(case['id'], arm) for case in resumed['cases'] for arm in case['arms']} == {
        ('case_000', 'hegel'), ('case_001', 'code_react'), ('case_002', 'hegel')}
    assert resumed['limits'] == {**plan['limits'], 'token_budget': 4_000_000}
    assert resumed['original_limits'] == plan['limits']
    assert resumed['source_archive'] == plan['source_archive']
    assert resumed['shared_research_task'] == plan['shared_research_task']
    assert json.loads((tmp_path / 'protocol.json').read_text()) == plan


def finite_fixture_source(laboratory):
    """Exact old physics only for verifying resume/evaluation ordering."""
    physical = laboratory._laboratory
    a, g, w = physical._decay, physical._current_gain, physical._state_gain
    offset, linear, square, history = (physical._response_offset, physical._response_control_gain,
        physical._response_square_gain, physical._response_history_gain)
    return f'''def predict(row):
    controls = [point['u'] for point in row['history']] + [row['features']['u']]
    previous = 0.
    for i, u in enumerate(controls):
        reading = ({g!r} + {w!r}) * u
        reading += sum({w!r} * {a!r} ** lag * controls[i-lag] for lag in (1, 2) if i >= lag)
        result = {offset!r} + {linear!r} * u + {square!r} * reading ** 2 + {history!r} * previous
        previous = reading
    return result
'''


def test_runner_continues_total_account_and_finishes_all_training_before_terminal(
        tmp_path, memoryless_source, monkeypatch):
    original = tmp_path / 'original'
    plan = representation_protocol(memoryless_source)
    plan['cases'] = plan['cases'][:1]  # Both arms on the original finite-history physical fixture.
    plan['independent_parameter_pairs'] = 1
    plan['limits'].update(max_model_calls=3, token_budget=2)
    source = finite_fixture_source(representation_laboratory(plan['cases'][0]))
    requests = []

    class SavingProvider:
        def __init__(self, workspace, action):
            self.workspace, self.action, self.calls = Path(workspace), action, 0

        def next_action(self, request):
            self.calls += 1
            requests.append(deepcopy(request))
            # Actual CodexCodeProvider persists this pre-action view.
            (self.workspace / 'research_context.json').write_text(json.dumps(request))
            action = self.action
            if self.action == 'evaluate' and self.calls == 1:
                action = 'write'
                arguments = {'path': 'shared/knowledge.json', 'code': json.dumps({
                    'hypotheses': [{'path': 'model.py'}],
                    'research_programs': [{'path': 'model.py'}], 'support_questions': []})}
            elif self.action == 'evaluate':
                arguments = {'model': 'model.py', 'code': source}
            else:
                arguments = {'model': 'model.py'}
            return {'action': action, 'arguments': arguments,
                    'usage': {'backend': 'scripted_resume_check', 'total_tokens': 1}}

    first = run_online(plan, original,
        provider_factory=lambda path: SavingProvider(path, 'evaluate'))
    assert all(trial['status'] == 'token_budget_exhausted' for trial in first['trials'])
    previous_bytes = {arm: (original / 'case_000' / arm / 'training.json').read_bytes()
                      for arm in plan['arms']}
    for arm in plan['arms']:
        (original / 'case_000' / arm / 'evaluation.json').write_text('{"terminal_label": 987654321}')
    continued = tmp_path / 'continued'
    resumed_plan = continuation_protocol(original)
    terminal_calls = []
    original_evaluation = TemporalInquiryLab.evaluation_rows

    def terminal_after_all_stopped(lab, **kwargs):
        assert all((continued / 'case_000' / arm / 'training.json').exists() for arm in plan['arms'])
        terminal_calls.append(kwargs)
        return original_evaluation(lab, **kwargs)

    monkeypatch.setattr(TemporalInquiryLab, 'evaluation_rows', terminal_after_all_stopped)
    requests.clear()
    result = run_online(resumed_plan, continued,
        provider_factory=lambda path: SavingProvider(path, 'finish'))
    assert len(requests) == len(terminal_calls) == 2
    for trial, request in zip(result['trials'], requests):
        assert trial['status'] == 'completed' and trial['terminal']['passed']
        assert request['costs']['model_calls'] == trial['prior_research_cost']['model_calls'] == 2
        assert trial['costs']['model_calls'] == 3 and trial['continuation_cost']['model_calls'] == 1
        assert trial['prior_research_cost']['observation_cost'] == 72
        assert trial['continuation_cost']['observation_cost'] == 96
        assert trial['costs']['observation_cost'] == 168
        assert trial['costs']['reported_tokens'] == 3
        assert request['limits'] == resumed_plan['limits']
        assert 'historical_prefix_cost' not in trial
        assert '987654321' not in json.dumps(request)
        assert (original / 'case_000' / trial['arm'] / 'training.json').read_bytes() == previous_bytes[trial['arm']]
    assert all(item['seed'] == plan['cases'][0]['evaluation_seed'] for item in terminal_calls)
    assert (continued / 'source_at_run/scripts/run_anonymous_online_research.py').exists()
