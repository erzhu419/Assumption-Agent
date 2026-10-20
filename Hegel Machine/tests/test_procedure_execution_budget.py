"""An exhausted procedure returns once; resumption retains its spent account."""

from dataclasses import asdict, replace
import json

import pytest

from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from scripts.experimental_design_control import ExperimentalDesignMethodAgent
from scripts.run_native_method_compilation import continuation_protocol
from test_experimental_design_control import prepared
from test_paid_scene_method_repair import make_repair


def test_timed_out_practice_returns_exhaustion_without_two_futile_invocations(tmp_path, monkeypatch):
    agent, lab, _ = prepared(tmp_path, ExperimentalDesignMethodAgent)
    spent = agent.costs['python_execution_seconds']
    agent.config = replace(agent.config, execution_seconds=spent + 2.)
    calls = []

    def timed_out(action, arguments, laboratory):
        assert action == 'query' and laboratory is lab
        calls.append(arguments)
        agent._current_proposal = {'action': 'evaluate', 'arguments': {'model': 'a.py'}}
        agent.costs['python_execution_seconds'] = agent.config.execution_seconds + .01
        return {'returncode': -9, 'timeout': True, 'stderr': '', 'stdout': ''}

    monkeypatch.setattr(agent, '_execute', timed_out)
    result = execute_procedure(agent, lab, METHOD_PATH, max_steps=12, record_events=True)
    assert result['status'] == 'python_execution_budget_exhausted'
    assert len(calls) == len(result['steps']) == 1
    assert result['steps'][0]['feedback']['returncode'] == -9
    assert agent.costs['python_execution_seconds'] == spent + 2.01
    assert agent._method_no_progress is None
    # A second invocation with the unchanged account must execute nothing.
    again = execute_procedure(agent, lab, METHOD_PATH, max_steps=12, record_events=True)
    assert again['status'] == 'python_execution_budget_exhausted'
    assert again['steps'] == [] and len(calls) == 1
    assert lab.paid == 1


@pytest.mark.parametrize('additional_seconds', [0., 120.])
def test_exhausted_author_restores_paid_scene_and_only_explicit_extra_seconds(tmp_path, additional_seconds):
    author, checkpoint, _ = make_repair(tmp_path / 'old')
    author.config = replace(author.config, max_model_calls=4, token_budget=100,
                            confirmation_rows=0)
    trial = author._practice('same_scene')
    paid_before = (author.workspace / trial['directory'] / 'result.json').read_bytes()
    source_before = checkpoint.read_bytes()
    author.events.append({'step': 0, 'action': 'run', 'arguments': {}, 'feedback': {},
                          'usage': {'total_tokens': 7}})
    author.costs.update(python_execution_seconds=author.config.execution_seconds + .25,
                        model_calls=1, reported_tokens=7)
    previous = author.result('python_execution_budget_exhausted')
    saved = author.workspace / 'training.json'
    saved.write_text(json.dumps(previous))
    (author.workspace.parent / 'protocol.json').write_text(json.dumps({
        'study': 'luna_paid_source_procedure_repair',
        'author_limits': asdict(author.config), 'limits': asdict(author.practice_limits)}))

    plan = continuation_protocol(saved, additional_author_execution_seconds=additional_seconds)
    assert plan['author_limits']['execution_seconds'] == max(0., additional_seconds - .25)
    assert plan['author_total_limits']['execution_seconds'] == author.config.execution_seconds + additional_seconds
    assert plan['author_limits']['max_model_calls'] == 3
    assert plan['author_limits']['token_budget'] == 93
    assert plan['author_limits']['compute_budget'] == author.config.compute_budget - author.costs['compute_cost']
    assert plan['author_limits']['observation_budget'] == author.config.observation_budget - author.costs['observation_cost']
    assert plan['prior_author_phases'][-1]['costs'] == previous['costs']
    assert plan['prior_author_phases'][-1]['status'] == 'python_execution_budget_exhausted'
    restored, _, _ = make_repair(tmp_path / 'new')
    restored.restore_practice(previous)
    restored.config = replace(restored.config, **plan['author_limits'])
    assert restored.practice_trials == author.practice_trials
    inherited = restored.workspace / trial['directory']
    assert inherited.is_symlink() and (inherited / 'result.json').read_bytes() == paid_before
    assert restored.costs['python_execution_seconds'] == 0.
    if additional_seconds == 0.:
        class NoCallProvider:
            def next_action(self, request):
                pytest.fail('restoring an exhausted account must not request another paid action')

        restored.provider = NoCallProvider()
        result = restored.run()
        assert result['status'] == 'python_execution_budget_exhausted'
        assert result['costs']['model_calls'] == 0
    assert (author.workspace / trial['directory'] / 'result.json').read_bytes() == paid_before
    assert checkpoint.read_bytes() == source_before
