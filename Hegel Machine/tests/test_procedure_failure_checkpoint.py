"""Failed finite procedure work keeps its actual account and recovery cursor."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent
from test_inquiry_revision import PublicLab


def test_stop_iteration_after_charged_work_is_checkpointed_as_failure(tmp_path, monkeypatch):
    driver = 'shared/author.py'
    proposal = {'action': 'run', 'arguments': {
        'operator': 'research_choice', 'models': ['retained.py']}}
    initial_library = tmp_path / 'initial_library.json'
    initial_library.write_text(json.dumps({'candidates': {}}))
    agent = RegisteredOrganizationAgent(None, tmp_path / 'agent', arm='archive',
        research_organization={'phase': 'source', 'theories': [],
            'initial_library_path': str(initial_library)},
        knowledge={'files': {driver:
            'def select_actions(spec, records, memory):\n    return ' + repr(proposal) + '\n'}},
        source_task=False, config=InquiryLLMConfig(observation_budget=8,
            confirmation_rows=1, compute_budget=100, execution_seconds=30.,
            max_model_calls=0, token_budget=0))
    finite = {'contract': {'kind': 'finite_native', 'path': driver,
              'max_steps': 1, 'confirm': False}, 'completed_steps': 0}
    agent._organization_state['finite_author'] = finite
    lab = PublicLab()
    monkeypatch.setattr(lab, 'execute', lambda _: pytest.fail('no observations may be bought'))

    def charged_failure(arguments, spec):
        assert arguments == proposal['arguments']
        agent.costs['compute_cost'] += 72
        raise StopIteration('no declared meter')

    monkeypatch.setattr(agent, '_python_research_choice', charged_failure)
    checkpoints, after_costs = [], []
    checkpoint_path = tmp_path / 'active.json'

    def after_step(laboratory):
        assert laboratory is lab
        finite['completed_steps'] += 1
        agent._after_model_turn(laboratory)
        after_costs.append(deepcopy(agent.costs))

    def checkpoint(result):
        checkpoints.append(deepcopy(result))
        checkpoint_path.write_text(json.dumps(result))

    outcome = execute_procedure(agent, lab, driver, max_steps=1,
        allow_confirmation=False, record_events=True,
        after_step=after_step, on_checkpoint=checkpoint)

    assert outcome['status'] == 'method_step_limit' and outcome['model'] is None
    failure = {'error': 'StopIteration: no declared meter', 'executed_path': driver}
    assert outcome['steps'][0]['proposal'] == proposal
    assert outcome['steps'][0]['feedback'] == failure
    saved = json.loads(checkpoint_path.read_text())
    assert saved['method_results'][0]['feedback'] == failure
    assert saved['method_results'][0]['proposal'] == proposal
    assert saved['events'][-1]['action'] == 'query'
    assert saved['events'][-1]['feedback'] == failure
    assert saved['research_organization']['finite_author'] == {
        **finite, 'completed_steps': 1}
    assert len(checkpoints) == len(after_costs) == 1
    assert saved['costs'] == checkpoints[0]['costs'] == after_costs[0] == agent.costs
    assert saved['costs']['compute_cost'] == 72
    assert saved['costs']['python_execution_seconds'] > 0
    assert saved['costs']['model_calls'] == saved['costs']['observation_cost'] == lab.paid == 0
    assert not agent.model and not agent.confirmations
