"""Returned local science remains usable after the next sibling investigation."""

from copy import deepcopy

from hegel_machine.cognition.code_provider_context import prepare_code_context
from test_common_archive_llm_inquiry import ContextSavingProvider, RestorableConfirmationLab
from test_scientific_revision_inquiry import prepared, prefix


def returned_scene(path):
    steps = [
        ('finish', {'model': 'parent.py'}),
        ('run', {'code': "print('Executed local relation; its scope remains unresolved')"}),
        ('run', {'code': "raise ValueError('Unfinished alternative calculation')"}),
        ('finish', {'lesson': 'Return both actual calculations without claiming confirmation.',
                    'artifacts': ['parent.py']}),
    ]
    steps += [('write', {'path': 'scratch_note.txt', 'code': str(index),
                        'research_question': {'id': 'next_investigation',
                            'question': 'Investigate another explanation',
                            'parent_id': 'task', 'depends_on': []},
                        'notes': 'Current investigation; no historical citations.'})
              for index in range(10)]
    provider = ContextSavingProvider(steps)
    agent = prepared(path, provider, calls=len(steps))
    provider.workspace = agent.workspace
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    agent.run(lab)
    return agent, lab


def test_returned_science_and_errors_survive_sibling_and_provider_windows(tmp_path):
    agent, lab = returned_scene(tmp_path)
    before = deepcopy((agent.questions, agent.events, agent.costs))
    request = agent._request(lab)
    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    facts = context['evidence_notebook']['facts']
    returned = facts['research_return:scientific_revision_0']

    assert context['event_history']['preview_start'] > 3
    assert returned['status'] == 'inquiry_finished'
    assert returned['artifacts'] == ['parent.py']
    assert returned['analysis_refs'] == ['analysis:2', 'analysis:3']
    assert facts['analysis:2']['stdout'] == 'Executed local relation; its scope remains unresolved\n'
    assert facts['analysis:3']['returncode'] == 1
    assert 'Unfinished alternative calculation' in facts['analysis:3']['stderr']
    assert 'supported' not in returned and 'answer' not in returned
    assert facts['analysis:2']['source']['json_path'] == ['events', 2, 'feedback']
    assert context['research_notes'] == 'Current investigation; no historical citations.'
    assert (agent.questions, agent.events, agent.costs) == before
    assert lab.paid == 3 and len(agent.confirmations) == 1


def test_returned_facts_restore_without_replaying_or_promoting_local_results(tmp_path):
    agent, lab = returned_scene(tmp_path / 'source')
    before = agent._request(lab)['evidence_notebook']
    previous = agent.result('model_call_budget_exhausted')
    restored = prepared(tmp_path / 'restored', None, calls=agent.config.max_model_calls + 1)
    fresh = RestorableConfirmationLab()

    restoration = restored.restore_research(previous, agent.workspace, fresh, confirmation_seed=1709)

    assert restoration['new_observations'] == 0
    assert fresh.actions == [] and fresh.paid == 3
    assert restored._request(fresh)['evidence_notebook'] == before
    assert restored.events == agent.events and restored.costs == agent.costs
    assert restored.model is None and restored.confirmations[-1]['passed'] is False
