"""The saved-source boundary executes its actual author once per research step."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from scripts import prepare_pr3_source_successor as runner
from scripts import run_anonymous_revision_inquiry as study
from test_registered_native_author import knowledge

DRIVER = 'shared/counting_author.py'
SOURCE = '''import json
from pathlib import Path
def select_actions(spec, records, memory):
    path = Path('author_invocations.json')
    count = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(count))
    state = json.loads(Path('research_state.json').read_text())
    return {'action': 'write', 'arguments': {'path': 'step.txt',
        'code': str(count) + '/' + str(len(state['method_results']))}}
'''


def agent_at(workspace, source=SOURCE):
    agent = CommonArchiveInquiryAgent(None, workspace, arm='code_react', source_task=False,
        knowledge={'files': {DRIVER: source},
                   'finite_native_author': {'question': {'id': 'task'}}},
        config=InquiryLLMConfig(execution_seconds=30., max_model_calls=0, token_budget=0))
    agent._open_question({'id': 'task', 'question': 'Retain the actual source action',
                         'parent_id': None, 'depends_on': []})
    agent._organization_state = {'finite_author': {'completed_steps': 0}}
    agent._before_model_turn = lambda _lab: None
    agent._after_model_turn = lambda _lab: None
    return agent


def test_prepared_source_executes_once_and_keeps_normal_dispatch_records_and_costs(tmp_path):
    laboratory = SimpleNamespace(public_spec=lambda: {})
    outcomes = []
    for mode in ('ordinary', 'saved_source'):
        agent = agent_at(tmp_path / mode)
        checkpoints = []
        if mode == 'ordinary':
            outcome = execute_procedure(agent, laboratory, DRIVER, max_steps=2,
                allow_confirmation=False, record_events=True, on_checkpoint=checkpoints.append)
        else:
            outcome = runner._source_research_steps(agent, laboratory,
                {'author': {'path': DRIVER, 'max_steps': 2}}, checkpoints.append)
        assert outcome['status'] == 'method_step_limit'
        if mode == 'saved_source':
            timing = outcome['execution_timing']
            assert timing['phase_seconds']['author'] == pytest.approx(agent.costs['python_execution_seconds'])
            assert timing['wall_seconds'] >= timing['phase_seconds']['author']
            assert timing['after_and_checkpoint_are_nested_in_procedure'] is True
        assert int((agent.workspace / 'author_invocations.json').read_text()) == 2
        assert (agent.workspace / 'step.txt').read_text() == '2/1'
        assert len(agent.method_results) == len(agent.events) == len(checkpoints) == 2
        assert agent._prepared_program_result is None
        assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
        program_seconds = sum(row['feedback']['program_execution']['python_execution_seconds']
                              for row in agent.method_results)
        assert agent.costs['python_execution_seconds'] == pytest.approx(program_seconds)
        outcomes.append([(row['proposal'], row['feedback']['program_action'],
                          row['evidence_before'], row['evidence_after'])
                         for row in agent.method_results])
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize('proposal', [
    {'action': 'query', 'arguments': {'actions': [{'new_purchase': True}]}},
    {'action': 'finish', 'arguments': {'model': 'old.py'}},
    {'model': 'old.py'},
    {'actions': [{'new_purchase': True}]},
])
def test_prepared_source_still_stops_before_acquisition_or_submission(tmp_path, proposal):
    source = "def select_actions(*args):\n    return " + repr(proposal) + '\n'
    agent = agent_at(tmp_path, source)
    laboratory = SimpleNamespace(public_spec=lambda: {})
    outcome = runner._source_research_steps(agent, laboratory,
        {'author': {'path': DRIVER, 'max_steps': 2}}, lambda _: pytest.fail('nothing executed'))
    assert outcome['status'] == 'same_paid_boundary'
    assert outcome['boundary']['proposed_action'] == proposal
    assert agent.method_results == agent.events == []
    assert agent.costs['observation_cost'] == agent.costs['compute_cost'] == 0
    assert agent.costs['python_execution_seconds'] > 0
    assert agent.model is None


def test_exhausted_budget_does_not_consume_prepared_source_proposal(tmp_path):
    agent = agent_at(tmp_path)
    agent.costs['python_execution_seconds'] = agent.config.execution_seconds
    proposal = {'path': DRIVER, 'proposal': {'action': 'write',
        'arguments': {'path': 'must_not_exist.txt', 'code': 'not executed'}},
        'program_execution': {'path': DRIVER, 'python_execution_seconds': .25}}
    outcome = execute_procedure(agent, SimpleNamespace(), DRIVER, max_steps=1,
        _prepared_program_result=proposal)
    assert outcome['status'] == 'python_execution_budget_exhausted'
    assert not (tmp_path / 'must_not_exist.txt').exists()
    assert agent.method_results == []
    assert agent._prepared_program_result is None
    assert agent.costs['python_execution_seconds'] == agent.config.execution_seconds


def test_registered_workspace_result_skips_discarded_portable_export_only(tmp_path, monkeypatch):
    inherited = knowledge()
    ordinary = CommonArchiveInquiryAgent(None, tmp_path / 'ordinary', arm='code_react',
        knowledge=deepcopy(inherited), source_task=False,
        config=InquiryLLMConfig(max_model_calls=0, token_budget=0))
    assert ordinary.result('method_step_limit')['outgoing_knowledge']['files']['old.py'] == inherited['files']['old.py']
    library = tmp_path / 'old_library.json'
    library.write_text(json.dumps({'candidates': {}, 'executions': {}, 'transfers': []}))
    registered = study.RegisteredOrganizationAgent(None, tmp_path / 'registered', arm='archive',
        research_organization={'phase': 'source', 'initial_library_path': str(library),
                               'theories': [], 'consolidation': {}},
        knowledge=deepcopy(inherited), source_task=False,
        config=InquiryLLMConfig(max_model_calls=0, token_budget=0))

    def discarded_export(*_args, **_kwargs):
        pytest.fail('workspace result must not first read all files only to discard them')

    monkeypatch.setattr(registered, '_result_files', discarded_export)
    monkeypatch.setattr(registered, '_shared_files', discarded_export)
    for status in ('researching', 'method_step_limit'):
        result = registered.result(status)
        assert result['knowledge_protocol'] == 'method_practice_workspace'
        assert result['outgoing_knowledge']['workspace'] == str(registered.workspace)
        assert 'files' not in result['outgoing_knowledge']
    assert (registered.workspace / 'old.py').read_text() == inherited['files']['old.py']


@pytest.mark.parametrize('prior_result', [False, True])
def test_source_author_exhausting_python_budget_does_not_execute_or_count_attempt(
        tmp_path, monkeypatch, prior_result):
    source = "from pathlib import Path\ndef select_actions(*args):\n" \
        "    Path('author_ran.txt').write_text('once')\n" \
        "    return {'action':'run','arguments':{'operator':'native_condition_research'}}\n"
    agent = agent_at(tmp_path, source)
    agent.costs.update(compute_cost=17, python_execution_seconds=5.)
    if prior_result:
        agent.method_results.append({'step': 0, 'path': DRIVER, 'proposal': {'stop': 'old'},
            'feedback': {'operator': 'research_choice', 'status': 'old retained result'},
            'evidence_before': {'records': 0, 'rows': 0},
            'evidence_after': {'records': 0, 'rows': 0}})
    retained = deepcopy(agent.method_results)
    # Phase diagnostics use the shared runtime clock. This module reads only
    # the author's start/end for its unchanged billed execution allowance.
    clock = iter([0., 31.])
    monkeypatch.setattr(runner, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(agent, '_execute', lambda *_args, **_kwargs:
                        pytest.fail('author exhausted the allowance before any action'))
    result = runner._source_research_steps(agent, SimpleNamespace(public_spec=lambda: {}),
        {'author': {'path': DRIVER, 'max_steps': 2}}, lambda _: pytest.fail('no action checkpoint'))
    assert result['status'] == 'python_execution_budget_exhausted'
    assert result['native_attempts'] == result['new_method_results'] == 0
    assert agent._organization_state['finite_author']['completed_steps'] == 0
    assert agent.method_results == retained
    assert agent.events == []
    assert agent.costs['python_execution_seconds'] == 36.
    assert agent.costs['compute_cost'] == 17
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
    assert (tmp_path / 'author_ran.txt').read_text() == 'once'
