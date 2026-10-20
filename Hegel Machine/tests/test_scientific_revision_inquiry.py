"""Small executed counterexamples test research control, never model ability."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from random import Random

import pytest

from hegel_machine.cognition.common_archive_inquiry import (
    CommonArchiveInquiryAgent, FlatScientificRevisionAgent)
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_common_archive_llm_inquiry import ContextSavingProvider, RestorableConfirmationLab
from test_common_archive_online_research import ScriptedProvider
from test_inquiry_revision import action


PARENT = {'id': 'task', 'question': 'Predict the observed response',
          'parent_id': None, 'depends_on': []}
MODEL = """import json
from pathlib import Path
def predict(row):
    slope = json.loads(Path(__file__).with_name('coefficient.json').read_text())['slope']
    return slope * row['features']['u']
"""


def prepared(path, provider, *, agent_type=CommonArchiveInquiryAgent, calls=8):
    return agent_type(provider, path, arm='hegel', source_task=False,
        confirmation_seed=1709,
        config=InquiryLLMConfig(observation_budget=16, confirmation_rows=2,
            compute_budget=1000, execution_seconds=30., max_model_calls=calls),
        knowledge={'files': {'parent.py': MODEL, 'coefficient.json': '{"slope": 0}',
            'shared/probe.py': 'def select_actions(spec,observations,memory): return []\n',
            'shared/knowledge.json': json.dumps({'hypotheses': [], 'research_programs': [],
                                                 'support_questions': []})}})


def prefix(agent, lab, value=0.):
    feedback = agent._execute('query', {'actions': [action(value)], 'question': PARENT}, lab)
    agent.events.append({'step': -1, 'action': 'query', 'arguments': {},
                         'feedback': feedback, 'usage': {'total_tokens': 0}})


@pytest.mark.parametrize('agent_type', [CommonArchiveInquiryAgent, FlatScientificRevisionAgent])
def test_failed_confirmation_returns_repaired_dependency_to_parent(tmp_path, agent_type):
    provider = ScriptedProvider([
        ('finish', {'model': 'parent.py', 'notes': 'Parent awaits independent evidence.'}),
        ('write', {'path': 'coefficient.json', 'code': '{"slope": 1}',
                   'notes': 'Revision changes the executable coefficient from paid counterexamples.'}),
        ('evaluate', {'model': 'parent.py'}),
        ('finish', {'lesson': 'Paid fit repaired; parent must independently confirm.',
                    'artifacts': ['coefficient.json']}),
        ('evaluate', {'model': 'parent.py'}),
        ('finish', {'model': 'parent.py'}),
    ])
    agent = prepared(tmp_path, provider, agent_type=agent_type, calls=6)
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    result = agent.run(lab)

    revision = provider.requests[1]
    assignment = revision['scientific_revision_assignment']
    assert assignment['id'] == 'scientific_revision_0'
    assert assignment['failed_confirmation']['confirmation_index'] == 0
    assert assignment['failed_confirmation']['mse'] > 0
    assert revision['costs']['observation_cost'] == 3
    assert bool(revision.get('llm_inquiry')) == (agent_type is CommonArchiveInquiryAgent)
    # Returning a scientific revision has not consumed an independent observation.
    parent = provider.requests[4]
    assert not parent.get('llm_inquiry')
    assert parent['costs']['observation_cost'] == 3
    assert parent['events'][1]['feedback']['fresh_confirmation']['passed'] is False
    assert result['status'] == 'completed' and result['fresh_confirmation']['passed']
    assert result['model'] == 'parent.py'
    assert (tmp_path / 'parent.py').read_text() == MODEL  # Actual dependency change, same parent code.
    assert json.loads((tmp_path / 'coefficient.json').read_text()) == {'slope': 1}
    assert lab.paid == result['costs']['observation_cost'] == 5
    assert len(lab.actions) == 3  # Prefix, failed independent check, successful independent check.
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 6
    assert len([q for q in result['online_questions']
                if q.get('origin') == 'fresh_confirmation_failure']) == 1


@pytest.mark.parametrize('agent_type', [CommonArchiveInquiryAgent, FlatScientificRevisionAgent])
def test_same_failed_check_consumed_once_and_resume_preserves_paid_evidence(tmp_path, agent_type):
    provider = ContextSavingProvider([
        ('finish', {'model': 'parent.py', 'notes': 'Original parent notebook.'}),
        ('finish', {'lesson': 'Unresolved scientific revision.'}),
        ('run', {'code': "print('Parent retains the unresolved counterexample')"}),
    ])
    old = prepared(tmp_path / 'old', provider, agent_type=agent_type, calls=3)
    provider.workspace = old.workspace
    original_lab = RestorableConfirmationLab()
    prefix(old, original_lab)
    previous = old.run(original_lab)
    assert previous['status'] == 'model_call_budget_exhausted'
    assert previous['model'] is None
    assert previous['fresh_confirmation']['passed'] is False
    assert previous['costs']['observation_cost'] == 3
    questions = [q for q in previous['online_questions']
                 if q.get('origin') == 'fresh_confirmation_failure']
    assert len(questions) == 1
    assert not provider.requests[2].get('llm_inquiry')
    later = ScriptedProvider([('run', {'code': "print('Resume parent, no duplicate revision')"})])
    restored = prepared(tmp_path / 'resumed', later, agent_type=agent_type, calls=4)
    lab = RestorableConfirmationLab()
    report = restored.restore_research(previous, old.workspace, lab, confirmation_seed=1709)
    before_events = deepcopy(restored.events)
    result = restored.run(lab)

    assert report['new_observations'] == 0
    assert lab.actions == [] and lab.paid == 3
    assert result['paid_observations'] == previous['paid_observations']
    assert result['events'][:-1] == before_events
    assert not later.requests[0].get('llm_inquiry')
    assert len([q for q in result['online_questions']
                if q.get('origin') == 'fresh_confirmation_failure']) == 1
    assert result['model'] is None and result['fresh_confirmation']['passed'] is False


def test_restore_before_revision_uses_actual_old_confirmation_anchor(tmp_path):
    original = ContextSavingProvider([('finish', {'model': 'parent.py'})])
    old = prepared(tmp_path / 'old', original, calls=5)
    original.workspace = old.workspace
    lab = RestorableConfirmationLab()
    prefix(old, lab)
    checkpoints = []

    def interrupt(data):
        if data.get('fresh_confirmation') is not None:
            checkpoints.append(deepcopy(data))
            raise InterruptedError('Preserve real failed confirmation before next model decision')

    with pytest.raises(InterruptedError):
        old.run(lab, on_checkpoint=interrupt)
    previous = checkpoints[-1]
    old_event_count = len(previous['events'])
    provider = ScriptedProvider([
        ('run', {'code': "print('Analyze restored paid counterexamples')"}),
        ('finish', {'lesson': 'Return unresolved; further parent work remains.'}),
        ('run', {'code': "print('Parent resumes')"}),
    ])
    restored = prepared(tmp_path / 'restored', provider, calls=4)
    fresh = RestorableConfirmationLab()
    restored.restore_research(previous, old.workspace, fresh, confirmation_seed=1709)
    result = restored.run(fresh)

    assert provider.requests[0]['llm_inquiry']['assignment']['origin'] == 'fresh_confirmation_failure'
    assert fresh.actions == [] and fresh.paid == 3
    assert result['paid_observations'] == previous['paid_observations']
    returned = result['events'][1]['feedback']['inquiry_result']
    assert returned['event_indices'] == [old_event_count, old_event_count + 1]
    assert returned['model_calls'] == 2
    assert result['events'][1]['feedback']['fresh_confirmation'] == previous['fresh_confirmation']
    assert result['model'] is None


@pytest.mark.parametrize('failure', ['paid_fit', 'execution'])
def test_no_scientific_revision_for_paid_fit_or_prediction_execution_error(tmp_path, failure):
    steps = [
        ('finish', {'model': 'parent.py'}),
        ('run', {'code': "print('Ordinary parent continuation')"}),
    ]
    if failure == 'paid_fit':
        steps[0] = ('evaluate', {'model': 'parent.py'})
    provider = ScriptedProvider(steps)
    agent = prepared(tmp_path, provider, calls=len(steps))
    lab = RestorableConfirmationLab()
    prefix(agent, lab, 1. if failure == 'paid_fit' else 0.)
    if failure == 'execution':
        agent._save('parent.py', "def predict(row):\n"
            "    if row['features']['u'] != 0: raise ValueError('prediction implementation failed')\n"
            "    return 0.\n")
    result = agent.run(lab)

    assert all(not request.get('scientific_revision_assignment') for request in provider.requests)
    assert not result['llm_inquiries'] and not result['inquiry_stack']
    assert not any(q.get('origin') == 'fresh_confirmation_failure' for q in result['online_questions'])
    assert result['model'] is None
    if failure == 'paid_fit':
        assert result['fresh_confirmation'] is None and lab.paid == 1
    else:
        assert result['fresh_confirmation']['returncode'] != 0 and lab.paid == 3


def test_flat_and_child_receive_identical_scientific_assignment(tmp_path):
    assignments = []
    for cls in (CommonArchiveInquiryAgent, FlatScientificRevisionAgent):
        provider = ScriptedProvider([
            ('finish', {'model': 'parent.py'}),
            ('finish', {'lesson': 'Return pending scientific work'}),
            ('run', {'code': "print('Parent still responsible for confirmation')"}),
        ])
        agent = prepared(tmp_path / cls.__name__, provider, agent_type=cls, calls=3)
        lab = RestorableConfirmationLab()
        prefix(agent, lab)
        result = agent.run(lab)
        assignments.append(provider.requests[1]['scientific_revision_assignment'])
        if cls is FlatScientificRevisionAgent:
            assert not result['llm_inquiries'] and not result['inquiry_stack']
    assert assignments[0] == assignments[1]


def test_registered_pair_uses_same_hegel_contract_and_scores_after_both_stop(tmp_path):
    from hegel_machine.cognition.models import Observation
    from scripts.run_anonymous_revision_inquiry import run_study
    from scripts.run_scientific_revision_comparison import comparison_agent

    output = tmp_path / 'paired'
    knowledge = prepared(tmp_path / 'knowledge', None).knowledge
    plan = {'study': 'scripted_scientific_revision_pair', 'arms': ['recursive', 'flat'],
        'cases': [{'id': 'case', 'kind': 'fixture', 'prefix_seed': 271,
                   'confirmation_seed': 1709, 'evaluation_seed': 371}],
        'limits': asdict(InquiryLLMConfig(observation_budget=16, confirmation_rows=2,
            compute_budget=1000, execution_seconds=30., max_model_calls=6)),
        'evaluation': {'rows': 4, 'mse_threshold': 1e-10},
        'scope': 'Scripted executed wiring; no scientific discovery or real model calls.'}
    providers = {}

    def provider_factory(path):
        provider = ScriptedProvider([
            ('finish', {'model': 'parent.py'}),
            ('write', {'path': 'coefficient.json', 'code': '{"slope": 1}'}),
            ('evaluate', {'model': 'parent.py'}),
            ('finish', {'lesson': 'Local revision complete; new confirmation still required.'}),
            ('evaluate', {'model': 'parent.py'}),
            ('finish', {'model': 'parent.py'}),
        ])
        providers[path.name] = provider
        return provider

    class ScoringLab(RestorableConfirmationLab):
        def evaluation_rows(self, length, seed):
            assert all((output / 'case' / arm / 'training.json').exists()
                       for arm in ('recursive', 'flat'))
            return [Observation({'u': value}, value, ()) for value in (-0.5, 0.25, 1., -1.)]

    def prepare(agent, lab, case):
        prefix(agent, lab)
        return {}

    result = run_study(plan, output, provider_factory=provider_factory, knowledge=knowledge,
        prepare_prefix=prepare, laboratory_factory=lambda case: ScoringLab(),
        workers=2, agent_factory=comparison_agent)

    assert {row['arm'] for row in result['trials']} == {'recursive', 'flat'}
    assert all(row['fresh_confirmation_passed'] and row['terminal']['passed']
               for row in result['trials'])
    assert all(row['costs']['model_calls'] == 6 and row['costs']['observation_cost'] == 5
               for row in result['trials'])
    trainings = [json.loads((output / 'case' / arm / 'training.json').read_text())
                 for arm in ('recursive', 'flat')]
    assert all(training['agent_arm'] == 'hegel' for training in trainings)
    assert trainings[0]['paid_observations'] == trainings[1]['paid_observations']
    assert providers['recursive'].requests[1]['scientific_revision_assignment'] == (
        providers['flat'].requests[1]['scientific_revision_assignment'])
    assert all(provider.requests[4]['costs']['observation_cost'] == 3
               for provider in providers.values())


@pytest.mark.parametrize('agent_type', [CommonArchiveInquiryAgent, FlatScientificRevisionAgent])
def test_active_revision_restores_changed_dependency_and_scope(tmp_path, agent_type):
    provider = ContextSavingProvider([
        ('finish', {'model': 'parent.py', 'notes': 'Parent notebook.'}),
        ('write', {'path': 'coefficient.json', 'code': '{"slope": 1}',
                   'notes': 'Unfinished local revision.'}),
    ])
    old = prepared(tmp_path / 'old', provider, agent_type=agent_type, calls=6)
    provider.workspace = old.workspace
    lab = RestorableConfirmationLab()
    prefix(old, lab)
    checkpoints = []

    def interrupt(result):
        if result['events'][-1]['action'] == 'write':
            checkpoints.append(deepcopy(result))
            raise InterruptedError('Pause after actual revision artifact, before return')

    with pytest.raises(InterruptedError):
        old.run(lab, on_checkpoint=interrupt)
    previous = checkpoints[-1]
    continuation = ScriptedProvider([
        ('evaluate', {'model': 'parent.py'}),
        ('finish', {'lesson': 'Return repaired dependency for parent confirmation'}),
        ('evaluate', {'model': 'parent.py'}),
        ('finish', {'model': 'parent.py'}),
    ])
    resumed = prepared(tmp_path / 'resumed', continuation, agent_type=agent_type, calls=6)
    fresh = RestorableConfirmationLab()
    resumed.restore_research(previous, old.workspace, fresh, confirmation_seed=1709)
    result = resumed.run(fresh)

    first = continuation.requests[0]
    assert first['scientific_revision_assignment']['id'] == 'scientific_revision_0'
    assert bool(first.get('llm_inquiry')) == (agent_type is CommonArchiveInquiryAgent)
    assert first['costs']['model_calls'] == 2 and first['costs']['observation_cost'] == 3
    assert first['research_notes'] == 'Unfinished local revision.'
    assert not continuation.requests[2].get('llm_inquiry')
    assert result['status'] == 'completed' and result['fresh_confirmation']['passed']
    assert result['costs']['model_calls'] == 6
    assert len(fresh.actions) == 1 and fresh.paid == 5
    assert result['events'][1]['feedback']['fresh_confirmation'] == previous['fresh_confirmation']


@pytest.mark.parametrize('agent_type', [CommonArchiveInquiryAgent, FlatScientificRevisionAgent])
def test_checked_child_answer_returns_before_call_limit_in_both_conditions(tmp_path, agent_type):
    # Test the proposed y=u law against one actual paid confirmation counterexample.
    control = Random(1709).uniform(-2., 2.)
    readout = f"""def readout(result,spec,memory):
    return {{'status': 'supported', 'answer': {{'predictor': 'parent.py'}},
        'evidence': [1], 'checks': [{{'id': 'response_matches_control',
            'terms': [{{'row': 1, 'coefficient': 1}}], 'relation': 'eq',
            'expected': {control!r}, 'atol': 1e-10,
            'scope': {{'question': 'Does the paid response match this nonzero control?'}}}}]}}
"""
    code = ("from pathlib import Path\n"
            "Path('coefficient.json').write_text('{\"slope\": 1}')\n"
            f"Path('checked_readout.py').write_text({readout!r})\n")
    provider = ScriptedProvider([
        ('finish', {'model': 'parent.py'}),
        ('run', {'code': code}),
        ('run', {'operator': 'analyze_evidence', 'path': 'checked_readout.py'}),
        ('evaluate', {'model': 'parent.py'}),
        ('finish', {'model': 'parent.py'}),
    ])
    agent = prepared(tmp_path, provider, agent_type=agent_type, calls=7)
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    result = agent.run(lab)

    parent = provider.requests[3]
    assert not parent.get('llm_inquiry')
    assert parent['scientific_revision_review']['phase'] == 'returned_to_parent'
    returned = result['events'][1]['feedback']['inquiry_result']
    assert returned['status'] == 'answered' and returned['model_calls'] == 2
    assert result['status'] == 'completed' and result['fresh_confirmation']['passed']
    assert result['costs']['model_calls'] == 5 and lab.paid == 5


@pytest.mark.parametrize('agent_type', [CommonArchiveInquiryAgent, FlatScientificRevisionAgent])
def test_first_paid_revision_run_recovers_preprovider_checkpoint(tmp_path, monkeypatch, agent_type):
    from hegel_machine.cognition.interrupted_online_inquiry import recover_interrupted_online_run

    code = "print('First analysis of the actual confirmation counterexamples')"

    class PaidReplyProvider(ContextSavingProvider):
        def next_action(self, request):
            decision = super().next_action(request)
            if decision['action'] == 'run':
                saved = json.loads((self.workspace / 'active.json').read_text())
                assert saved['costs']['model_calls'] == 1
                review = next(q for q in saved['online_questions']
                              if q.get('origin') == 'fresh_confirmation_failure')
                assert review['id'] == 'scientific_revision_0'
                assert bool(saved['inquiry_stack']) == (agent_type is CommonArchiveInquiryAgent)
                if agent_type is FlatScientificRevisionAgent:
                    assert review['flat_review']['status'] == 'running'
                # Match the real paid CLI reply: the model did not supply a question id.
                (self.workspace / 'code_action_002.json').write_text(json.dumps({
                    'action': 'run', 'arguments_json': '{}', 'source_code': code,
                    'research_notes': 'First scientific revision calculation.',
                    'research_question': None}))
                (self.workspace / 'code_call_002.jsonl').write_text(json.dumps({
                    'type': 'turn.completed', 'usage': {'input_tokens': 1,
                        'output_tokens': 0, 'cached_input_tokens': 0}}) + '\n')
            return decision

    provider = PaidReplyProvider([
        ('finish', {'model': 'parent.py', 'notes': 'Parent notebook.'}),
        ('run', {'code': code, 'notes': 'First scientific revision calculation.'}),
    ])
    old = prepared(tmp_path / 'old', provider, agent_type=agent_type, calls=5)
    provider.workspace = old.workspace
    lab = RestorableConfirmationLab()
    prefix(old, lab)
    checkpoints = []

    def save(result):
        checkpoints.append(deepcopy(result))
        (old.workspace / 'active.json').write_text(json.dumps(result))

    original_python = old._python

    def interrupted(command, *args, **kwargs):
        if len(command) > 1 and Path(command[1]).name.startswith('scratch_'):
            raise RuntimeError('Worker stopped during the first paid review calculation')
        return original_python(command, *args, **kwargs)

    monkeypatch.setattr(old, '_python', interrupted)
    with pytest.raises(RuntimeError, match='first paid review'):
        old.run(lab, on_checkpoint=save)
    assert not any(q.get('origin') == 'fresh_confirmation_failure'
                   for q in checkpoints[0]['online_questions'])
    assert len(checkpoints) == 2  # Ordinary failed confirmation, then pre-provider review state.
    previous = recover_interrupted_online_run(old.workspace, model='scripted',
        interruption={'executed_path': 'scratch_002.py', 'interrupted_python_seconds': 0.1},
        result_path=old.workspace / 'interrupted_training.json')
    assert previous['events'][-1]['feedback']['research_question_id'] == 'scientific_revision_0'
    assert previous['costs']['model_calls'] == previous['costs']['reported_tokens'] == 2
    assert previous['costs']['observation_cost'] == 3
    assert previous['costs']['python_execution_seconds'] == pytest.approx(
        checkpoints[-1]['costs']['python_execution_seconds'] + 0.1)
    continuation = ScriptedProvider([
        ('finish', {'lesson': 'Calculation interrupted; return actual unfinished result.'}),
        ('run', {'code': "print('Parent retains the failed confirmation')"}),
    ])
    resumed = prepared(tmp_path / 'resumed', continuation, agent_type=agent_type, calls=4)
    fresh = RestorableConfirmationLab()
    resumed.restore_research(previous, old.workspace, fresh, confirmation_seed=1709)
    result = resumed.run(fresh)
    first = continuation.requests[0]
    assert bool(first.get('llm_inquiry')) == (agent_type is CommonArchiveInquiryAgent)
    question = next(q for q in first['online_research']['questions']
                    if q['id'] == 'scientific_revision_0')
    assert any(row['event_index'] == 2 for row in question['executed_analyses'])
    assert result['paid_observations'] == previous['paid_observations']
    assert fresh.actions == [] and fresh.paid == 3
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 4
    assert sum(event['feedback'].get('status') == 'runtime_interrupted'
               for event in result['events']) == 1
    assert result['model'] is None and result['fresh_confirmation']['passed'] is False
