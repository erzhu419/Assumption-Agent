"""Scripted runtime checks, not evidence of autonomous scientific discovery."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_inquiry_revision import PublicLab, action, revision, write


OLD_MODEL = "def predict(row): return 0.\n"
NEW_MODEL = "def predict(row): return row['features']['u']\n"
BROKEN_READOUT = "def readout(result,spec,memory): raise ValueError('unfinished analysis')\n"
READOUT = """def readout(result,spec,memory):
    candidates = result['forecasts']
    rejected = [c['model'] for c in candidates if c['prospective_mse'] > 0]
    surviving = [c['model'] for c in candidates if c['prospective_mse'] == 0]
    return {'status':'refuted', 'answer':{'rejected':rejected, 'surviving':surviving},
            'evidence':result['observations'][0]['row_indices']}
"""
PARENT = {'id': 'parent', 'question': 'Predict this apparatus response',
          'parent_id': None, 'depends_on': []}
CHILD = {'id': 'child', 'question': 'Do these compatible predictions agree away from zero?',
         'parent_id': 'parent', 'depends_on': []}


class ConfirmationLab(PublicLab):
    def confirmation_actions(self, length, rng):
        return [action(*[rng.uniform(-2, 2) for _ in range(length)])]


class ScriptedProvider:
    def __init__(self, steps):
        self.steps = steps
        self.requests = []

    def next_action(self, request):
        index = len(self.requests)
        self.requests.append(deepcopy(request))
        selected, arguments = self.steps[index]
        return {'action': selected, 'arguments': deepcopy(arguments),
                'usage': {'backend': 'scripted_runtime_check', 'total_tokens': 1}}


def prepared(tmp_path, arm, provider=None, calls=8):
    return CommonArchiveInquiryAgent(provider, tmp_path, arm=arm, source_task=False,
        config=InquiryLLMConfig(observation_budget=12, confirmation_rows=2,
            compute_budget=1000, execution_seconds=30., max_model_calls=calls),
        knowledge={'files': {
            'old.py': OLD_MODEL, 'new.py': NEW_MODEL, 'readout.py': BROKEN_READOUT,
            'shared/probe.py': "def select_actions(spec,observations,memory): return []\n",
            'shared/knowledge.json': json.dumps({
                'hypotheses': [{'id': 'final', 'path': 'final.py'}],
                'research_programs': [{'id': 'probe', 'path': 'shared/probe.py'}],
                'support_questions': []})}})


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_online_readout_repair_returns_paid_counterexample_to_parent(tmp_path, arm):
    """The next model decision must see the real failure, then its corrected answer."""
    provider = ScriptedProvider([
        ('query', {'actions': [action(0)], 'question': PARENT,
                   'notes': 'Both candidates agree on the paid zero input.'}),
        ('query', {'actions': [action(2)], 'question': CHILD,
                   'revision': revision(['old.py', 'new.py'], readout='readout.py'),
                   'notes': 'Compare the two forecasts on this experiment.'}),
        ('write', {'path': 'readout.py', 'code': READOUT,
                   'notes': 'The experiment ran; repair its failed analysis using paid evidence.'}),
        ('run', {'operator': 'analyze_revision', 'revision_index': 0, 'path': 'readout.py',
                 'notes': 'Analyze the saved forecasts and observed response.'}),
        ('evaluate', {'model': 'final.py', 'code': NEW_MODEL,
                      'research_question': {'id': 'parent'},
                      'notes': 'The child refuted old.py; evaluate the revised parent model.'}),
        ('finish', {'model': 'final.py', 'notes': 'Request a fresh check of the revised model.'}),
    ])
    agent, lab = prepared(tmp_path, arm, provider), ConfirmationLab()
    result = agent.run(lab)

    assert result['status'] == 'completed'
    assert result['fresh_confirmation']['passed']
    assert result['costs']['model_calls'] == len(provider.requests) == 6
    assert lab.actions[:2] == [action(0), action(2)]
    assert lab.paid == result['costs']['observation_cost'] == 4
    assert len(lab.actions) == 3  # Initial data, experiment, and final fresh check only.

    failure_request = provider.requests[2]
    assert failure_request['research_notes'] == 'Compare the two forecasts on this experiment.'
    assert 'unfinished analysis' in failure_request['events'][-1]['feedback']['revision']['readout_error']
    assert failure_request['online_research']['pending_analyses']
    returned = provider.requests[4]
    assert returned['costs']['observation_cost'] == 2
    assert returned['costs']['compute_cost'] == provider.requests[3]['costs']['compute_cost']
    assert returned['research_notes'] == 'Analyze the saved forecasts and observed response.'
    questions = {q['id']: q for q in returned['online_research']['questions']}
    assert set(questions) >= {'parent', 'child'}
    assert questions['child']['status'] == 'refuted'
    assert questions['child']['evidence'] == [1]
    parent_return = next(item for item in returned['online_research']['parent_returns']
                         if item['parent_id'] == 'parent')
    assert parent_return['answers'][-1]['question_id'] == 'child'
    assert parent_return['answers'][-1]['answer'] == {'rejected': ['old.py'], 'surviving': ['new.py']}
    assert not returned['online_research']['pending_analyses']
    assert provider.requests[5]['events'][-1]['feedback']['mse'] == 0
    assert (tmp_path / 'final.py').read_text() == NEW_MODEL

    record = result['revision_proposals'][0]
    assert len(result['revision_proposals']) == 1
    assert record['evidence_cutoff'] == {'records': 1, 'rows': 1}
    assert record['forecasts'][0]['predictions'] == [0.]
    assert record['forecasts'][1]['predictions'] == [2.]
    assert record['forecasts'][0]['prospective_mse'] == 4.
    assert record['forecasts'][1]['prospective_mse'] == 0.
    assert record['readout_source'] == BROKEN_READOUT
    assert len(record['analysis_attempts']) == 2
    assert 'unfinished analysis' in record['analysis_attempts'][0]['error']
    assert record['analysis_attempts'][1]['source'] == READOUT
    for question_id in ('parent', 'child'):
        assert sum(q['id'] == question_id for q in result['online_questions']) == 1
    assert (tmp_path / 'research_notes.md').read_text() == 'Request a fresh check of the revised model.'


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_same_question_id_keeps_assignment_and_accumulates_actual_acquisitions(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm), PublicLab()
    agent._execute('query', {'actions': [action(1)], 'question': CHILD}, lab)
    narrower = {**CHILD, 'question': 'Check the remaining response at two'}
    agent._execute('query', {'actions': [action(2)], 'question': narrower}, lab)
    matching = [q for q in agent.questions if q['id'] == 'child']
    assert len(matching) == 1
    assert matching[0]['assignment']['question'] == CHILD['question']
    assert matching[0]['question'] == narrower['question']
    assert matching[0]['acquired_evidence'] == [0, 1]
    assert lab.paid == 2


def test_inline_evaluate_scores_exact_new_file_and_retains_supported_incumbent(tmp_path):
    agent, lab = prepared(tmp_path, 'code_react'), PublicLab()
    agent._execute('query', {'actions': [action(2)]}, lab)
    write(agent, 'candidate.py', OLD_MODEL, lab)
    evaluated = agent._execute('evaluate', {'model': 'candidate.py', 'code': NEW_MODEL}, lab)
    assert evaluated['mse'] == 0
    assert evaluated['executed_path'] == 'candidate.py'
    assert (tmp_path / 'candidate.py').read_text() == NEW_MODEL
    assert evaluated['evaluated_model']['source_model'] == 'candidate.py'
    incumbent = agent._evaluated_summary()['model']
    broken = "def predict(row): raise ValueError('new candidate failed')\n"
    failure = agent._execute('evaluate', {'model': 'candidate.py', 'code': broken}, lab)
    assert failure['returncode'] != 0
    assert 'new candidate failed' in failure['stderr']
    assert (tmp_path / 'candidate.py').read_text() == broken
    assert agent._evaluated_summary()['model'] == incumbent
    assert agent._score(incumbent)['mse'] == 0
    assert lab.paid == 1


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
@pytest.mark.parametrize('source_task', [False, True])
def test_displayed_phase_matches_actual_program_execution_requirement(tmp_path, arm, source_task):
    manifest = {'hypotheses': [{'id': 'response', 'path': 'shared/model.py'}],
                'research_programs': [{'id': 'probe', 'path': 'shared/probe.py'}],
                'support_questions': []}
    knowledge = {'manifest': manifest, 'files': {
        'shared/model.py': NEW_MODEL,
        'shared/probe.py': """def select_actions(spec,observations,memory):
    return {'actions':[{'kind':'trajectory','sequence':[{'u':2.}]}]}
""",
        'shared/knowledge.json': json.dumps(manifest)}}
    provider = ScriptedProvider([('finish', {'model': 'shared/model.py'})])
    agent = CommonArchiveInquiryAgent(provider, tmp_path, arm=arm, knowledge=knowledge,
        source_task=source_task, config=InquiryLLMConfig(observation_budget=12,
            confirmation_rows=2, max_model_calls=1, execution_seconds=30.))
    lab = ConfirmationLab()
    agent._execute('query', {'actions': [action(1)]}, lab)
    result = agent.run(lab)
    request = provider.requests[0]
    assert request['research_phase'] == ('source_learning' if source_task else 'continuation')
    source_requirement = 'During source learning, Hegel must actually query at least once'
    assert (source_requirement in request['task_interface']) is source_task
    if not source_task:
        assert 'A new program or\nquery is optional' in request['task_interface']
        assert 'finish registers your explicitly selected model path' in request['task_interface']
    assert [event['action'] for event in result['events']] == ['finish']
    if source_task and arm == 'hegel':
        assert result['model'] is None and lab.paid == 1
        assert 'source learning must actually execute' in result['events'][0]['feedback']['error']
        agent._execute('query', {'path': 'shared/probe.py', 'question': PARENT}, lab)
        completed = agent._execute('finish', {'model': 'shared/model.py'}, lab)
        assert completed['fresh_confirmation']['passed']
        assert lab.paid == 4  # Initial row, real source-program row, fresh confirmation.
    else:
        assert result['status'] == 'completed' and result['fresh_confirmation']['passed']
        assert lab.paid == 3  # Initial row and confirmation; no additional query.
