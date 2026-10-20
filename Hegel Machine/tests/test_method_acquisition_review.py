"""Live methods hand opaque acquisition bursts back to their author, not a solver."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from test_common_archive_online_research import ConfirmationLab, NEW_MODEL, READOUT
from test_inquiry_revision import action, revision, write


PATH = 'shared/research.py'


def prepared(tmp_path, body):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='hegel', source_task=False,
        config=InquiryLLMConfig(observation_budget=100, confirmation_rows=2,
            compute_budget=2000, execution_seconds=30., max_model_calls=0, token_budget=0))
    lab = ConfirmationLab()
    source = ('import json\ndef select_actions(spec, records, memory):\n'
              "    step = len(json.load(open('research_state.json'))['method_results'])\n" + body)
    write(agent, PATH, source, lab)
    write(agent, 'model.py', NEW_MODEL, lab)
    return agent, lab, source


def run_live(agent, lab, *, steps=8):
    return agent._execute('run', {'operator': 'procedure', 'path': PATH,
                                  'max_steps': steps, 'confirm': True}, lab)


@pytest.mark.parametrize('wrapped', [False, True])
def test_live_third_opaque_acquisition_returns_actual_records_without_spending(tmp_path, wrapped):
    acquisition = {'actions': [action(2.)]}
    proposal = {'action': 'query', 'arguments': acquisition} if wrapped else acquisition
    agent, lab, source = prepared(tmp_path, f'    return {proposal!r}\n')
    result = run_live(agent, lab)

    assert result['status'] == 'method_review_required'
    assert lab.paid == agent.costs['observation_cost'] == 2
    assert len(result['steps']) == len(agent.method_results) == 3
    review = result['review']
    assert review['kind'] == 'unassessed_acquisition_loop'
    assert review['record_indices'] == [0, 1]
    assert review['observation_cost'] == 2
    assert review['proposed_action'] == proposal
    assert result['fresh_confirmation'] is None
    assert agent.model is None and agent.costs['model_calls'] == 0
    assert (tmp_path / PATH).read_text() == source
    exposed = agent._online_research()['latest_method_review']
    assert {key: exposed[key] for key in review} == review
    assert exposed['subsequent_method_steps'] == 0

    # Returning to the author is not an acquisition prohibition or a scientific verdict.
    direct = agent._execute('query', {'path': PATH}, lab)
    assert direct['observation_cost'] == 1
    assert lab.paid == 3


@pytest.mark.parametrize('next_step', ['model', 'evaluate'])
def test_two_acquisitions_do_not_block_declared_confirmation_or_paid_analysis(tmp_path, next_step):
    terminal = ({'model': 'model.py'} if next_step == 'model' else
                {'action': 'evaluate', 'arguments': {'model': 'model.py'}})
    agent, lab, _ = prepared(tmp_path,
        f"    if step < 2: return {{'actions': [{action(2.)!r}]}}\n"
        f'    if step == 2: return {terminal!r}\n'
        "    return {'stop': 'The authored paid-data analysis is complete.'}\n")
    result = run_live(agent, lab)

    assert 'review' not in result
    if next_step == 'model':
        assert result['status'] == 'completed'
        assert result['fresh_confirmation']['passed']
        assert lab.paid == 4
    else:
        assert result['status'] == 'method_stopped'
        assert result['steps'][2]['feedback']['mse'] == 0.
        assert lab.paid == 2


@pytest.mark.parametrize('intermediate', ['analysis', 'write'])
def test_explicit_inspection_or_edit_allows_a_new_acquisition_burst(tmp_path, intermediate):
    step = ({'action': 'run', 'arguments': {'code':
                "import json; print(len(json.load(open('research_state.json'))['method_results']))"}}
            if intermediate == 'analysis' else
            {'action': 'write', 'arguments': {'path': 'analysis.txt', 'code': 'revised plan'}})
    agent, lab, _ = prepared(tmp_path,
        f'    if step == 2: return {step!r}\n'
        "    if step == 5: return {'stop': 'The two planned batches are complete.'}\n"
        f"    return {{'actions': [{action(2.)!r}]}}\n")
    result = run_live(agent, lab)
    assert result['status'] == 'method_stopped' and 'review' not in result
    assert lab.paid == 4


@pytest.mark.parametrize('continuation', ['prospective', 'after_observation'])
def test_acquisitions_with_real_evidence_analysis_keep_running(tmp_path, continuation):
    acquisition = {'actions': [action(2.)]}
    if continuation == 'prospective':
        acquisition['revision'] = revision(['model.py'], readout='readout.py')
    else:
        acquisition['after_observation'] = [{'operator': 'analyze_evidence', 'path': 'paid.py'}]
    agent, lab, _ = prepared(tmp_path,
        "    if step == 4: return {'stop': 'All planned evidence has been analyzed.'}\n"
        f'    return {acquisition!r}\n')
    write(agent, 'readout.py', READOUT, lab)
    write(agent, 'paid.py', '''def readout(result, spec, memory):
    indices = [i for record in result['observations'] for i in record['row_indices']]
    return {'status':'inconclusive', 'answer':{'count':len(indices)}, 'evidence':indices}
''', lab)
    result = run_live(agent, lab)

    assert result['status'] == 'method_stopped' and 'review' not in result
    assert lab.paid == 4
    assert len(agent.revisions) == 4
    assert all(record.get('authored_answer') for record in agent.revisions)
    if continuation == 'prospective':
        assert all(record['forecasts'][0]['prospective_mse'] == 0. for record in agent.revisions)
    else:
        assert [record['authored_answer']['answer']['count'] for record in agent.revisions] == [1, 2, 3, 4]


def test_frozen_eight_acquisition_program_retains_its_execution_protocol(tmp_path):
    agent, lab, source = prepared(tmp_path, f"    return {{'actions': [{action(2.)!r}]}}\n")
    result = execute_procedure(agent, lab, PATH, max_steps=8, record_events=True)
    assert result['status'] == 'method_step_limit' and 'review' not in result
    assert lab.paid == 8
    assert len(agent.events) == len(agent.method_results) == 8
    assert all(event['usage']['total_tokens'] == 0 for event in agent.events)
    assert (tmp_path / PATH).read_text() == source


def test_old_confirmation_is_not_reported_as_a_new_procedure_confirmation(tmp_path):
    agent, lab, _ = prepared(tmp_path, f"    return {{'actions': [{action(2.)!r}]}}\n")
    write(agent, 'wrong.py', 'def predict(row): return 999.\n', lab)
    failed = agent._execute('finish', {'model': 'wrong.py'}, lab)
    assert failed['fresh_confirmation']['passed'] is False
    previous = deepcopy(agent.confirmations)
    before_paid = lab.paid

    result = run_live(agent, lab, steps=1)
    assert result['status'] == 'method_step_limit'
    assert result['fresh_confirmation'] is None
    assert agent.confirmations == previous
    assert lab.paid == before_paid + 1


def locally_adequate_globally_rejected(tmp_path, *, repeat_axis=False):
    from test_native_condition_inquiry import prepared as native_agent, PaidConditionalLab, execute, ROOT
    from test_native_condition_research import random_sequence

    agent, lab = native_agent(tmp_path), PaidConditionalLab()
    background = {'drive': 0., 'switch': 0.}
    sequences = [random_sequence(29), [dict(background) for _ in range(16)],
        [dict(background, drive=1. if index == 4 else 0.) for index in range(16)],
        [dict(background, switch=-1. if index == 4 else 0.) for index in range(16)]]
    if repeat_axis:
        sequences.append(deepcopy(sequences[-1]))
    execute(agent, lab, 'query', {'question': ROOT, 'actions': [
        {'kind': 'trajectory', 'channels': [], 'sequence': sequence} for sequence in sequences]})
    native = execute(agent, lab, 'run', {'operator': 'native_condition_research',
        'research_question': {'id': ROOT['id']}})['native_condition_research']
    candidate, = native['candidates']
    assert candidate['fit_mse'] > .1
    assert all(item['status'] == 'smooth' for item in native['evidence'])
    source = ("import json\ndef select_actions(*args):\n"
              "    steps = json.load(open('research_state.json'))['method_results']\n"
              f"    if not steps: return {{'action':'evaluate','arguments':{{'model':{candidate['model']!r}}}}}\n"
              "    return {'stop':'The candidate does not fit; leave it unresolved.'}\n")
    write(agent, PATH, source, lab)
    agent.knowledge['manifest'] = {'hypotheses': [], 'research_programs': [
        {'id': 'retained_method', 'path': PATH, 'max_steps': 4}], 'support_questions': []}
    write(agent, 'shared/knowledge.json', json.dumps(agent.knowledge['manifest']), lab)
    return agent, lab, native, source


def test_global_review_keeps_all_contributing_matched_records(tmp_path):
    agent, lab, native, source = locally_adequate_globally_rejected(tmp_path, repeat_axis=True)
    evidence = next(item for item in native['evidence'] if item['control'] == 'switch')
    assert evidence['record_indices'] == [3, 4]
    before = agent.costs['observation_cost']
    result = run_live(agent, lab)
    review = next(item for item in result['review']['local_evidence']
                  if item['control'] == 'switch')
    for key in ('matched_pairs', 'reference_record_indices', 'record_indices',
                'reference_evidence', 'evidence'):
        assert review[key] == evidence[key]
    assert 'record_index' not in review
    assert agent._online_research()['latest_method_review']['local_evidence'] == result['review']['local_evidence']
    assert agent.costs['observation_cost'] == lab.paid == before
    assert agent.costs['model_calls'] == 0 and agent.confirmations == []
    assert (tmp_path / PATH).read_text() == source


def test_global_rejection_returns_existing_experiment_and_refit_to_author(tmp_path):
    from test_native_condition_inquiry import decision_context

    agent, lab, native, source = locally_adequate_globally_rejected(tmp_path)
    before = deepcopy(agent.costs)
    outcome = run_live(agent, lab)
    assert outcome['status'] == 'method_review_required'
    review = outcome['review']
    assert review['kind'] == 'global_candidate_misfit'
    assert review['mse'] == outcome['steps'][0]['feedback']['mse'] > .1
    assert review['current_paid_rows'] == 64
    assert review['available_experiments']
    for proposed in review['available_experiments']:
        original = native['proposed_experiments'][proposed['proposal_index']]
        assert proposed['query_action'] == original['query_action']
        assert proposed['query_action']['arguments']['after_observation']
    assert agent.costs['model_calls'] == before['model_calls'] == 0
    assert agent.costs['observation_cost'] == before['observation_cost'] == lab.paid == 64
    assert agent.confirmations == [] and agent.model is None
    assert (tmp_path / PATH).read_text() == source
    assert agent._online_research()['latest_method_review']['kind'] == 'global_candidate_misfit'
    method, = decision_context(agent, lab)['method_decision']['methods']
    assert method['continuation']['review'] == review

    # The author can choose one existing proposal. It then buys and refits
    # normally; the review itself has performed neither operation.
    action = review['available_experiments'][0]['query_action']
    bought = agent._execute(action['action'], action['arguments'], lab)
    assert bought['observation_cost'] > 0
    assert bought['after_observation'][0]['operator'] == 'native_condition_research'
    assert len(agent.questions[0]['native_condition_research']) == 2


@pytest.mark.parametrize('supported', [False, True])
def test_stopped_old_method_returns_same_bank_choice_after_new_evidence(tmp_path, supported):
    from hegel_machine.cognition.structural_inquiry import native_global_misfit_review
    from test_native_condition_inquiry import decision_context

    agent, lab, native, source = locally_adequate_globally_rejected(tmp_path)
    # This is the actual pre-repair state: the method has already stopped.
    original = execute_procedure(agent, lab, PATH, max_steps=4, record_events=True)
    assert original['status'] == 'method_stopped'
    agent._execute('query', {'actions': [{'kind': 'trajectory', 'channels': [],
        'sequence': [{'drive': .2, 'switch': -.4}]}]}, lab)
    if supported:
        agent.evaluated_model.update(current_mse=0., evaluated_rows=len(agent.rows))
    before = deepcopy(agent.costs)
    assert native_global_misfit_review(agent, PATH) is None
    method, = decision_context(agent, lab)['method_decision']['methods']
    if supported:
        assert 'continuation' not in method
    else:
        review = method['continuation']['review']
        assert review['kind'] == 'research_evidence_extended'
        assert review['previous_paid_rows'] == 64 and review['current_paid_rows'] == 65
        assert review['new_paid_rows'] == 1
        assert review['resume_action']['arguments'] == {
            'operator': 'native_condition_research', 'research_question': {'id': 'task'}}
        assert 'available_experiments' not in review
    assert agent.costs == before

    outcome = run_live(agent, lab)
    if supported:
        assert outcome['status'] == 'method_stopped' and 'review' not in outcome
    else:
        assert outcome['status'] == 'method_review_required'
        assert outcome['review']['kind'] == 'research_evidence_extended'
        assert outcome['review']['resume_action'] == review['resume_action']
    assert agent.costs['observation_cost'] == 65 and agent.costs['model_calls'] == 0
    assert agent.confirmations == []
    assert (tmp_path / PATH).read_text() == source
    if not supported:
        selected = review['resume_action']
        exhausted = agent._execute(selected['action'], {
            **selected['arguments'], 'max_compute': 0}, lab)
        assert exhausted['native_condition_research']['candidates'] == []
        method, = decision_context(agent, lab)['method_decision']['methods']
        assert method['continuation']['review']['kind'] == 'research_evidence_extended'
        refreshed = agent._execute(selected['action'], selected['arguments'], lab)
        assert len(refreshed['native_condition_research']['fit_evidence']) == 65
        assert any(candidate['source'] == 'retained_program'
                   for candidate in refreshed['native_condition_research']['candidates'])
        method, = decision_context(agent, lab)['method_decision']['methods']
        assert 'continuation' not in method
        assert agent._online_research()['latest_method_review'] is None


@pytest.mark.parametrize('condition', ['fresh_rows', 'stale_evaluation', 'supported_candidate', 'already_purchased', 'frozen'])
def test_global_rejection_review_does_not_override_supported_stale_or_frozen_work(tmp_path, condition):
    from hegel_machine.cognition.structural_inquiry import native_global_misfit_review

    agent, lab, native, source = locally_adequate_globally_rejected(tmp_path)
    if condition == 'frozen':
        outcome = execute_procedure(agent, lab, PATH, max_steps=4, record_events=True)
        assert outcome['status'] == 'method_stopped' and 'review' not in outcome
        assert agent._online_research()['latest_method_review'] is None
    else:
        run_live(agent, lab, steps=1)
        if condition == 'fresh_rows':
            agent._execute('query', {'actions': [{'kind': 'trajectory', 'channels': [],
                'sequence': [{'drive': .2, 'switch': -.4}]}]}, lab)
        elif condition == 'supported_candidate':
            # A separately supported retained candidate is an actual reason
            # not to hand a previously rejected alternative back for repair.
            agent.evaluated_model.update(current_mse=0., evaluated_rows=len(agent.rows))
        elif condition == 'stale_evaluation':
            agent.method_results[-1]['feedback']['paid_residuals']['evaluated_rows'] -= 1
        else:
            existing = agent.questions[0]['native_condition_research'][0]
            for proposal in existing['proposed_experiments']:
                proposal['query_action']['arguments']['actions'] = [
                    agent.observations[0]['action']]
        assert native_global_misfit_review(agent, PATH) is None
    assert (tmp_path / PATH).read_text() == source
