"""The formal hooks must not let PR2 reacquire ahead of native evidence consumption."""
from copy import deepcopy

import pytest

from hegel_machine.cognition.finite_native_author import export_finite_native_author
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent, ORGANIZATION_ORIGIN
from test_finite_native_author import ChoiceLab, POLICY, declaration
from test_inquiry_revision import action
from test_question_routing import future_question, probe, theory


def registered_scene(tmp_path):
    library = tmp_path / 'fixed_library.json'
    library.write_text('{"candidates": {}}')
    agent = RegisteredOrganizationAgent(None, tmp_path / 'work', arm='fixed_pr2', source_task=False,
        research_organization={'phase': 'source', 'initial_library_path': str(library),
            'exploration_policy': {'kind': 'seeded_legal', 'seed': 11},
            'theories': [theory('linear', Program('feature', name='u')),
                theory('constant_two', Program('const', value=2.)),
                theory('constant_four', Program('const', value=4.))]},
        knowledge={'files': {POLICY: export_finite_native_author(),
            'archive/failed.py': 'def predict(row): return 10.\n'}},
        config=InquiryLLMConfig(observation_budget=20, compute_budget=3000,
            confirmation_rows=2, execution_seconds=30., max_model_calls=0))
    lab = ChoiceLab()
    agent._execute('query', {'actions': [action(1.)]}, lab)
    current = future_question(agent)
    # Keep the task readout unobserved after the first distinguishing purchase;
    # its public input is independent of the experimental panel inputs.
    current['query_semantics']['query_expression']['record_index'] = len(agent.observations) + 1
    # Both legal panels initially split one model from two. Whichever is
    # selected, the fixture's realised u value leaves two explanations alive;
    # a genuine next pending plan can therefore be prepared by the next hook.
    current.update(candidate_theory_ids=['linear', 'constant_two', 'constant_four'],
        resolution_experiments=[probe(agent, 2., 'two'), probe(agent, 4., 'four')])
    # A registered task may itself have an executable contract. The pending
    # purchase returns focus to this parent, so its next real hook can prepare
    # a further experiment rather than stopping at a legacy prose obligation.
    agent._open_question({**deepcopy(current), 'id': 'task', 'parent_id': None})
    # A meaning change intentionally discards the previous executable plans;
    # install these public panels only after the typed task has been adopted.
    agent._open_question({'id': 'task', 'resolution_experiments': deepcopy(current['resolution_experiments'])})
    contract = declaration()
    contract['question'] = deepcopy(current)
    contract['predictor_paths'] = ['archive/failed.py']
    agent.knowledge['finite_native_author'] = contract
    initial = agent._execute('run', {'operator': 'native_condition_research',
        'research_question': current, 'programs': [Program('const', value=5.).to_dict(),
            Program('const', value=0.).to_dict()],
        'max_compute': contract['max_compute'], 'max_candidates': contract['max_candidates']}, lab)
    assert 'error' not in initial
    assert initial['native_condition_research']['fit_record_indices'] == [0]
    return agent, lab, current


def run_formal_steps(agent, lab, steps):
    return execute_procedure(agent, lab, POLICY, max_steps=steps, allow_confirmation=False,
        record_events=True, before_step=agent._before_model_turn, after_step=agent._after_model_turn)


def test_registered_hooks_refit_new_pending_query_evidence_before_any_second_purchase(tmp_path):
    agent, lab, current = registered_scene(tmp_path)
    owner = next(q for q in agent.questions if q['id'] == current['id'])
    original = deepcopy(owner['native_condition_research'][0])
    compared = run_formal_steps(agent, lab, 1)
    assert compared['steps'][0]['proposal']['arguments']['operator'] == 'research_choice'
    assert not agent.operator_results[-1].get('next_action')
    before = deepcopy(agent.costs)
    result = run_formal_steps(agent, lab, 2)
    first, second = result['steps']
    assert first['proposal']['arguments'] == {'pending_question': 'task'}
    assert second['proposal'].get('arguments', {}).get('operator') == 'native_condition_research', {
        'proposals': [step['proposal'] for step in result['steps']],
        'paid_rows': lab.paid, 'native_rounds': len(owner['native_condition_research']),
        'latest_native_observed_records': owner['native_condition_research'][-1]['observed_record_count'],
        'compute_delta': agent.costs['compute_cost'] - before['compute_cost']}
    assert lab.paid == 2 and len(lab.actions) == 2
    assert owner['native_condition_research'][0] == original
    updated = owner['native_condition_research'][1]
    assert updated['fit_record_indices'] == [0, 1] and updated['fit_evidence'] == [0, 1]
    parent = next(q for q in agent.questions if q['id'] == 'task')
    assert parent['pending_query']['actions']
    assert all(item['program'] in [kept['program'] for kept in updated['candidates']]
               for item in original['candidates'])
    assert updated['compute_cost'] == 6
    assert any(event.get('origin') == ORGANIZATION_ORIGIN for event in agent.events)
    assert agent.costs['observation_cost'] - before['observation_cost'] == 1
    assert agent.costs['model_calls'] == 0 and not agent.confirmations


@pytest.mark.parametrize('record_indices', [None, [0]])
def test_registered_hooks_do_not_repeat_a_native_after_observation_fit(tmp_path, record_indices):
    agent, lab, current = registered_scene(tmp_path)
    continuation = {'operator': 'native_condition_research', 'research_question': current,
                    'max_compute': 300, 'max_candidates': 16}
    if record_indices is not None:
        continuation['record_indices'] = record_indices
    feedback = agent._execute('query', {'actions': [action(2.)],
        'research_question': current, 'after_observation': [continuation]}, lab)
    assert 'after_observation_error' not in feedback
    owner = next(q for q in agent.questions if q['id'] == current['id'])
    rounds = deepcopy(owner['native_condition_research'])
    assert rounds[-1]['observed_record_count'] == 2
    before_compute = agent.costs['compute_cost']
    result = run_formal_steps(agent, lab, 1)
    proposal = result['steps'][0]['proposal']
    assert proposal.get('arguments', {}).get('operator') != 'native_condition_research'
    assert owner['native_condition_research'] == rounds
    assert agent.costs['compute_cost'] >= before_compute
    assert agent.costs['model_calls'] == 0 and not agent.confirmations


def test_registered_hooks_consume_an_existing_choice_after_automatic_routing(tmp_path):
    library = tmp_path / 'fixed_library.json'
    library.write_text('{"candidates": {}}')
    contract = declaration()
    contract['predictor_paths'] = ['archive/good.py', 'archive/failed.py']
    files = {POLICY: export_finite_native_author(),
        'archive/good.py': "def predict(row): return row['features']['u']\n",
        'archive/failed.py': 'def predict(row): return 0.\n'}
    agent = RegisteredOrganizationAgent(None, tmp_path / 'work', arm='fixed_pr2', source_task=False,
        research_organization={'phase': 'source', 'initial_library_path': str(library)},
        knowledge={'files': files, 'finite_native_author': contract},
        config=InquiryLLMConfig(observation_budget=20, compute_budget=3000,
            confirmation_rows=2, execution_seconds=30., max_model_calls=0))
    lab = ChoiceLab()
    agent._execute('query', {'actions': [action(2., 3.)]}, lab)
    initial = agent._execute('run', {'operator': 'native_condition_research',
        'research_question': contract['question'],
        'programs': [Program('feature', name='u').to_dict(), Program('const', value=0.).to_dict()],
        'max_compute': 300, 'max_candidates': 16}, lab)
    assert 'error' not in initial
    owner = next(q for q in agent.questions if q['id'] == contract['question']['id'])
    rounds = deepcopy(owner['native_condition_research'])
    assert rounds[-1]['observed_record_count'] == len(agent.observations)
    assert not any(q.get('pending_query') for q in agent.questions)
    before = deepcopy(agent.costs)
    result = run_formal_steps(agent, lab, 2)
    choices = [item for item in agent.operator_results if item.get('operator') == 'research_choice']
    diagnostic = {'proposals': [step['proposal'] for step in result['steps']],
        'operators': [item.get('operator') for item in agent.operator_results],
        'choice_next_actions': [item.get('next_action') for item in choices],
        'compute_delta': agent.costs['compute_cost'] - before['compute_cost'],
        'paid_rows': lab.paid}
    assert result['steps'][0]['proposal']['arguments']['operator'] == 'research_choice', diagnostic
    assert result['status'] == 'confirmation_not_authorized', diagnostic
    assert result['steps'][1]['proposal']['action'] == 'finish'
    assert len(choices) == 1
    assert choices[0]['evaluated_model']['source_model'] == 'archive/good.py'
    assert [item['path'] for item in choices[0]['evaluations']][:2] == contract['predictor_paths']
    assert owner['native_condition_research'] == rounds
    assert lab.paid == agent.costs['observation_cost'] == 2 and len(lab.actions) == 1
    assert agent.costs['model_calls'] == 0 and agent.model is None and not agent.confirmations


def current_second_native_round(agent, lab, question):
    feedback = agent._execute('query', {'actions': [action(2.)], 'research_question': question,
        'after_observation': [{'operator': 'native_condition_research', 'research_question': question,
                              'max_compute': 300, 'max_candidates': 16}]}, lab)
    assert 'after_observation_error' not in feedback
    owner = next(q for q in agent.questions if q['id'] == question['id'])
    assert len(owner['native_condition_research']) == 2
    return owner


def test_registered_pending_does_not_starve_current_native_bank_comparison(tmp_path):
    agent, lab, current = registered_scene(tmp_path)
    owner = current_second_native_round(agent, lab, current)
    rounds = deepcopy(owner['native_condition_research'])
    result = run_formal_steps(agent, lab, 1)
    proposal = result['steps'][0]['proposal']
    assert proposal['arguments'].get('operator') == 'research_choice', proposal
    paths = {item['model'] for round_ in rounds for item in round_['candidates']}
    assert set(proposal['arguments']['models']) == {'archive/failed.py', *paths}
    compared = agent.operator_results[-1]
    assert compared['operator'] == 'research_choice' and compared['paid_rows'] == 2
    assert {item['path'] for item in compared['evaluations']} == set(proposal['arguments']['models'])
    assert all(item['mse'] > 0 for item in compared['evaluations'])
    parent = next(q for q in agent.questions if q['id'] == 'task')
    assert parent['pending_query']['actions']
    assert owner['native_condition_research'] == rounds
    assert lab.paid == 2 and len(lab.actions) == 2
    assert agent.costs['model_calls'] == 0 and not agent.confirmations


def test_registered_pending_does_not_override_an_old_winners_finish(tmp_path):
    agent, lab, current = registered_scene(tmp_path)
    owner = current_second_native_round(agent, lab, current)
    rounds = deepcopy(owner['native_condition_research'])
    agent._save('archive/good.py', "def predict(row): return row['features']['u']\n")
    agent.knowledge['finite_native_author']['predictor_paths'].append('archive/good.py')
    result = run_formal_steps(agent, lab, 2)
    first, second = result['steps']
    assert first['proposal']['arguments'].get('operator') == 'research_choice'
    assert second['proposal']['action'] == 'finish', second['proposal']
    assert result['status'] == 'confirmation_not_authorized'
    compared = next(item for item in reversed(agent.operator_results)
                    if item.get('operator') == 'research_choice')
    assert compared['evaluated_model']['source_model'] == 'archive/good.py'
    assert compared['evaluated_model']['mse'] == 0
    parent = next(q for q in agent.questions if q['id'] == 'task')
    assert parent['pending_query']['actions']
    assert owner['native_condition_research'] == rounds
    assert lab.paid == 2 and len(lab.actions) == 2
    assert agent.costs['model_calls'] == 0 and agent.model is None and not agent.confirmations
