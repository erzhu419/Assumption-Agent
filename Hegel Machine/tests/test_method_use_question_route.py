"""Paid method questions execute through the ordinary agenda and parent route."""

from copy import deepcopy

from hegel_machine.cognition.finite_native_author import export_revision_candidates, select_actions
from hegel_machine.cognition.method_feedback import _read
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.question_agenda import build_agenda
from test_common_method_feedback import scene


def measured_method(path):
    agent, lab, method, parent = scene(path, useful=False)
    agent._execute('run', {'operator': 'research_route', 'research_question': parent,
        'allocation': {'compute': 3000}}, lab)
    owner = next(item for item in agent.questions if item['id'] == parent['id'])
    exported = export_revision_candidates(agent, owner)
    models = ['old.py', *[item['model'] for item in owner['revision_candidate_exports']]]
    assert len(models) > 1 and not exported['errors']
    compared = agent._execute('run', {'operator': 'research_choice', 'models': models,
        'research_question': {'id': parent['id']}}, lab)
    record = _read(agent, agent.theory_consolidation['method_feedback'][method['id']]['outcome_refs'][0])
    assert record['status'] == 'completed' and compared['models_requested'] == compared['models_completed']
    execution = next(value for key, value in agent.theory_consolidation['executions'].items()
                     if key.startswith('research_program_transfer:'))
    return agent, lab, method, owner, deepcopy(execution['basis']['binding']), compared


def publish(agent, method, parent, binding):
    question = {'id': 'paid_method_use', 'kind': 'method_use',
        'question': 'Should the paid method be repeated, or should ordinary research continue?',
        'answer_type': 'research_action', 'parent_id': parent['id'],
        'parent_question_ref': {'question_id': parent['id'],
                                'meaning_version': parent.get('meaning_version', 1)},
        'queue': 'theory', 'mission': {'mission_id': 'paid_method_conditions',
            'purpose': 'method_learning', 'allocation': {'compute': 2000}},
        'use_sites': [{'kind': 'select_research_action', 'parent_id': parent['id']}],
        'theory_target': deepcopy(parent['theory_target']),
        'method_use_plan': {'candidate_id': method['id'], 'binding': deepcopy(binding)},
        'agenda_basis': {'actual_paid_return': deepcopy(
            agent.theory_consolidation['method_feedback'][method['id']]['outcome_refs'])}}
    return agent._open_question(question)


def test_paid_unhelpful_question_is_executed_then_ordinary_route_defers_same_work(tmp_path):
    agent, lab, method, parent, binding, _ = measured_method(tmp_path)
    question = publish(agent, method, parent, binding)
    agenda = build_agenda([question], {}, [], budget={'compute': 2000, 'author_calls': 0})
    assert agenda['selected_question_id'] == question['id']
    assert agenda['theory_agenda'][0]['work_kind'] == 'method_conditions'
    before = deepcopy(agent.costs)
    result = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}, 'allocation': {'compute': 2000}}, lab)
    assert result['status'] == 'method_use_decided'
    assert result['method_use_decision']['admission'] == 'defer'
    assert result['selected_research_action'] == 'ordinary_research'
    assert result['condition_learning']['frozen_candidates']
    assert agent.costs['compute_cost'] > before['compute_cost']
    follow = result['next_action']
    continued = agent._execute(follow['action'], follow['arguments'], lab)
    work = continued['frozen_recall']['work']
    assert any(item['status'] == 'method_use_deferred' for item in work)
    assert all(not item.get('theory_ids') for item in work if item['status'] == 'method_use_deferred')
    assert lab.paid == 0 and agent.costs['model_calls'] == 0 and not agent.confirmations
    assert not result['support_updated'] and not result['independent_confirmation']
    # Analysis can finish with the world question unresolved; it does not loop
    # forever ahead of ordinary work under the same paid evidence.
    repeated = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}}, lab)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0


def test_changed_paid_contents_reopen_bounded_method_work_instead_of_banning_it(tmp_path):
    agent, lab, method, parent, binding, _ = measured_method(tmp_path)
    question = publish(agent, method, parent, binding)
    agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}, 'allocation': {'compute': 2000}}, lab)
    last = agent.rows[-1]
    agent.rows[-1] = Observation(last.features, last.target + .25, last.history)
    agent._write_evidence()
    # The publisher updates the existing question's actual evidence basis.
    question['agenda_basis']['new_paid_evidence'] = 'replacement_of_same_size_view'
    question['method_use_plan']['binding']['candidate_version'] = agent.theory_consolidation[
        'candidates'][method['id']]['version']
    result = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}, 'allocation': {'compute': 1500}}, lab)
    assert result['status'] == 'method_use_unresolved'
    assert result['method_use_decision']['admission'] == 'bounded_exploration'
    assert result['method_use_decision']['question']['parent_use']['kind'] == 'select_research_action'
    before = agent.costs['compute_cost']
    follow = result['next_action']
    continued = agent._execute(follow['action'], follow['arguments'], lab)
    assert continued['frozen_recall']['generated_theory_ids']
    assert agent.costs['compute_cost'] > before
    assert lab.paid == 0 and not result['independent_confirmation']


def test_unfunded_method_question_stays_visible_without_becoming_author_work(tmp_path):
    agent, _, method, parent, binding, _ = measured_method(tmp_path)
    question = publish(agent, method, parent, binding)
    question['mission']['allocation']['compute'] = 0
    agenda = build_agenda([question], {}, [], budget={'compute': 100, 'author_calls': 0})
    assert agenda['selected_question_id'] is None
    assert agenda['theory_agenda'][0]['work_kind'] == 'method_conditions'
    assert agenda['theory_agenda'][0]['reason'] == 'theory_allocation_missing'


def test_incomplete_condition_allocation_reuses_attempt_and_can_use_existing_larger_allowance(tmp_path):
    agent, lab, method, parent, binding, _ = measured_method(tmp_path)
    question = publish(agent, method, parent, binding)
    arguments = {'operator': 'research_route', 'research_question': {'id': question['id']},
        'allocation': {'compute': 0}}
    incomplete = agent._execute('run', arguments, lab)
    assert incomplete['status'] == 'scoped_unknown' and incomplete['required_compute'] == 1
    assert 'completed_agenda_basis' not in question
    assert agent._execute('run', arguments, lab)['cache_hit']
    arguments['allocation']['compute'] = 1
    completed = agent._execute('run', arguments, lab)
    assert completed['status'] == 'method_use_decided'
    assert completed['method_use_decision']['admission'] == 'defer'
    assert completed['new_compute_cost'] == 1


def test_ordinary_author_consumes_ready_method_question_and_parent_action_once(tmp_path, monkeypatch):
    agent, lab, method, parent, binding, comparison = measured_method(tmp_path)
    question = publish(agent, method, parent, binding)
    agent.research_agenda = build_agenda([question], {}, [], budget={'compute': 2000, 'author_calls': 0})
    parent['context_ref'] = deepcopy(parent['theory_target']['context_ref'])
    models = ['old.py', *[item['model'] for item in parent['revision_candidate_exports']]]
    memory = {'finite_native_author': {'question': deepcopy(parent),
        'predictor_paths': ['old.py'], 'predictor_catalog_ref': {'file': 'retained_catalog.json'},
        'max_compute': 3000, 'max_candidates': 4, 'boundary_resolution': .1},
        'program_results': [{'proposal': {'action': 'run', 'arguments': {
            'operator': 'research_choice', 'models': models}}, 'feedback': comparison,
            'evidence_after': {'rows': len(agent.rows)}}]}
    monkeypatch.chdir(agent.workspace)
    agent._prepare_program_state()
    selected = select_actions(lab.public_spec(), agent.observations, memory)
    assert selected['arguments']['operator'] == 'research_route'
    assert selected['arguments']['research_question']['id'] == question['id']
    assert selected['arguments']['allocation']['compute'] == 2000
    feedback = agent._execute(selected['action'], selected['arguments'], lab)
    assert feedback['method_use_decision']['admission'] == 'defer'
    memory['program_results'].append({'proposal': deepcopy(selected), 'feedback': feedback,
        'evidence_after': {'rows': len(agent.rows)}})
    agent._prepare_program_state()
    continued = select_actions(lab.public_spec(), agent.observations, memory)
    assert continued['arguments'] == feedback['next_action']['arguments']
    result = agent._execute(continued['action'], continued['arguments'], lab)
    assert any(item['status'] == 'method_use_deferred' for item in result['frozen_recall']['work'])
    memory['program_results'].append({'proposal': deepcopy(continued), 'feedback': result,
        'evidence_after': {'rows': len(agent.rows)}})
    agent._prepare_program_state()
    ordinary = select_actions(lab.public_spec(), agent.observations, memory)
    assert ordinary['action'] == 'finish'
    assert lab.paid == 0 and not agent.confirmations and agent.costs['model_calls'] == 0
