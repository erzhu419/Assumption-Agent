"""Published paid feedback changes ordinary author actions without new science."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint, write_json_checkpoint
from hegel_machine.cognition.finite_native_author import select_actions
from hegel_machine.cognition.method_feedback import method_evidence_basis, publish_method_use_questions
from hegel_machine.cognition.models import Observation
from test_method_use_question_route import measured_method


def publication(agent, method, parent):
    agent.research_mission = {'mission_id': 'existing_method_mission',
        'purpose': 'method_learning', 'phase': 'theory', 'allocation': {'compute': 2000}}
    reference = agent.theory_consolidation['method_feedback'][method['id']]['outcome_refs'][0]
    proposal = {'kind': 'method_use', 'answer_type': 'research_action',
        'question': 'Does the retained paid feedback warrant this method or ordinary research?',
        'evidence_refs': [deepcopy(reference)],
        'alternatives': ['try_or_continue_method', 'ordinary_research']}
    context = {'question_ref': {'question_id': parent['id'],
        'meaning_version': parent.get('meaning_version', 1)},
        'theory_target': deepcopy(parent['theory_target'])}
    return proposal, context, reference


def test_condition_version_and_per_seed_binding_reuse_the_same_paid_question(tmp_path):
    agent, lab, method, parent, binding, _ = measured_method(tmp_path)
    parent['mission'] = {'purpose': 'task_prediction'}
    proposal, context, _ = publication(agent, method, parent)
    first, = publish_method_use_questions(agent, [proposal], binding=binding, context=context)
    current = next(q for q in agent.questions if q['id'] == first['question_id'])
    assert agent.research_agenda['selected_question_id'] == current['id']
    assert 'mission' not in current  # Use the existing global method allocation.
    agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': current['id']}, 'allocation': {'compute': 2000}}, lab)
    version = agent.theory_consolidation['candidates'][method['id']]['version']
    revised = {**deepcopy(binding), 'candidate_version': version,
        'source_ref': deepcopy(method['research_program_source_refs'][-1]), 'seed_index': 1}
    repeated, = publish_method_use_questions(agent, [proposal], binding=revised, context=context)
    assert repeated == first
    assert current['method_use_plan']['binding']['candidate_version'] == version
    assert current['agenda_basis']['evidence_basis'] == method_evidence_basis(binding, agent=agent)
    assert 'source_ref' not in current['agenda_basis']['evidence_basis']
    assert current['agenda_basis']['evidence_basis']['source_refs'] == binding['source_refs']
    assert agent.research_mission['resource_usage']['compute'] == 1


def test_same_size_actual_paid_replacement_creates_a_distinct_question(tmp_path):
    agent, _, method, parent, binding, _ = measured_method(tmp_path)
    proposal, context, _ = publication(agent, method, parent)
    first, = publish_method_use_questions(agent, [proposal], binding=binding, context=context)
    old = next(q for q in agent.questions if q['id'] == first['question_id'])
    old_basis = deepcopy(old['agenda_basis']['evidence_basis'])
    row = agent.rows[-1]
    agent.rows[-1] = Observation(row.features, row.target + .25, row.history)
    second, = publish_method_use_questions(agent, [proposal], binding=binding, context=context)
    assert second['question_id'] != first['question_id']
    new = next(q for q in agent.questions if q['id'] == second['question_id'])
    assert old['agenda_basis']['evidence_basis'] == old_basis
    assert new['agenda_basis']['evidence_basis']['paid_cutoff'] == old_basis['paid_cutoff']
    assert new['agenda_basis']['evidence_basis']['paid_evidence_ref'] != old_basis['paid_evidence_ref']


def test_historical_context_keeps_its_actual_paid_view_instead_of_current_rows(tmp_path):
    agent, _, method, parent, binding, _ = measured_method(tmp_path)
    proposal, context, _ = publication(agent, method, parent)
    context['evidence_basis'] = method_evidence_basis(binding, agent=agent)
    old_rows = read_json_checkpoint(context['evidence_basis']['paid_evidence_ref']['file'])
    row = agent.rows[-1]
    agent.rows[-1] = Observation(row.features, row.target + 10., row.history)
    snapshots = list((agent.workspace / 'method_feedback').glob('paid_*.json'))
    reference, = publish_method_use_questions(agent, [proposal], binding=binding, context=context)
    question = next(q for q in agent.questions if q['id'] == reference['question_id'])
    assert question['agenda_basis']['evidence_basis'] == context['evidence_basis']
    assert list((agent.workspace / 'method_feedback').glob('paid_*.json')) == snapshots
    assert read_json_checkpoint(context['evidence_basis']['paid_evidence_ref']['file']) == old_rows


@pytest.mark.parametrize('changed', ['paired_future', 'incomplete_status'])
def test_inplace_feedback_refreshes_same_question_and_author_executes_new_decision(tmp_path, monkeypatch, changed):
    agent, lab, method, parent, binding, comparison = measured_method(tmp_path)
    proposal, context, feedback_ref = publication(agent, method, parent)
    context['evidence_basis'] = method_evidence_basis(binding, agent=agent)
    published, = publish_method_use_questions(agent, [proposal], binding=binding, context=context)
    question = next(q for q in agent.questions if q['id'] == published['question_id'])
    assert agent.research_agenda['selected_question_id'] == question['id']
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
    first = select_actions(lab.public_spec(), agent.observations, memory)
    assert first['arguments']['research_question']['id'] == question['id']
    result = agent._execute(first['action'], first['arguments'], lab)
    assert result['method_use_decision']['admission'] == 'defer'
    memory['program_results'].append({'proposal': deepcopy(first), 'feedback': result,
        'evidence_after': {'rows': len(agent.rows)}})
    prior = deepcopy(question['agenda_basis'])
    record = read_json_checkpoint(feedback_ref['file'])
    pair_path = agent.workspace / 'retained_paired_loss_fixture.json'
    pair = {'before_loss': .2, 'after_loss': .01, 'reference_rows': 8,
        'same_rows': True, 'data_role': 'new_paid_readings', 'execution_complete': True,
        'source_refs': [{'file': str(pair_path), 'json_path': []}]}
    write_json_checkpoint(pair_path, {'before_loss': .2, 'after_loss': .01, 'rows': 8})
    if changed == 'paired_future':
        record['paired_future'] = pair
    else:
        record.update(status='incomplete', execution_complete=False)
    write_json_checkpoint(feedback_ref['file'], record)
    revised = {**deepcopy(binding), 'candidate_version':
        agent.theory_consolidation['candidates'][method['id']]['version']}
    updated, = publish_method_use_questions(agent, [proposal], binding=revised, context=context)
    assert updated == published
    assert question['agenda_basis'] != prior
    assert question['agenda_basis']['method_feedback_refs'] == prior['method_feedback_refs']
    assert agent.research_agenda['selected_question_id'] == question['id']
    agent._prepare_program_state()
    selected = select_actions(lab.public_spec(), agent.observations, memory)
    assert selected['arguments']['research_question']['id'] == question['id']
    assert selected != first
    result = agent._execute(selected['action'], selected['arguments'], lab)
    assert result['method_use_decision']['admission'] == (
        'admit' if changed == 'paired_future' else 'bounded_exploration')
    if changed == 'paired_future':
        assert result['method_use_decision']['deciding_return_basis']['kind'] == 'paired_future'
    assert not result['independent_confirmation'] and lab.paid == 0
