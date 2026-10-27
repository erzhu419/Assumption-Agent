"""Computed representation work reaches Common's ordinary research route."""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_agenda import build_agenda
from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_native_representation_inquiry import agent_with_records, retained_procedure, measured_failure_basis


class PublicOnlyLab:
    def public_spec(self):
        return {'controls': ['u'], 'feature_bounds': {'u': [-1., 1.]},
                'max_probe_length': 12, 'action_kinds': ['trajectory']}

    def execute(self, action):
        raise AssertionError('native representation routing bought data')


def scene(tmp_path):
    agent = CommonArchiveInquiryAgent(None, tmp_path / 'common', arm='hegel', source_task=False,
        config=replace(InquiryLLMConfig(), compute_budget=150000, max_model_calls=0))
    records = agent_with_records(tmp_path / 'records')
    agent.observations, agent.rows = records.observations, records.rows
    return agent, PublicOnlyLab()


def declaration():
    return {'id': 'representation:old', 'question': 'Which actual input distinctions are missing?',
        'kind': 'representation', 'origin': 'computed_representation_conflict',
        'parent_id': 'task', 'depends_on': [], 'answer_type': 'program',
        'use_sites': [{'kind': 'supply_component', 'model': 'old.py', 'slot': 'representation'}]}


def plan(**changes):
    return {'parent_theory_ids': ['old', 'strong_component'], 'validation_records': [4],
            'compute_limit': 50000, 'max_candidates': 4, **changes}


def route(agent, lab, *, incoming=None, with_theories=False):
    arguments = {'operator': 'research_route', 'research_question': declaration()}
    if incoming is not None:
        arguments['native_representation_plan'] = incoming
    if with_theories:
        arguments['theories'] = [
            {'id': 'old', 'program': Program('feature', name='u').to_dict()},
            {'id': 'strong_component', 'program': Program('const', value=0.).to_dict()}]
    return agent._execute('run', arguments, lab)


def test_declared_native_work_is_compute_ready_without_author_calls():
    question = {**declaration(), 'native_representation_plan': plan(),
                'pending_author_work': {'agenda_basis': {'meaning_version': 1, 'answer_version': None}}}
    agenda = build_agenda([question], {}, [], budget={'compute': 50000, 'author_calls': 0})
    assert agenda['selected_question_id'] == question['id']
    assert agenda['task_agenda'][0]['work_kind'] == 'native_representation'


def test_common_routes_native_work_exports_candidates_and_keeps_original_parent_pool(tmp_path):
    agent, lab = scene(tmp_path)
    before = deepcopy(agent.costs)
    result = route(agent, lab, incoming=plan(), with_theories=True)
    assert result['status'] == 'native_representation_candidates_ready'
    assert result['selected_model'] in agent.candidate_paths
    assert (agent.workspace / result['selected_model']).is_file()
    assert result['native_representation']['selected']['changed_prediction_rows'] > 0
    assert result['native_representation']['selected']['validation_mse'] < 1e-10
    assert set(result['retained_parent_theory_ids']) == {'old', 'strong_component'}
    assert set(agent.theory_index) == {'old', 'strong_component'}
    assert not result['parent_revision_completed'] and not result['representation_child_resolved']
    assert agent.model is None
    assert result['new_compute_cost'] == agent.costs['compute_cost'] - before['compute_cost']
    assert agent.costs['model_calls'] == before['model_calls'] == 0
    assert agent.costs['observation_cost'] == before['observation_cost']
    assert result['next_action']['arguments']['operator'] == 'research_choice'
    native = load_source_values([result['native_representation_ref']], agent.workspace)[0]['value']
    assert native['selected']['validation_mse'] == result['native_representation']['selected']['validation_mse']
    for exported in result['exported_models']:
        candidate = load_source_values([exported['source_ref']], agent.workspace)[0]['value']
        assert candidate['id'] == exported['candidate_id']
        assert (agent.workspace / exported['model']).read_text() == export_temporal_program(
            Program.from_dict(candidate['program']))
    question = next(q for q in agent.questions if q['id'] == 'representation:old')
    state_file = agent.workspace / 'receipt_state.json'
    state_file.write_text(json.dumps({'questions': agent.questions}))
    question_index = agent.questions.index(question)
    for index, candidate in enumerate(native['candidates']):
        old_ref = {'file': str(state_file), 'json_path': ['questions', question_index,
            'native_representation_research', 0, 'candidates', index, 'program']}
        assert load_source_values([old_ref], agent.workspace)[0]['value'] == candidate['program']
    assert all('program' not in entry and 'research_program' not in entry
               for entry in result['native_representation']['candidates'])
    assert len(json.dumps(result)) < len(json.dumps(native))


def test_same_route_reuses_work_and_changed_split_starts_a_distinct_research_record(tmp_path):
    agent, lab = scene(tmp_path)
    first = route(agent, lab, incoming=plan(), with_theories=True)
    before = agent.costs['compute_cost']
    reused = route(agent, lab)
    assert reused['cache_hit'] and reused['new_compute_cost'] == 0
    assert reused['exported_models'] == first['exported_models']
    assert agent.costs['compute_cost'] == before
    changed = route(agent, lab, incoming=plan(validation_records=[3]))
    assert not changed['cache_hit']
    question = next(q for q in agent.questions if q['id'] == 'representation:old')
    assert len(question['native_representation_research']) == 2
    assert question['native_representation_research'][0]['split']['validation_records'] == [4]
    assert question['native_representation_research'][1]['split']['validation_records'] == [3]
    assert all((agent.workspace / entry['model']).is_file() for entry in first['exported_models'])


def test_missing_declared_parent_is_unknown_and_does_not_shrink_the_pool(tmp_path):
    agent, lab = scene(tmp_path)
    result = route(agent, lab, incoming=plan(parent_theory_ids=['old', 'unavailable']), with_theories=True)
    assert result['status'] == 'scoped_unknown'
    assert result['missing_theory_ids'] == ['unavailable']
    assert result['new_compute_cost'] == 0
    assert set(agent.theory_index) == {'old', 'strong_component'}
    assert not agent.candidate_paths


def test_cold_question_restore_reuses_canonical_study_without_native_search(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    first = route(agent, lab, incoming=plan(), with_theories=True)
    saved = json.loads(json.dumps({
        'questions': agent.questions, 'theory_index': agent.theory_index,
        'costs': agent.costs, 'research_mission': agent.research_mission}))
    restored, lab = scene(tmp_path)
    for name, value in saved.items():
        setattr(restored, name, value)

    def unexpected(*args, **kwargs):
        raise AssertionError('cached native work was repeated')

    monkeypatch.setattr('hegel_machine.cognition.representation_inquiry.execute_native_representation_inquiry',
                        unexpected)
    before = deepcopy(restored.costs)
    result = route(restored, lab)
    assert result['cache_hit'] and result['new_compute_cost'] == 0
    assert result['native_representation_ref'] == first['native_representation_ref']
    assert result['exported_models'] == first['exported_models']
    assert restored.costs == before
    native = load_source_values([result['native_representation_ref']], restored.workspace)[0]['value']
    assert native['comparison_completed']
    assert len(list((restored.workspace / 'native_representation_research').glob('record_*.json'))) == 1


def test_common_program_revision_routes_actual_new_parent_without_representation_conflict(tmp_path, monkeypatch):
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    agent, lab = scene(tmp_path)
    question = declaration()
    question.update(id='revision:new_parent', kind='program_revision',
        origin={'kind': 'frozen_prediction_counterexample', 'row_indices': [48, 49]})
    native_plan = plan(parent_theory_ids=['actual_new_parent'])
    agenda = build_agenda([{**question, 'native_representation_plan': native_plan}], {}, [],
        budget={'compute': 50000, 'author_calls': 0})
    assert agenda['selected_question_id'] == question['id']

    def no_coordinates(*args):
        raise AssertionError('a narrow retained revision changed its declared directions')

    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', no_coordinates)
    result = agent._execute('run', {'operator': 'research_route', 'research_question': question,
        'native_representation_plan': native_plan,
        'theories': [{'id': 'actual_new_parent', 'program': Program('feature', name='u').to_dict()}]}, lab)
    assert result['status'] == 'native_representation_candidates_ready'
    assert result['retained_parent_theory_ids'] == ['actual_new_parent']
    assert result['native_representation']['origin'] == question['origin']
    assert result['native_representation']['research_kind'] == 'program_revision'
    assert result['exported_models'] and not result['parent_revision_completed']
    assert agent.model is None and agent.costs['observation_cost'] == 0


def test_new_representation_evidence_invalidates_previous_narrow_route(tmp_path):
    agent, lab = scene(tmp_path)
    question = {**declaration(), 'id': 'revision:evidence', 'kind': 'program_revision'}
    arguments = {'operator': 'research_route', 'research_question': question,
        'native_representation_plan': plan(parent_theory_ids=['old']),
        'theories': [{'id': 'old', 'program': Program('feature', name='u').to_dict()}]}
    first = agent._execute('run', arguments, lab)
    assert first['native_representation']['alternative_representation_basis'] is None
    stored = next(q for q in agent.questions if q['id'] == question['id'])
    stored['representation_research_basis'] = measured_failure_basis(
        agent, Program('lag', name='u', lag=1))
    second = agent._execute('run', {'operator': 'research_route', 'research_question': stored}, lab)
    assert not second.get('cache_hit')
    assert second['new_compute_cost'] > 0
    assert second['native_representation']['alternative_representation_basis'] == (
        'prospective_failures_with_public_unrepresented_distinction')
    assert second['native_representation']['selected']['validation_mse'] < 1e-10


def test_common_routes_actual_parent_graph_and_reuses_it_without_a_second_replay(tmp_path, monkeypatch):
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    agent, lab = scene(tmp_path)
    parent, reference = retained_procedure(tmp_path)
    question = {**declaration(), 'id': 'revision:actual', 'kind': 'program_revision'}
    calls = []

    def compose(self, spec, rows, budget, parents, actions, *, replayed):
        calls.append(1)
        assert parents == [parent]
        assert replayed[0]['research_program']['source_ref'] == reference
        return replayed

    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', compose)
    arguments = {'operator': 'research_route', 'research_question': question,
        'native_representation_plan': plan(parent_theory_ids=['actual'],
            retained_research_program_refs=[reference]),
        'theories': [{'id': 'actual', 'program': parent.to_dict()}]}
    result = agent._execute('run', arguments, lab)
    assert result['status'] == 'native_representation_candidates_ready'
    assert result['exported_models']
    native = load_source_values([result['native_representation_ref']], agent.workspace)[0]['value']
    assert native['candidates'][0]['research_program']['source_ref'] == reference
    before = deepcopy(agent.costs)
    reused = agent._execute('run', arguments, lab)
    assert reused['cache_hit'] and reused['new_compute_cost'] == 0
    assert agent.costs == before and calls == [1]



def test_resolved_representation_semantics_invalidates_cached_optional_route(tmp_path):
    agent, lab = scene(tmp_path)
    question = {**declaration(), 'id': 'revision:semantics', 'kind': 'program_revision',
        'representation_research_basis': {'semantic_question_ref': {
            'kind': 'question', 'id': 'history_semantics', 'version': 1}}}
    semantics = {'id': 'history_semantics', 'kind': 'representation', 'status': 'open',
        'use_sites': deepcopy(question['use_sites']),
        'answer_resolution_ref': {'evaluator': 'representation_diagnostic_and_paid_comparison',
            'parent_model': 'old.py'}}
    agent.questions.append(semantics)
    first = agent._execute('run', {'operator': 'research_route', 'research_question': question,
        'native_representation_plan': plan(parent_theory_ids=['old']),
        'theories': [{'id': 'old', 'program': Program('feature', name='u').to_dict()}]}, lab)
    assert first['native_representation']['alternative_representation_basis'] == (
        'unresolved_bound_representation_semantics')
    semantics.update(status='completed', semantic_answer={'version': 1, 'status': 'answered'})
    stored = next(item for item in agent.questions if item['id'] == question['id'])
    second = agent._execute('run', {'operator': 'research_route', 'research_question': stored}, lab)
    assert not second.get('cache_hit')
    assert second['native_representation']['alternative_representation_basis'] is None
    assert [item['direction'] for item in second['native_representation']['search_directions']] == ['retained_composition']
