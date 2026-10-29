"""A registered procedure reaches its action after one bounded public visit."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition import common_archive_inquiry, finite_native_author
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from hegel_machine.cognition.language import Program
from scripts import run_method_action_pairing as runner
from test_finite_choice_reuse import choice_scene
from test_inquiry_revision import action


def controlled_scene(tmp_path, monkeypatch):
    agent, lab = choice_scene(tmp_path)
    owner = agent._open_question(agent.knowledge['finite_native_author']['question'])
    agent._active_question_id = owner['id']
    calls = {'route': 0, 'export': 0, 'consolidation': 0, 'bindings': 0}
    monkeypatch.setattr(common_archive_inquiry.CommonArchiveInquiryAgent,
        '_before_model_turn', lambda *_: False)

    def bind(*_):
        calls['bindings'] += 1
    monkeypatch.setattr(agent, '_bind_revision_target', bind)

    def consolidate(*_):
        calls['consolidation'] += 1
        return False
    monkeypatch.setattr(agent, '_consolidate_registered_source', consolidate)

    def route(arguments, laboratory, kind):
        calls['route'] += 1
        calls['route_ceiling'] = agent.config.compute_budget
        agent.events.append({'feedback': {'operator': 'research_route', 'status': 'bounded_fixture_visit'}})
        agent._organization_state[kind] = {'event_index': len(agent.events) - 1}
        return True
    monkeypatch.setattr(agent, '_automatic_work', route)

    def export(actual, question):
        assert actual is agent
        calls['export'] += 1
        path = f'public_{calls["export"]}.py'
        agent._save(path, export_temporal_program(Program('const', value=0.), feature_names=('u',)))
        agent.candidate_paths.append(path)
        return {'new_exports': 1, 'exports': [{'model': path}]}
    monkeypatch.setattr(finite_native_author, 'export_revision_candidates', export)
    return agent, lab, owner, calls


def compare(agent, lab, owner):
    models = list(dict.fromkeys([*agent.knowledge['finite_native_author']['predictor_paths'],
                                *agent.candidate_paths]))
    result = agent._execute('run', {'operator': 'research_choice', 'models': models,
        'research_question': {'id': owner['id']}}, lab)
    assert result['models_completed'] == result['models_requested'] == len(models)
    agent.method_results.append({'path': 'engineering_compare',
        'proposal': {'action': 'run', 'arguments': {'operator': 'research_choice',
            'models': models, 'research_question': {'id': owner['id']}}},
        'feedback': deepcopy(result)})
    return result


@pytest.mark.parametrize('control', [3., 4.])
def test_real_hooks_reach_distinct_query_callback_and_complete_strong_pool(
        tmp_path, monkeypatch, control):
    agent, lab, owner, calls = controlled_scene(tmp_path, monkeypatch)
    callback_rows = []

    def native(actual, arguments, laboratory, question):
        callback_rows.append(len(actual.rows))
        path = 'native_after.py'
        actual._save(path, export_temporal_program(Program('const', value=0.), feature_names=('u',)))
        actual.candidate_paths.append(path)
        round_ = {'status': 'candidates_require_evaluation', 'candidates': [],
            'round_index': len(question.get('native_condition_research', [])),
            'fit_evidence': list(range(len(actual.rows))),
            'exported_models': [{'model': path}], 'observed_record_count': len(actual.observations),
            'fit_record_indices': list(range(len(actual.observations)))}
        question.setdefault('native_condition_research', []).append(round_)
        return {'operator': 'native_condition_research', 'native_condition_research': round_}
    monkeypatch.setattr(common_archive_inquiry, 'execute_native_condition_research', native)
    source = '''import json
from pathlib import Path
def select_actions(public_spec, observations, memory):
    state = json.loads(Path('research_state.json').read_text())
    paths = list(dict.fromkeys(memory['finite_native_author']['predictor_paths'] + state['exported_candidate_paths']))
    rows = sum(len(record.get('rows', [])) for record in observations)
    complete = any(result.get('operator') == 'research_choice'
        and result.get('paid_rows') == rows and not result.get('truncated')
        and result.get('models_requested') == result.get('models_completed') == len(paths)
        and [item['path'] for item in result.get('evaluations', [])] == paths
        for result in state['operator_results'])
    if not complete:
        return {'action':'run', 'arguments':{'operator':'research_choice','models':paths,
            'research_question':{'id':'source_native'}}}
    return QUERY
'''.replace('QUERY', repr({'action': 'query', 'arguments': {'actions': [action(control)],
        'research_question': {'id': owner['id']}, 'after_observation': [
            {'operator': 'native_condition_research', 'research_question': {'id': owner['id']}}]}}))
    agent._save('registered.py', source)
    result = runner.execute_registered_method_unit(agent, lab, 'registered.py',
        before_step=agent._before_model_turn, after_step=agent._after_model_turn,
        native_attempt_limit=1, max_steps=6)
    assert calls['route'] == calls['export'] == calls['consolidation'] == 1
    assert calls['bindings'] == 2 and callback_rows == [3]
    assert result['status'] == 'registered_native_attempt_returned'
    assert len(result['actual_action_content']) == 1
    assert result['actual_action_content'][0]['arguments']['actions'] == [action(control)]
    assert agent.method_results[0]['proposal']['arguments']['operator'] == 'research_choice'
    assert agent.method_results[0]['feedback']['models_completed'] == 3
    assert result['registered_research_unit']['phase'] == 'research'
    assert result['registered_research_unit']['status'] == 'research_returned'
    assert result['registered_research_unit']['preparation']['whole_library_search_completed'] is False
    closed = compare(agent, lab, owner)
    assert closed['models_completed'] == 4
    frozen = runner._freeze(agent, agent._evaluated_summary()['model'], tmp_path / 'frozen')
    assert (tmp_path / 'frozen/_evaluated_model' / frozen['model']).is_file()
    assert set(agent.knowledge['finite_native_author']['predictor_paths']).issubset(
        entry['path'] for entry in closed['evaluations'])


@pytest.mark.parametrize('field', ['meaning_version', 'context_ref', 'role_signature', 'output_unit',
                                  'parent_theory_id'])
def test_changed_meaning_or_binding_stops_the_old_unit_before_new_science(tmp_path, monkeypatch, field):
    agent, lab, owner, calls = controlled_scene(tmp_path, monkeypatch)
    unit = agent._begin_registered_method_unit('method.py', {'file': 'state', 'json_path': ['method_results', 0]})
    agent._before_model_turn(lab)
    compare(agent, lab, owner)
    agent._before_model_turn(lab)
    assert unit['phase'] == 'research' and unit['preparation']['completion_valid_for_current_scope']
    if field == 'meaning_version':
        owner[field] = owner.get(field, 1) + 1
    else:
        owner['theory_target'][field] = ('different' if field in {'output_unit', 'parent_theory_id'}
                                         else {'changed': True})
    before = deepcopy(agent.costs)
    with pytest.raises(ValueError, match='new preparation and quote'):
        agent._before_model_turn(lab)
    assert unit['phase'] == 'requires_rebind'
    assert not unit['preparation']['completion_valid_for_current_scope']
    assert agent.costs == before and calls['route'] == 1


def test_new_evidence_keeps_the_callback_stage_but_invalidates_old_comparison(tmp_path, monkeypatch):
    agent, lab, owner, calls = controlled_scene(tmp_path, monkeypatch)
    unit = agent._begin_registered_method_unit('method.py', {'file': 'state', 'json_path': ['method_results', 0]})
    agent._before_model_turn(lab)
    compare(agent, lab, owner)
    agent._before_model_turn(lab)
    agent._execute('query', {'actions': [action(3.)], 'research_question': {'id': owner['id']}}, lab)
    agent._before_model_turn(lab)
    assert unit['phase'] == 'research' and calls['route'] == calls['export'] == 1
    assert not unit['preparation']['completion_valid_for_current_scope']
    assert unit['dependency_updates'][-1]['after']['rows'] == 3
    compare(agent, lab, owner)
    agent._after_model_turn(lab)
    assert unit['preparation']['completion_valid_for_current_scope']
    assert unit['preparation']['compared_rows'] == 3 and calls['route'] == 1


def test_public_visit_preserves_planned_research_and_complete_closure_budget(tmp_path, monkeypatch):
    agent, lab, owner, calls = controlled_scene(tmp_path, monkeypatch)
    agent.config = replace(agent.config, compute_budget=1000)
    start = agent.costs['compute_cost']
    agent._unit_research_compute_ceiling = start + 600
    unit = agent._begin_registered_method_unit('method.py', {'file': 'state'}, research_compute_reserve=400)
    agent._before_model_turn(lab)
    assert calls['route_ceiling'] == start + 200
    assert unit['research_compute_reserved'] == 400
    assert agent.config.compute_budget == 1000


def test_direct_paid_comparison_without_question_header_reaches_registered_choice(tmp_path, monkeypatch):
    agent, lab, owner, calls = controlled_scene(tmp_path, monkeypatch)
    unit = agent._begin_registered_method_unit('method.py', {'file': 'state'})
    agent._before_model_turn(lab)
    paths = [*agent.knowledge['finite_native_author']['predictor_paths'], *agent.candidate_paths]
    agent._execute('run', {'operator': 'research_choice', 'models': paths,
        'research_question': {'id': owner['id']}}, lab)
    report = agent.operator_results[-1]
    report.pop('research_question_id', None)
    report.pop('question_meaning_version', None)
    assert not agent.method_results
    agent._before_model_turn(lab)
    assert unit['phase'] == 'research'
    assert unit['preparation']['completion_valid_for_current_scope']
    assert unit['preparation']['complete_comparison_ref']['json_path'] == [
        'operator_results', len(agent.operator_results) - 1]
    assert report.get('research_question_id') is None
    assert calls['route'] == calls['export'] == 1


def test_same_origin_recovery_retains_stage_new_origin_reopens_public_visit(tmp_path, monkeypatch):
    agent, lab, owner, calls = controlled_scene(tmp_path, monkeypatch)
    origin = {'file': 'state', 'json_path': ['method_results', 0]}
    unit = agent._begin_registered_method_unit('method.py', origin)
    agent._before_model_turn(lab)
    compare(agent, lab, owner)
    agent._before_model_turn(lab)
    agent._end_registered_method_unit(unit, interrupted=True)
    saved = json.loads(json.dumps(agent.checkpoint_result('researching')['research_organization']))
    agent._organization_state = saved
    restored = agent._begin_registered_method_unit('method.py', origin)
    agent._before_model_turn(lab)
    assert restored['phase'] == 'research' and calls['route'] == 1
    fresh = agent._begin_registered_method_unit('method.py', {'file': 'state', 'json_path': ['method_results', 1]})
    agent._before_model_turn(lab)
    assert calls['route'] == 2 and fresh['origin_ref'] != origin
