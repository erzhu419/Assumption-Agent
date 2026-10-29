"""The actual Common route reviews native returns without scalar induction."""

import json

from hegel_machine.cognition.common_object_research import (
    CommonObjectDependencyMethods, record_native_parent_research_return,
)
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.question_agenda import has_method_use_plan


def save(path, value):
    path.write_text(json.dumps(value))
    return {'file': str(path)}


def make_common(tmp_path):
    return CommonObjectDependencyMethods(tmp_path / 'common', [],
        budget=SearchBudget(100), actions=[]).common


def retain(common, tmp_path, name, *, source, options=None, mission_compute=30):
    result = save(tmp_path / f'{name}.json', {'status': 'confirmed', 'full_comparison': True,
        'paid_mse': 1e-21, 'compute_cost': 17822})
    decision_ref = save(tmp_path / f'{name}_options.json', options) if options is not None else None
    returned = record_native_parent_research_return(common, source_task_ref=source,
        result_ref=result, comparison_ref={**result, 'json_path': ['full_comparison']},
        decision_options_ref=decision_ref,
        mission={'purpose': 'method_learning', 'allocation': {'compute': mission_compute}})
    return returned


def route(common, allowance=30):
    return common._execute('run', {'operator': 'research_route',
        'research_question': {'id': 'task:method_choice'},
        'allocation': {'compute': allowance}}, None)


def test_native_common_route_never_calls_scalar_and_same_return_reuses_paid_review(tmp_path, monkeypatch):
    from hegel_machine.cognition import method_action_research

    def scalar_must_not_run(*args, **kwargs):
        raise AssertionError('Native object evidence is not a scalar startup witness')

    monkeypatch.setattr(method_action_research, 'review_python_method_returns', scalar_must_not_run)
    common = make_common(tmp_path)
    retain(common, tmp_path, 'first', source={'target': 106029101}, options={
        'default_operation': 'local_structure', 'options': [
            {'operation': 'local_structure'}, {'operation': 'structure_experiment'}]})
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    assert has_method_use_plan(owner)
    first = route(common)
    assert first['status'] == 'complete_native_return_review'
    reviewed = first['method_action_research']
    assert reviewed['reason'] == 'different_native_segments_require_matched_execution'
    assert reviewed['relative_labels_created'] == 0 and reviewed['installed_candidate'] is None
    assert reviewed['matched_development_inputs'][0]['alternative_operation'] == 'structure_experiment'
    assert not first['parent_revision_completed'] and not first['independent_confirmation']
    paid = common.costs['compute_cost']
    repeated = route(common, allowance=10)
    assert repeated['cache_hit'] is True and repeated['new_compute_cost'] == 0
    assert common.costs['compute_cost'] == paid == 2
    assert owner['mission']['resource_usage']['compute'] == 2
    assert not common.theory_consolidation.get('candidates')


def test_two_native_returns_of_one_target_do_not_create_two_sources(tmp_path):
    common = make_common(tmp_path)
    source = save(tmp_path / 'target.json', {'seed': 106029101})
    retain(common, tmp_path, 'before', source=source)
    first = route(common)
    retain(common, tmp_path, 'after', source=source)
    second = route(common)
    reviewed = second['method_action_research']
    assert len(reviewed['ordinary_returns']) == 2
    assert reviewed['distinct_source_refs'] == [source]
    assert second['cache_hit'] is False and second['new_compute_cost'] == 3
    assert first['new_compute_cost'] == 2
    assert reviewed['relative_labels_created'] == 0
    assert reviewed['independent_method_gain'] is None


def test_only_default_stage_is_not_a_new_method_alternative(tmp_path):
    common = make_common(tmp_path)
    retain(common, tmp_path, 'only_confirm', source={'target': 1}, options=[{
        'status': 'ready', 'default_operation': 'confirm',
        'options': [{'operation': 'confirm'}]}])
    result = route(common)
    reviewed = result['method_action_research']
    assert reviewed['matched_development_inputs'] == []
    assert reviewed['reason'] == 'no_new_method_content_from_recorded_native_returns'
    assert reviewed['recorded_decisions'][0]['decision_ref']['json_path'] == [0]
    assert reviewed['next_action'] is None and reviewed['installed_candidate'] is None


def test_unfunded_native_review_remains_pending_until_actual_allocation(tmp_path):
    common = make_common(tmp_path)
    retain(common, tmp_path, 'returned', source={'target': 1}, mission_compute=1)
    insufficient = route(common, allowance=1)
    assert insufficient['status'] == 'scoped_unknown'
    assert common.costs['compute_cost'] == 0
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    assert 'completed_agenda_basis' not in owner
    owner['mission']['allocation']['compute'] = 3
    admitted = route(common, allowance=3)
    assert admitted['status'] == 'complete_native_return_review'
    assert common.costs['compute_cost'] == 2


def test_ordinary_native_choice_survives_return_without_fabricating_other_quotes(tmp_path):
    consumer = CommonObjectDependencyMethods(tmp_path / 'common', [],
        budget=SearchBudget(100), actions=[], source_task_ref={'target': 1})
    delivered = consumer.record_parent_response({'status': 'unresolved', 'actions': [
        {'action': 'fit', 'compute_cost': 50},
        {'action': 'choose_research_action', 'selected_action': 'program_replay',
         'eligible_actions': ['program_replay', 'structural_research'],
         'quoted_compute_cost': 4096, 'reason': 'bounded_trial_before_confirmed_direction'}]})
    common = consumer.common
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    owner['mission']['allocation']['compute'] = 3
    reviewed = route(common)['method_action_research']
    assert reviewed['reason'] == 'retained_native_choices_require_same_state_binding'
    source = reviewed['native_choice_sources'][0]
    assert source['ordinary_operation'] == 'program_replay'
    assert source['alternative_operations'] == ['structural_research']
    assert source['decision_ref']['file'] == delivered['response_ref']['file']
    assert source['decision_ref']['json_path'] == ['decision_options', 0]
    assert reviewed['matched_development_inputs'] == []
    assert reviewed['relative_labels_created'] == 0
    assert reviewed['installed_candidate'] is None
