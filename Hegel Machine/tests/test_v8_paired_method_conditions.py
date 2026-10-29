"""Real paired-action contracts, public state, and delivery denominators."""

from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from hegel_machine.cognition.method_conditions import (
    _method_coordinates, _method_features, _method_observables, _conflict_predictions,
    paired_method_preference, research_paired_method_conditions, same_research_condition,
    method_condition_support,
)
from hegel_machine.cognition.method_feedback import _observe_return, _pre_use_research_state
from hegel_machine.cognition.models import SearchBudget
from test_method_use_conditions import METHOD, profile


def pair(index, *, stage='incomplete', challenger_cost=50, role='development'):
    public = profile()
    public['research'] = {'stage': stage, 'budget': {'remaining_compute': 1000},
                         'source_refs': {'stage': {'file': f'case_{index}/state.json'}}}
    basis = {key: {'version': index} for key in (
        'start_state', 'data', 'split', 'quality_contract', 'resource_contract', 'evaluation')}
    baseline = {'basis': basis, 'profile': public, 'execution_complete': True,
        'status': 'completed', 'completed': True, 'coverage': 24,
        'quality': {'loss': 1e-20, 'threshold': 1e-10},
        'costs': {'observation_cost': 24, 'compute_cost': 100}}
    challenger = deepcopy(baseline)
    challenger['costs']['compute_cost'] = challenger_cost
    return {'feedback_ref': {'file': 'pairs.json', 'json_path': [index]},
        'origin_ref': {'file': 'origins.json', 'json_path': [index]},
        'physical_source_ref': {'file': 'sources.json', 'json_path': [index]},
        'baseline': baseline, 'challenger': challenger, 'data_role': role,
        'baseline_action_content': {'operator': 'theory_transfer'},
        'challenger_action_content': {'operator': 'native_condition_research'}}


def learn(pairs):
    return research_paired_method_conditions(METHOD, pairs, SearchBudget(100000), max_conditions=8)


def test_unknown_number_is_not_zero_and_unavailable_old_numeric_guard_cannot_predict():
    known, unknown = profile(), profile()
    known['research'] = {'budget': {'remaining_compute': 0}}
    unknown['research'] = {'budget': {'remaining_compute': None}}
    views = [_method_observables(value) for value in (known, unknown)]
    coordinates = _method_coordinates(views)
    selected = [c for c in coordinates if c['path'] == ['research', 'budget', 'remaining_compute']]
    assert selected == [{'path': ['research', 'budget', 'remaining_compute'], 'kind': 'available'}]
    assert _method_features(views[0], selected) != _method_features(views[1], selected)
    old = {'coordinate_paths': [{'path': selected[0]['path'], 'kind': 'number'}],
        'candidates': [{'condition': {'op': 'feature', 'name': 'public_0', 'children': []}}]}
    assert _conflict_predictions(old, unknown) == []
    categorical = [{'path': ['research', 'stage'], 'kind': 'equals', 'value': 'search_incomplete'}]
    assert _method_features(views[1], categorical) == {'public_0': None}


def test_identity_future_labels_and_provenance_are_not_strategy_features():
    before = pair(0)['baseline']['profile']
    altered = deepcopy(before)
    altered.update(case='other', kind='private', seed=456, path='/different', future_target=900)
    altered['research']['source_refs'] = {'stage': {'file': '/another/case/state.json'}}
    altered['research']['evidence'] = {'case': 'leak', 'future_target': 900}
    assert _method_observables(before) == _method_observables(altered)
    altered['research']['stage'] = 'complete'
    assert _method_observables(before) != _method_observables(altered)


def test_coverage_is_keyed_by_role_condition_but_ordered_branches_stay_ordered():
    before = profile()
    before['data']['conditional_coverage'] = [
        {'condition': 'startup', 'coverage': [{'role': 0, 'bins': [0, 1]}]},
        {'condition': 'previous_response_1', 'coverage': [{'role': 1, 'bins': [2, 3]}]}]
    after = deepcopy(before)
    after['data']['conditional_coverage'].reverse()
    after['data']['conditional_coverage'][0]['coverage'][0]['bins'].reverse()
    assert _method_observables(before) == _method_observables(after)
    assert same_research_condition(before, after, 'state_transition')
    before['boundary'] = {'branches': [{'history_depth': 0}, {'history_depth': 1}]}
    after = deepcopy(before)
    after['boundary']['branches'].reverse()
    assert _method_observables(before) != _method_observables(after)


def test_unexecuted_alternative_and_mismatched_start_do_not_get_relative_labels():
    item = pair(0)
    item['challenger']['execution_complete'] = False
    assert paired_method_preference(item)['preference'] == 'UNKNOWN'
    item = pair(0)
    item['same_start'] = True
    item['challenger']['basis']['data'] = {'different': True}
    assert paired_method_preference(item)['preference'] == 'UNKNOWN'
    assert learn([item])['new_version_supported'] is False


def test_same_quality_less_observation_is_process_benefit_but_tradeoff_is_unordered():
    item = pair(0, challenger_cost=100)
    item['challenger']['costs']['observation_cost'] = 0
    assert paired_method_preference(item)['preference'] == 'challenger'
    item['challenger']['costs']['compute_cost'] = 101
    assert paired_method_preference(item)['preference'] == 'incomparable'
    item = pair(0, challenger_cost=50)
    item['challenger']['cost_intervals'] = {'compute_cost': [50, 150]}
    assert paired_method_preference(item)['preference'] == 'UNKNOWN'


def test_quality_and_coverage_precede_resource_savings():
    item = pair(0, challenger_cost=0)
    item['challenger']['quality']['loss'] = .1
    assert paired_method_preference(item)['preference'] == 'baseline'
    item = pair(0, challenger_cost=0)
    item['challenger']['coverage'] = 12
    assert paired_method_preference(item)['preference'] == 'baseline'


def test_one_sided_or_all_same_actions_never_publish_condition():
    assert learn([pair(i) for i in range(4)])['condition'] is None
    items = [pair(0), pair(1), pair(2, stage='complete', challenger_cost=200)]
    for item in items:
        item['challenger_action_content'] = deepcopy(item['baseline_action_content'])
    assert learn(items)['status'] == 'no_distinct_executable_action'


def test_mixed_pairs_need_group_check_and_exact_origin_updates_do_not_add_samples():
    items = [pair(0), pair(1), pair(2, stage='complete', challenger_cost=200)]
    result = learn(items)
    assert result['condition'] is None and not result['new_version_supported']
    assert result['status'] == 'development_condition_without_group_support'
    items.extend([pair(3, role='condition_validation'),
        pair(4, stage='complete', challenger_cost=200, role='condition_validation')])
    result = learn(items)
    assert result['new_version_supported'] and result['condition']
    assert len(result['candidates']) <= 8
    assert method_condition_support(items[0]['baseline']['profile'], result)['can_apply']
    outside = deepcopy(items[0]['baseline']['profile'])
    outside['research']['budget']['remaining_compute'] = 1001
    assert not method_condition_support(outside, result)['can_apply']
    assert learn([*items, deepcopy(items[-1])])['paired_invocations'] == 5


def test_condition_validation_cannot_select_a_local_body_or_supply_its_action_witness():
    validation = pair(9, role='condition_validation')
    result = research_paired_method_conditions(METHOD, [validation], SearchBudget(0))
    assert not result['local_program_supported'] and result['local_support_refs'] == []
    assert not result['general_guard_supported'] and result['condition'] is None
    items = [pair(0), pair(1), pair(2, stage='complete', challenger_cost=200)]
    for item in items:
        item['challenger_action_content'] = deepcopy(item['baseline_action_content'])
    result = learn([*items, validation])
    assert result['status'] == 'no_distinct_executable_action'
    assert not result['local_program_supported'] and not result['general_guard_supported']


@pytest.mark.parametrize('status,extra,denominator', [
    ('time_exhausted', {}, 1), ('execution_error', {'execution_error': 'bad contract'}, 1),
    ('administrative_pause', {'administrative_censoring': True}, 0)])
def test_method_non_delivery_and_administrative_censor_preserve_origin_costs(tmp_path, status, extra, denominator):
    agent = SimpleNamespace(workspace=tmp_path)
    group = {}
    origin = {'file': 'paid_use.json', 'json_path': [0]}
    record = {'method_ref': METHOD, 'profile': profile(), 'status': status,
        'execution_complete': False, 'fit_compute_cost': None, 'origin_ref': origin,
        'cost_refs': [{'file': 'old_ledger.json', 'json_path': []}], **extra}
    for _ in range(2):
        _observe_return(agent, group, record, {'file': 'local_copy.json', 'json_path': []})
    result = group['returns']['delivery_reliability']
    assert result['total_invocations'] == 1 and result['method_outcome_denominator'] == denominator
    assert result['world_refutations'] == 0
    assert group['returns']['delivery_outcomes'][0]['cost_refs'] == record['cost_refs']


def test_pre_use_stage_and_quote_are_source_bound_without_unmeasured_evidence_zero(tmp_path):
    owner = {'id': 'q', 'native_condition_research': [{'status': 'search_incomplete',
        'search_stages': [{'stage': 'joint_conditional_response'}],
        'compute_continuation': {'research_unit': {'quoted_paid_rows': 4, 'minimum_compute': 300,
            'completion_reserve': {'retained_pool_models': 0}}}}]}
    agent = SimpleNamespace(workspace=tmp_path, rows=[None] * 4, questions=[owner],
        config=SimpleNamespace(compute_budget=1000, execution_seconds=100),
        costs={'compute_cost': 0, 'python_execution_seconds': 0}, method_results=[],
        candidate_paths=[], knowledge={})
    state = _pre_use_research_state(agent, owner, {})
    assert state['budget']['complete_unit_payable'] is True
    assert state['evidence']['unconsumed_answer'] is None
    assert state['comparison']['stale'] is None
    assert state['source_refs']['native_round']['json_path'][-1] == 0


def test_reference_wrapper_uses_actual_pair_and_retains_only_local_body_support(tmp_path):
    from hegel_machine.cognition.method_feedback import research_paired_method_conditions as read_pairs

    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {'file': str(path), 'json_path': []}

    item = pair(0)
    public = save('before_profile.json', item['baseline']['profile'])
    contracts = {name: save(name + '.json', value) for name, value in item['baseline']['basis'].items()}
    refs = {}
    for side in ('baseline', 'challenger'):
        outcome = deepcopy(item[side])
        outcome.pop('basis'); outcome.pop('profile')
        outcome.update(profile_ref=public, **{name + '_ref': ref for name, ref in contracts.items()})
        refs[side + '_ref'] = save(side + '.json', outcome)
    pair_ref = save('pair.json', {**{key: value for key, value in item.items()
        if key not in {'baseline', 'challenger', 'feedback_ref'}}, **refs})
    agent = SimpleNamespace(workspace=tmp_path, costs={'compute_cost': 0})
    result = read_pairs(agent, METHOD, [pair_ref], 1000)
    assert result['local_program_supported'] is True
    assert result['general_guard_supported'] is False and result['condition'] is None
    assert agent.costs['compute_cost'] == result['new_compute_cost'] == 0


def test_python_carrier_feedback_uses_actual_origin_and_updates_original_method_question(tmp_path):
    from hegel_machine.cognition.method_feedback import (
        begin_method_feedback, record_python_method_feedback, finalize_python_method_feedback, _read)
    from hegel_machine.cognition.models import Observation
    from test_theory_consolidation import agent_scene

    agent, lab = agent_scene(tmp_path)
    lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
        'feature_bounds': {'u': [-1., 1.]}, 'max_probe_length': 2}
    candidate = {'id': 'python-method', 'version': 1, 'kind': 'python_research_method'}
    agent.theory_consolidation.setdefault('candidates', {})[candidate['id']] = candidate
    agent.rows = [Observation({'u': 0.}, 0.)]
    question = agent._open_question({'id': 'public', 'question': 'Research public response',
        'parent_id': 'task', 'context_ref': {'domain': 'public'}, 'answer_type': 'number'})
    agent._active_question_id = question['id']
    context = begin_method_feedback(agent, lab, candidate=candidate)
    assert context['profile']['research']['source_refs']
    origin = {'file': 'original_invocation.json', 'json_path': []}
    binding = {'candidate_id': candidate['id'], 'candidate_version': 1, 'origin_ref': origin}
    outcome = {'status': 'time_exhausted', 'execution_complete': False,
        'cost_refs': [{'file': 'paid_ledger.json', 'json_path': []}]}
    first = record_python_method_feedback(agent, candidate, binding, outcome, context)
    second = record_python_method_feedback(agent, candidate, binding, outcome, context)
    group = agent.theory_consolidation['method_feedback'][candidate['id']]
    assert group['outcome_refs'] == [first['feedback_ref']] == [second['feedback_ref']]
    assert group['returns']['delivery_reliability']['method_outcome_denominator'] == 1
    assert len([q for q in agent.questions if q.get('kind') == 'method_use']) == 1
    assert _read(agent, first['feedback_ref'])['execution_carrier'] == 'python_select_actions'
    assert 'support' not in candidate and candidate['version'] == 1
    whole = tmp_path / 'actual_whole_result.json'
    whole.write_text(json.dumps({'status': 'completed', 'execution_complete': True,
        'comparison': {'models_requested': 1, 'models_completed': 1},
        'cost_refs': [{'file': 'paid_ledger.json', 'json_path': []}]}))
    finalized = finalize_python_method_feedback(agent, candidate, binding, context,
        result_ref={'file': str(whole), 'json_path': []})
    assert finalized['feedback_ref'] == first['feedback_ref']
    record = _read(agent, first['feedback_ref'])
    assert record['execution_complete'] is True and 'working_comparison' not in record
    assert group['returns']['delivery_reliability']['delivered'] == 1
    assert group['returns']['delivery_reliability']['total_invocations'] == 1


def test_python_ordinary_return_reopens_original_and_published_topics_without_pair_labels(tmp_path):
    from hegel_machine.cognition.method_feedback import (
        begin_method_feedback, record_python_method_feedback, finalize_python_method_feedback, _read)
    from hegel_machine.cognition.models import Observation
    from test_theory_consolidation import agent_scene

    agent, lab = agent_scene(tmp_path)
    lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
        'feature_bounds': {'u': [-1., 1.]}, 'max_probe_length': 2}
    candidate = {'id': 'python-return', 'version': 1, 'kind': 'python_research_method'}
    agent.theory_consolidation.setdefault('candidates', {})[candidate['id']] = candidate
    agent.rows = [Observation({'u': 0.}, 0.)]
    public = agent._open_question({'id': 'public', 'question': 'Research public response', 'parent_id': 'task'})
    origin = {'file': 'original_invocation.json', 'json_path': []}
    binding = {'candidate_id': candidate['id'], 'candidate_version': 1, 'origin_ref': origin}
    prior_pair = {'file': 'old_actual_pair.json', 'json_path': []}
    owner = agent._open_question({'id': 'original-method-question', 'kind': 'method_use',
        'question': 'Which method should this parent use?', 'parent_id': public['id'],
        'method_use_plan': {'candidate_id': candidate['id'], 'binding': deepcopy(binding),
            'action_revision': {'pair_refs': [prior_pair]}}})
    candidate['method_question_ref'] = {'question_id': owner['id'], 'meaning_version': 1}
    agent._active_question_id = public['id']
    context = begin_method_feedback(agent, lab, candidate=candidate)
    owner.update(status='supported', completed_agenda_basis={'old': 'answered'})
    first = record_python_method_feedback(agent, candidate, binding,
        {'status': 'time_exhausted', 'execution_complete': False}, context)
    assert owner['status'] == 'open' and 'completed_agenda_basis' not in owner
    before = deepcopy(owner['method_use_plan']['ordinary_return_basis'])
    repeated = record_python_method_feedback(agent, candidate, binding,
        {'status': 'time_exhausted', 'execution_complete': False}, context)
    assert repeated['feedback_ref'] == first['feedback_ref']
    assert owner['method_use_plan']['ordinary_return_basis'] == before
    whole = tmp_path / 'whole.json'; whole.write_text(json.dumps({
        'status': 'completed', 'execution_complete': True, 'comparison': {
            'models_requested': 1, 'models_completed': 1, 'truncated': False}}))
    final = finalize_python_method_feedback(agent, candidate, binding, context,
        result_ref={'file': str(whole), 'json_path': []})
    assert final['feedback_ref'] == first['feedback_ref']
    plan = owner['method_use_plan']
    assert plan['ordinary_return_refs'] == [first['feedback_ref']]
    assert plan['ordinary_return_basis'] != before
    assert plan['ordinary_return_basis'][0]['execution_complete'] is True
    assert plan['action_revision']['pair_refs'] == [prior_pair]
    for reference in final['method_question_refs']:
        published = next(question for question in agent.questions if question['id'] == reference['question_id'])
        assert published['origin_method_question_ref'] == candidate['method_question_ref']
        assert published['method_use_plan']['ordinary_return_refs'] == [first['feedback_ref']]
        assert published['method_use_plan']['ordinary_return_basis'][0]['execution_complete'] is True
    record = _read(agent, first['feedback_ref'])
    assert 'working_comparison' not in record and 'paired_future' not in record
    assert candidate['version'] == 1


def test_paid_answer_analysis_state_reads_actual_callback_completion_without_new_query(tmp_path):
    callback = {'operator': 'native_condition_research', 'question': {'id': 'q'},
        'boundary_update': {'round_index': 0, 'proposal_index': 0}}
    commands = [{'kind': 'trajectory', 'sequence': [{'u': .5}]}]
    owner = {'id': 'q', 'context_ref': {'domain': 'public'}, 'native_condition_research': [{
        'status': 'search_incomplete', 'proposed_experiments': [{'predicted_separation': 1.,
            'query_action': {'action': 'query', 'arguments': {'actions': commands,
                'question': {'id': 'q'}, 'after_observation': [callback]}}}]}]}
    paid = [{'action': commands[0], 'rows': [{'features': {'u': .5}, 'history': [], 'target': 0.}]}]
    agent = SimpleNamespace(workspace=tmp_path, rows=[None], observations=paid,
        questions=[owner], method_results=[], candidate_paths=[],
        config=SimpleNamespace(compute_budget=1000, execution_seconds=100),
        costs={'compute_cost': 0, 'python_execution_seconds': 0},
        knowledge={'finite_native_author': {'question': {'id': 'q', 'context_ref': owner['context_ref']}}})
    public = {'controls': ['u'], 'reset': 'Every trajectory starts from reset.', 'max_probe_length': 1,
        'feature_bounds': {'u': [-1., 1.]}, 'control_levels': {'u': [-1., 0., .5, 1.]}}
    before = _pre_use_research_state(agent, owner, {}, public)
    assert before['evidence']['unconsumed_answer'] is True
    assert before['evidence']['distinguishing_experiment_available'] is True
    agent.method_results.append({'feedback': {'after_observation_results': [{
        **callback, 'arguments': callback,
        'research_question_id': 'q', 'question_meaning_version': 1,
        'native_condition_research': {'boundary_update': {'status': 'boundary_refined',
            'observation_index': 0}}}]}})
    after = _pre_use_research_state(agent, owner, {}, public)
    assert after['evidence']['unconsumed_answer'] is False
    assert agent.observations == paid


def test_public_pair_distinction_reads_the_actual_nested_proposal_shape_and_keeps_unknown(tmp_path):
    commands = [{'kind': 'trajectory', 'sequence': [{'u': value}] * 12} for value in (-.4, .4)]
    pair = {'reference_action_index': 0, 'action_index': 1, 'predicted_separation': 1.}
    owner = {'id': 'q', 'native_condition_research': [{'status': 'candidates_require_evaluation',
        'proposed_experiments': [{'matched_action_pairs': [pair],
            'query_action': {'action': 'query', 'arguments': {'actions': commands}}}]}]}
    agent = SimpleNamespace(workspace=tmp_path, rows=[None] * 145, observations=[],
        questions=[owner], method_results=[], candidate_paths=[],
        config=SimpleNamespace(compute_budget=2000000, execution_seconds=1800),
        costs={'compute_cost': 0, 'python_execution_seconds': 0}, knowledge={})
    public = {'controls': ['u'], 'reset': 'Each trajectory starts from reset.', 'max_probe_length': 24,
        'feature_bounds': {'u': [-1., 1.]}, 'control_levels': {'u': [-.4, .4]}}
    assert _pre_use_research_state(agent, owner, {}, public)['evidence']['distinguishing_experiment_available'] is True
    pair['predicted_separation'] = None
    assert _pre_use_research_state(agent, owner, {}, public)['evidence']['distinguishing_experiment_available'] is None
    pair['predicted_separation'] = 0.
    assert _pre_use_research_state(agent, owner, {}, public)['evidence']['distinguishing_experiment_available'] is False
    assert agent.observations == [] and agent.costs['compute_cost'] == 0
