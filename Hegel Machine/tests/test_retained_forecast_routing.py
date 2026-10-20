"""Real Common routes over retained forecast refs, keeping exact ties and scope."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_continuation import pending_query_arguments
from hegel_machine.cognition.question_routing import resolved_theories
from hegel_machine.cognition import inquiry_revision


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def replay_module():
    specification = importlib.util.spec_from_file_location('retained_forecast_routing',
        ROOT / 'scripts/replay_retained_forecast_routing.py')
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / 'scripts'))
        specification.loader.exec_module(module)
    if not (module.SOURCE / 'research_state.json').exists():
        pytest.skip('The retained historical target_000 is not mounted')
    return module


@pytest.fixture(scope='module')
def actual_route(replay_module, tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError('Saved forecast routing attempted a Python model execution or fit')
        patch.setattr(inquiry_revision, 'evaluate_python_model', forbidden)
        patch.setattr(inquiry_revision, 'retain_evaluated_python_model', forbidden)
        return replay_module.replay(output=tmp_path_factory.mktemp('retained_forecast_route') / 'receipt.json')


def test_real_ref_only_candidates_reach_Common_and_FindDistinguishing_with_same_complete_panel_tie(actual_route):
    receipt = actual_route
    assert receipt['status'] == 'completed'
    assert all('program' not in item and 'forecast_ref' in item for item in receipt['registered_candidates'])
    assert all(item['source_status'] == 'retained_forecast_ready' for item in receipt['resolved_candidates'])
    assert all(item['support'] == {'status': 'candidate', 'evidence': []}
               for item in receipt['resolved_candidates'])
    selection = receipt['route']['selection']
    assert receipt['route']['status'] == selection['status'] == 'tied'
    assert len(selection['groups']) == 2
    assert {row['G'] for row in selection['scores']} == {1.}
    assert {row['effective_pairs'] for row in selection['scores']} == {1}
    assert all(row['declared_cost'] == [48, 0, 0] for row in selection['scores'])
    assert selection['tied_experiments'] == ['alternating_and_hold', 'switches_and_reversals']
    assert selection['selected_experiment_id'] is None and receipt['route']['parent_use'] is None
    assert receipt['find_distinguishing']['status'] == receipt['find_distinguishing']['query_answer']['status'] == 'tied'
    assert receipt['find_distinguishing']['same_scores_as_research_route']
    assert receipt['find_distinguishing']['selection']['groups'] == selection['groups']
    assert receipt['repeat'] == {'cache_hit': True, 'new_compute_cost': 0, 'costs_unchanged': True}
    assert all(panel['trajectory_lengths'] == [24, 24] and panel['observation_cost'] == 48
               for panel in receipt['protocol']['panels'])
    assert receipt['protocol']['endpoint_step'] == 23
    assert receipt['protocol']['original_mse_threshold'] == 1e-10
    assert receipt['protocol']['answer_resolution_atol'] == 1e-6


def test_only_paid_frozen_consequences_are_checked_without_creating_support_or_copying_source(actual_route):
    check = actual_route['historical_consequence_check']
    assert check['result']['status'] == 'answered'
    assert check['result']['evidence'] == [95, 119]
    assert check['result']['value'] == -3.8406849916050874
    assert [row['frozen_value'] for row in check['frozen_consequences']] == [-.6260315476690146, 22.043474837665464]
    assert all(row['within_declared_resolution'] is False for row in check['frozen_consequences'])
    assert check['support_updated'] is False and check['independent_confirmation'] is False
    assert check['unexecuted_panel'] == {'id': 'switches_and_reversals', 'status': 'UNKNOWN', 'record_indices': []}
    assert actual_route['costs']['Common_route'] == actual_route['costs']['Common_FindDistinguishingExperiment'] == 8
    assert actual_route['costs']['historical_two_readout_check'] == 2
    assert actual_route['costs']['total_new_compute'] == 18
    assert actual_route['costs']['original_account']['compute_cost'] == 1152
    assert actual_route['costs']['new_acquisition_cost'] == 0
    assert actual_route['storage'] == {'scientific_Programs_in_index': False,
        'forecast_columns_copied_to_index_or_library': False,
        'source_file_copied': False, 'source_access': 'absolute_forecast_ref'}


def test_complete_48_coordinate_query_preserves_the_same_tie_without_copying_vectors(replay_module, tmp_path):
    receipt = replay_module.replay(output=tmp_path / 'vector.json', readout='full_panel_vector')
    selection = receipt['route']['selection']
    assert selection['status'] == 'tied' and selection['selected_experiment_id'] is None
    assert selection['tied_experiments'] == ['alternating_and_hold', 'switches_and_reversals']
    comparison = selection['tie_diagnostics']['comparisons'][0]
    assert comparison['same_task_relevant_coverage'] and comparison['same_pairwise_separation']
    assert comparison['same_exact_prediction_partition'] and not comparison['same_candidate_predictions']
    assert receipt['find_distinguishing']['same_scores_as_research_route']
    assert receipt['protocol']['query_contract']['answer_type'] == 'vector'
    check = receipt['historical_consequence_check']
    assert check['result']['scope']['case_count'] == check['result']['scope']['answered_case_count'] == 48
    assert check['result']['evidence'] == list(range(72, 120))
    assert check['result']['value']['retained_values_ref']['record_indices'] == [24, 25]
    assert all(not row['within_declared_resolution'] for row in check['frozen_consequences'])
    assert check['unexecuted_panel']['status'] == 'UNKNOWN' and not check['support_updated']
    assert receipt['costs']['Common_route'] == receipt['costs']['Common_FindDistinguishingExperiment'] == 192
    assert receipt['costs']['historical_finite_readout_check'] == 48
    assert receipt['costs']['total_new_compute'] == 432
    assert receipt['storage']['finite_vectors_stored_by_reference']
    assert all('retained_values_ref' in group['task_answer'] for group in selection['groups'])


def test_deleting_owned_small_forecast_source_blocks_an_existing_pending_plan_before_reroute(
        replay_module, tmp_path):
    original = replay_module.SOURCE / 'research_state.json'
    state = json.loads(original.read_text())
    declarations, panel, question = replay_module.build_inputs(state, original)
    # This dependency fixture has one legal plan. The historical comparison
    # above keeps both original panels and its tie; no gain is claimed here.
    columns = []
    for declaration in declarations:
        forecast = state['experiment_comparisons'][0]['forecasts'][declaration['forecast_ref']['json_path'][-1]]
        columns.append({'model': forecast['model'], 'returncode': 0,
            'prediction_inputs': [deepcopy(forecast['prediction_inputs'][index]) for index in (23, 47)],
            'predictions': [forecast['predictions'][index] for index in (23, 47)]})
    fixture_file = tmp_path / 'owned_endpoint_forecasts.json'
    fixture_file.write_text(json.dumps({'forecasts': columns}))
    assert fixture_file.stat().st_size < 5000
    for index, declaration in enumerate(declarations):
        declaration['forecast_ref'].update(file=str(fixture_file), json_path=['forecasts', index])
    agent = CommonArchiveInquiryAgent(None, tmp_path / 'workspace', arm='hegel', source_task=False,
        config=InquiryLLMConfig(compute_budget=250000, observation_budget=360,
            max_model_calls=0, execution_seconds=30.))
    parent = agent._open_question({'id': 'dependency_fixture_parent', 'question': 'Prepare the declared legal plan'})
    question['parent_id'] = parent['id']
    question['use_sites'][0]['parent_id'] = parent['id']
    lab = replay_module.PublicReplayLab()
    first = agent._execute('run', {'operator': 'research_route', 'research_question': question,
        'theories': declarations, 'experiments': panel[:1],
        'mission': {'purpose': 'task', 'mission_id': 'source_dependency_fixture'}}, lab)
    assert first['status'] == 'selected' and first['parent_use']['status'] == 'parent_query_prepared'
    assert pending_query_arguments(agent, parent['id'])['actions'] == panel[0]['actions']
    fixture_file.unlink()
    with pytest.raises(ValueError, match='needs rebuilding'):
        pending_query_arguments(agent, parent['id'])
    second = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}}, lab)
    assert second['status'] == 'scoped_unknown' and not second['cache_hit']
    assert all(item['source_status'] == 'unavailable_retained_forecast' for item in resolved_theories(agent))
    with pytest.raises(ValueError, match='needs rebuilding'):
        pending_query_arguments(agent, parent['id'])
    persisted = json.dumps(agent.theory_index) + json.dumps(agent.theory_consolidation)
    assert '"program"' not in persisted and '"prediction_inputs"' not in persisted
    assert '"_retained_forecast"' not in persisted
    assert not (agent.workspace / 'research_state.json').read_text().count('"prediction_inputs"')
    assert lab.execute_calls == 0 and agent.costs['model_calls'] == agent.costs['observation_cost'] == 0
