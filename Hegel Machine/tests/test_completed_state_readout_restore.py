"""Completed paid scenes resume with terminal fees, exports and exact readouts."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts import run_state_method_pairing as study
from scripts.run_paid_state_revision import restore_completed_state, restore_exposed_common_state
from test_native_representation_routing import PublicOnlyLab
from test_state_method_pairing import paid_fixture


def completed_fixture(tmp_path, monkeypatch, *, new_panel=None):
    paid, case, values = paid_fixture(tmp_path)
    if new_panel is not None:
        values['public']['control_levels'] = {'u': [index / 40 for index in range(-40, 41)]}
        study.save(tmp_path / 'inputs.json', values)
    registration = study.prepare(paid, tmp_path / 'prepared', case_ref=case,
        evaluation_seed=991, research_compute_cap=10000, compute_cap=30000,
        evaluation_rows=20, approval_ref=paid)

    class Lab(PublicOnlyLab):
        def __init__(self, **_):
            self._past_costs = {'observation_cost': 0, 'measurement_cost': 0, 'queries': 0}

        @property
        def costs(self):
            return self._past_costs

        def public_spec(self):
            return values['public']

        def observation_rows(self, record):
            return [Observation(row['features'], row['target'], tuple(row['history']))
                    for row in record['rows']]

        def restore_paid_account(self, observations, costs):
            self._past_costs = {'observation_cost': costs['observation_cost'],
                'measurement_cost': costs['measurement_cost'], 'queries': len(observations)}
            self.restored_row_count = sum(len(record['rows']) for record in observations)

        def evaluation_rows(self, *, length, seed):
            if new_panel is not None:
                return new_panel
            return [row for record in values['records'] for row in self.observation_rows(record)]

    monkeypatch.setattr(study, 'AnonymousHistoryLab', Lab)
    output = tmp_path / 'completed'
    study.run(registration, output)
    old = read_json_checkpoint(output / '00/workspace/training.json')
    outcome = read_json_checkpoint(output / '00/outcome.json')
    state = read_json_checkpoint(output / '00/workspace/research_state.json')
    assert outcome['costs_after']['compute_cost'] > old['costs']['compute_cost']
    request = {'training_ref': {'file': str(output / '00/workspace/training.json'), 'json_path': []},
        'outcome_ref': {'file': str(output / '00/outcome.json'), 'json_path': []},
        'evaluation_ref': {'file': str(output / '00/frozen_evaluation.json'), 'json_path': []},
        'panel_ref': {'file': str(output / 'evaluation_material.json'), 'json_path': []},
        'finite_native_author_ref': {'file': str(tmp_path / 'prepared/registration.json'),
                                     'json_path': ['finite_native_author']},
        'confirmation_seed': 992}
    return values, Lab, output, old, outcome, state, request


def test_finished_scene_restores_full_pool_and_paid_suffix_without_execution(tmp_path, monkeypatch):
    values, Lab, output, old, outcome, state, request = completed_fixture(tmp_path, monkeypatch)

    def no_science(*args, **kwargs):
        pytest.fail('restoration must not query, fit or execute a frozen predictor')

    monkeypatch.setattr(CommonArchiveInquiryAgent, '_python', no_science)
    monkeypatch.setattr('hegel_machine.cognition.state_search.closed_startup_claims', no_science)
    cfg = replace(InquiryLLMConfig(), max_model_calls=0,
        compute_budget=outcome['costs_after']['compute_cost'] + 30000)
    restored = restore_completed_state(request, tmp_path / 'resumed', Lab(), config=cfg)
    agent, receipt = restored['agent'], restored['restoration']
    assert agent.costs == outcome['costs_after']
    assert restored['lab'].restored_row_count == len(agent.rows)
    assert len(agent.rows) == 2 * len(values['records']) * 5
    assert set(state['exported_candidate_paths']).issubset(restored['models'])
    assert len(restored['models']) == outcome['comparison']['models_completed']
    assert receipt['paid_readout_reuse']['model_executions'] == 0
    assert receipt['paid_readout_reuse']['reused_readout_rows'] == 20
    assert receipt['paid_readout_reuse']['reused_prefix_rows'] == 20
    contract = agent.knowledge['finite_native_author']
    bound = receipt['actual_parent_binding']['theory_id']
    assert bound in contract['native_representation_plan']['parent_theory_ids']
    assert contract['question']['theory_target']['parent_theory_id'] == bound
    assert contract['question']['use_sites'][0]['model'] == old['evaluated_model']['source_model']
    assert contract['native_representation_plan']['validation_records'] == [3]
    assert not contract['native_representation_plan'].get('search_directions')
    assert 'native_representation_research' not in contract['question']
    assert 'routing_execution' not in contract['question']
    suffix = agent.observations[len(values['records']):]
    assert [record['row_indices'][0] for record in suffix] == [20, 25, 30, 35]
    assert all(not record['rows'][0]['history'] for record in suffix)
    before = deepcopy(agent.costs)
    result = agent._execute('run', {'operator': 'research_choice', 'models': [receipt['model']],
        'research_question': {'id': contract['question']['id']}}, restored['lab'])
    assert result['reused_score_count'] == 1 and result['new_scored_count'] == 0
    assert agent.costs['compute_cost'] == before['compute_cost']
    assert read_json_checkpoint(output / '00/workspace/training.json')['costs'] == old['costs']


def test_restored_ordinary_loop_keeps_new_round_and_frontier_after_complete_comparison(tmp_path, monkeypatch):
    from hegel_machine.cognition.temporal_paired_method import execute_procedure
    from hegel_machine.cognition.representation_inquiry import load_native_representation_result

    panel = [Observation({'u': .75}, 0., ()),
             Observation({'u': .8}, 1., ({'u': .75},)),
             Observation({'u': .3}, 1., ({'u': .75}, {'u': .8}))]
    _, Lab, _, _, outcome, _, request = completed_fixture(tmp_path, monkeypatch, new_panel=panel)
    cfg = replace(InquiryLLMConfig(), max_model_calls=0,
        compute_budget=outcome['costs_after']['compute_cost'] + 100000,
        observation_budget=28, confirmation_rows=24)
    scene = restore_completed_state(request, tmp_path / 'resumed', Lab(), config=cfg)
    agent, lab = scene['agent'], scene['lab']
    owner_id = agent.knowledge['finite_native_author']['question']['id']
    owner = next(question for question in agent.questions if question['id'] == owner_id)
    old_rounds = deepcopy(owner['native_representation_research'])
    original_models = set(scene['models'])
    result = execute_procedure(agent, lab, 'ordinary.py', max_steps=3,
        allow_confirmation=False, record_events=True,
        before_step=agent._before_model_turn, after_step=agent._after_model_turn)
    assert result['status'] == 'method_step_limit', result
    actions = [step['proposal']['arguments'].get('operator') for step in result['steps']]
    assert actions == ['research_choice', 'research_route', 'research_choice']
    owner = next(question for question in agent.questions if question['id'] == owner_id)
    assert owner['native_representation_research'][:len(old_rounds)] == old_rounds
    assert len(owner['native_representation_research']) == len(old_rounds) + 1
    new_round = owner['native_representation_research'][-1]
    native = load_native_representation_result(new_round['source_ref'], agent.workspace)
    assert native['experiment_frontier']['predicted_separation'] > 0
    compared = agent.operator_results[-1]
    assert compared['operator'] == 'research_choice'
    assert compared['models_completed'] == compared['models_requested']
    assert original_models <= {entry['path'] for entry in compared['evaluations']}
    for step in result['steps']:
        for declaration in (step['proposal'].get('research_question', {}),
                            step['proposal']['arguments'].get('research_question', {})):
            assert 'native_representation_research' not in declaration
    from hegel_machine.cognition.finite_native_author import select_actions
    from test_native_representation_method_selection import current_memory
    monkeypatch.chdir(agent.workspace)
    next_action = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    assert next_action.get('action') == 'query', next_action
    assert next_action['arguments']['after_observation'][0]['operator'] == 'research_route'


def test_common_counterexamples_restore_full_pool_before_pair_proposals_and_reuse_prefixes(tmp_path, monkeypatch):
    _, Lab, output, old, _, state, original = completed_fixture(tmp_path, monkeypatch)
    old_bytes = (output / '00/workspace/training.json').read_bytes()
    incoming = deepcopy(old['paid_observations'][:2])
    incoming[0]['rows'] = incoming[0]['rows'][:1]
    incoming[0]['action']['sequence'] = incoming[0]['action']['sequence'][:1]
    incoming[1]['rows'] = incoming[1]['rows'][:2]
    incoming[1]['action']['sequence'] = incoming[1]['action']['sequence'][:2]
    material = study.save(tmp_path / 'paid_common.json', incoming)
    costs = {key: 0 for key in old['costs']}
    costs.update(compute_cost=7, observation_cost=3, python_execution_seconds=.1)
    costs_ref = study.save(tmp_path / 'paid_common_costs.json', costs)
    request = {key: original[key] for key in ('training_ref', 'finite_native_author_ref')}
    request.update(restore_confirmation_seed=992,
        append_record_refs=[{**material, 'json_path': [index]} for index in range(2)],
        additional_paid_cost_refs=[costs_ref],
        expected_rows=sum(len(record['rows']) for record in old['paid_observations']) + 3)
    restore = CommonArchiveInquiryAgent.restore_practice_research
    seeds = []

    def restored(self, *args, confirmation_seed):
        seeds.append(confirmation_seed)
        return restore(self, *args, confirmation_seed=confirmation_seed)

    monkeypatch.setattr(CommonArchiveInquiryAgent, 'restore_practice_research', restored)
    cfg = replace(InquiryLLMConfig(), max_model_calls=0,
        compute_budget=old['costs']['compute_cost'] + 100000)
    scene = restore_exposed_common_state(request, tmp_path / 'paid_scene', Lab(), config=cfg)
    agent, receipt = scene['agent'], scene['restoration']
    assert len(agent.rows) == request['expected_rows']
    assert receipt['exposed_rows'] == 3 and receipt['new_observations'] == 0
    assert all(agent.costs[key] == pytest.approx(old['costs'][key] + costs[key]) for key in costs)
    assert receipt['additional_paid_cost_refs'] == [costs_ref]
    assert scene['lab'].restored_row_count == len(agent.rows)
    assert set(state['exported_candidate_paths']) <= set(scene['models'])
    assert len(scene['models']) > 1  # The confirmation model never replaces this bank.
    assert agent.model is None
    question = agent.knowledge['finite_native_author']['question']
    assert 'native_representation_research' not in question
    assert 'routing_execution' not in question
    before = agent.costs['compute_cost']
    selected = agent.evaluated_model['source_model']
    comparison = agent._execute('run', {'operator': 'research_choice', 'models': [selected],
        'research_question': {'id': question['id']}}, scene['lab'])
    score = comparison['evaluations'][0]
    assert score['paid_score_reuse']['reused_prediction_rows'] == request['expected_rows'] - 3
    assert score['paid_score_reuse']['newly_predicted_rows'] == 3
    assert agent.costs['compute_cost'] - before == 3
    agent._prepare_program_state()
    checkpoint = study.save(agent.workspace / 'training.json', agent.checkpoint_result('researching'))
    paid = study.read({'file': str(tmp_path / 'paid.json'), 'json_path': []})
    paid.update(paid_records_ref={**checkpoint, 'json_path': ['paid_observations']},
        append_record_refs=[], expected_rows=len(agent.rows))
    paid_ref = study.save(tmp_path / 'expanded_paid.json', paid)

    def no_fit(*args, **kwargs):
        pytest.fail('proposal preparation must not perform another research fit')

    monkeypatch.setattr('hegel_machine.cognition.state_search.closed_startup_claims', no_fit)
    monkeypatch.setattr('hegel_machine.cognition.state_search.closed_state_claims', no_fit)
    prepared = study.prepare(paid_ref, tmp_path / 'restored_prepared',
        case_ref={'file': str(tmp_path / 'inputs.json'), 'json_path': ['case']},
        evaluation_seed=993, research_compute_cap=10000, compute_cap=100000,
        initial_scene_ref=checkpoint, restore_confirmation_seed=992,
        finite_native_author_ref=original['finite_native_author_ref'])
    assert seeds == [992, 992]
    assert set(prepared['finite_native_author']['predictor_paths']) == set(scene['models'])
    assert prepared['finite_native_author']['question']['theory_target'] == question['theory_target']
    assert 'native_representation_research' not in prepared['question']
    assert 'routing_execution' not in prepared['question']
    assert all(cell['scope']['pool_models'] == len(scene['models'])
        and cell['start_state_ref'] == checkpoint for cell in prepared['cells'].values())
    plan = prepared['cells']['10']['selection_plan']['first_segment']
    assert plan['arguments']['native_representation_plan']['search_directions'] == ['discrete_state']
    assert plan['arguments']['native_representation_plan']['parent_theory_ids'] == (
        prepared['finite_native_author']['native_representation_plan']['parent_theory_ids'])
    assert (output / '00/workspace/training.json').read_bytes() == old_bytes


def test_imported_rows_keep_new_query_and_actual_confirmation_addresses_separate_from_fees(tmp_path):
    from dataclasses import asdict
    from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
    from scripts.run_paid_state_revision import append_records

    laboratory = AnonymousHistoryLab('hysteresis', seed=31)
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        config=replace(InquiryLLMConfig(), compute_budget=10000, observation_budget=10,
                       confirmation_rows=3, max_model_calls=0))
    action = {'kind': 'trajectory', 'sequence': [{'u': -1.}, {'u': -1.}]}
    agent._execute('query', {'actions': [action]}, laboratory)
    # An exposed frozen readout contributes real rows, but its acquisition fee
    # remains at its separate original source rather than this agent's account.
    exposed = laboratory.evaluation_sequences([action['sequence']])
    agent.observations = append_records(agent.observations,
        [{'action': action, 'rows': [asdict(row) for row in exposed]}])
    agent.rows.extend(exposed)
    laboratory.restore_paid_account(agent.observations, agent.costs)
    assert agent.costs['observation_cost'] == laboratory.costs['observation_cost'] == 2
    agent._execute('query', {'actions': [action]}, laboratory)
    assert agent.observations[-1]['row_indices'] == [4, 5]
    assert agent.costs['observation_cost'] == laboratory.costs['observation_cost'] == 4
    agent._save('zero.py', 'def predict(row):\n    return 0.\n')
    finished = agent._execute('finish', {'model': 'zero.py'}, laboratory)
    confirmation = finished['fresh_confirmation']
    assert confirmation['observation_indices'] == [6, 7, 8]
    assert not confirmation['observations'][0]['rows'][0]['history']
    assert agent.costs['observation_cost'] == laboratory.costs['observation_cost'] == 7
    assert len(agent.rows) == 9
