"""Paid stage evidence forms executable proposals without future-label leakage."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.common_object_research import record_native_parent_research_return
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.method_conditions import method_condition_features
from hegel_machine.cognition.staged_method_research import baseline_return_profile, research_staged_method_conditions
from test_native_paired_method_feedback import common, route, save
from test_transfer_partitions import plan


def fixture(tmp_path):
    agent = common(tmp_path)
    bodies = {'baseline': plan(dormant=True), 'challenger': plan(.85, weight=3., dormant=True)}
    bodies['baseline']['transfer_partition'] = {
        'kind': 'Pi0', 'max_attempts': 1, 'max_candidates': 4}
    bodies['challenger']['transfer_partition'] = {
        'kind': 'Pi1', 'released_groups': ['r'], 'max_attempts': 48, 'max_candidates': 4}
    artifacts = {}
    for side, body in bodies.items():
        program = save(tmp_path / (side + '_program.json'), body)
        artifacts[side] = save(tmp_path / (side + '_method.json'), {
            'id': side, 'version': 1, 'kind': 'native_research_segment',
            'execution': {'entry': 'replay_research_program', 'research_program_ref': program}})
    refs = []
    for i, (loss, role) in enumerate(((.4, 'development'), (.2, 'development'),
                                    (0., 'condition_validation'))):
        root = tmp_path / ('source' + str(i))
        contract = {key + '_ref': save(root / (key + '.json'), {'source': i})
            for key in ('start_state', 'data', 'split', 'evaluation')}
        contract['quality_contract_ref'] = save(root / 'quality.json', {'threshold': 1e-7})
        contract['resource_contract_ref'] = save(root / 'resource.json', {'compute_limit': 10000})
        profile = save(root / 'profile.json', {'residual': {'loss': 99.}, 'case': 'private'})
        outcomes = {}
        for side, score, cost in (('baseline', loss, 30), ('challenger', 0., 100)):
            # The old-pool selected loss deliberately disagrees with the new
            # P0 output. Only the addressed new score may train the next stage.
            fitted = save(root / side / 'new_entry.json', {'research_program': bodies[side]})
            save(root / side / 'comparison.json', {'full_comparison': True,
                'new_scores': [{'candidate_ref': fitted, 'loss': score}]})
            actual = save(root / side / 'outcome.json', {'execution_complete': True,
                'full_comparison': True, 'new_candidates': 1, 'selected_loss': 0., 'development_rows': 24})
            outcomes[side] = save(root / (side + '.json'), {**contract,
                'profile_ref': profile, 'method_ref': {'id': side, 'version': 1},
                'method_artifact_ref': artifacts[side], 'actual_outcome_ref': actual,
                'execution_complete': True, 'status': 'completed', 'completed': True,
                'coverage': 24, 'loss': 0.,
                'costs': {'compute_cost': cost, 'observation_cost': 0, 'measurement_cost': 0}})
        pair_ref = save(root / 'pair.json', {
            'baseline_ref': outcomes['baseline'], 'challenger_ref': outcomes['challenger'],
            'physical_source_ref': {'file': str(root / 'registration.json')},
            'origin_ref': {'file': str(root / 'registration.json')}, 'data_role': role,
            'baseline_action_content': {'method': 'baseline'}, 'challenger_action_content': {'method': 'challenger'}})
        result = save(root / 'result.json', {'status': 'paired_completed',
            'method_pair_ref': pair_ref, 'method_ref': {'id': 'challenger', 'version': 1},
            'method_artifact_ref': artifacts['challenger']})
        record_native_parent_research_return(agent, source_task_ref={'file': str(root / 'registration.json')},
            result_ref=result, mission={'purpose': 'method_learning', 'allocation': {'compute': 100000}})
        refs.append(pair_ref)
    group = {'method_ref': {'id': 'challenger', 'version': 1},
        'baseline_method_artifact_ref': artifacts['baseline'], 'method_artifact_ref': artifacts['challenger'],
        'pair_refs': refs}
    return agent, group, bodies


def test_common_route_learns_post_P0_guard_from_own_fit_not_old_pool_winner(tmp_path):
    agent, group, bodies = fixture(tmp_path)
    # Although Pi1 is cheaper from zero here, after P0 it adds positive cost
    # without improving quality. That next-stage label must favor stopping.
    last = read_method_reference(group['pair_refs'][-1], agent.workspace)
    path = Path(last['challenger_ref']['file'])
    cheaper = json.loads(path.read_text())
    cheaper['costs']['compute_cost'] = 1
    path.write_text(json.dumps(cheaper))
    reviewed = route(agent)['method_action_research']
    assert reviewed['status'] == 'method_content_proposed_pending_validation'
    assert reviewed['paired_condition_research'][0]['candidates'] == []  # Old conditions retained.
    result = reviewed['staged_method_proposals'][0]
    assert result['status'] == 'staged_program_proposed'
    assert result['new_fits'] == result['new_observations'] == 0
    assert not result['general_guard_supported'] and not result['independent_support']
    bundle = read_method_reference(result['proposal_ref'], agent.workspace)
    assert [item['profile']['residual']['loss'] for item in bundle['baseline_evidence']] == [.4, .2, 0.]
    assert [item['preference'] for item in bundle['research']['preferences']] == ['challenger', 'challenger', 'baseline']
    assert bundle['baseline_evidence'][2]['previous_data_role'] == 'condition_validation'
    assert all(item['data_role'] == 'development' for item in bundle['baseline_evidence'])
    assert bundle['research']['group_validation_refs'] == []
    for i, item in enumerate(bundle['baseline_evidence']):
        assert item['stage_costs']['shared_paid_prefix']['compute_cost'] == 30
        assert item['stage_costs']['baseline_increment']['compute_cost'] == 0
        assert item['stage_costs']['challenger_increment']['compute_cost'] == (1 if i == 2 else 100)
    body = read_method_reference(result['research_program_refs'][0], agent.workspace)
    assert {key: value for key, value in body.items() if key != 'staged_research'} == bodies['baseline']
    stage = body['staged_research']
    assert stage['baseline'] == bodies['baseline'] and stage['continuation'] == bodies['challenger']
    assert all(coordinate['path'][0] in {'residual', 'research'} for coordinate in stage['coordinate_paths'])
    paid = agent.costs['compute_cost']
    owner = next(q for q in agent.questions if q['id'] == 'task:method_choice')
    assert 'completed_agenda_basis' not in owner
    reused = research_staged_method_conditions(agent, owner, group, 100000)
    assert reused['reused'] and reused['new_compute_cost'] == 0
    assert reused['research_program_refs'] == result['research_program_refs']
    assert agent.costs['compute_cost'] == paid
    repeated = route(agent)
    assert repeated['cache_hit'] and repeated['status'] == 'method_content_proposed_pending_validation'


def test_same_origin_is_one_stage_sample_and_missing_fit_score_is_not_pool_zero(tmp_path):
    agent, group, _ = fixture(tmp_path)
    group['pair_refs'].append(deepcopy(group['pair_refs'][0]))
    pair = read_method_reference(group['pair_refs'][1], agent.workspace)
    outcome = read_method_reference(pair['baseline_ref'], agent.workspace)
    comparison = Path(outcome['actual_outcome_ref']['file']).with_name('comparison.json')
    comparison.unlink()
    owner = next(q for q in agent.questions if q['id'] == 'task:method_choice')
    result = research_staged_method_conditions(agent, owner, group, 100000)
    assert result['unique_origins'] == 3 and result['usable_origins'] == 2
    assert result['candidate_count'] == 0
    bundle = read_method_reference(result['proposal_ref'], agent.workspace)
    assert bundle['excluded_sources'][0]['baseline']['reason'] == 'baseline_fit_comparison_missing'


def test_stage_profile_does_not_consume_future_or_pool_score():
    outcome = {'fit_loss': .2, 'execution_complete': True, 'selected_loss': 0.,
               'future_loss': 0., 'case': 99}
    profile = baseline_return_profile(outcome, 1e-7)
    changed = baseline_return_profile({**outcome, 'future_loss': 999., 'selected_loss': 8.}, 1e-7)
    assert profile == changed
    coords = [{'path': ['research', 'execution_complete'], 'kind': 'number'}]
    assert method_condition_features(profile, coords) == {'public_0': 1.}
    assert baseline_return_profile({'selected_loss': 0.}, 1e-7)['residual']['state'] == 'unmeasured'


def test_actual_incomplete_stage_return_uses_existing_route_without_closing_question(tmp_path):
    from hegel_machine.cognition.method_action_research import record_native_partition_study
    agent, group, _ = fixture(tmp_path)
    proposed = route(agent)['method_action_research']['staged_method_proposals'][0]
    bundle = read_method_reference(proposed['proposal_ref'], agent.workspace)
    owner = next(q for q in agent.questions if q['id'] == 'task:method_choice')
    study = save(tmp_path / 'incomplete_study.json', {'proposal': bundle['proposal'],
        'basis': {'minimum_improvement': 1e-7}, 'methods': [], 'pairs': [], 'budget_incomplete': True})
    returned = record_native_partition_study(agent, owner, proposal_ref=proposed['proposal_ref'],
        study_ref=study, compute_limit=10)
    assert returned['status'] == 'incomplete_not_negative'
    result = route(agent)
    assert result['status'] == 'scoped_unknown'
    assert result['method_action_research']['reason'] == 'staged_method_development_incomplete'
    assert 'completed_agenda_basis' not in owner


def test_unsupported_stage_sequence_keeps_ordinary_method_pair_learning(tmp_path):
    agent, group, bodies = fixture(tmp_path)
    for side, kind in (('baseline', 'Pi1'), ('challenger', 'Pi3')):
        bodies[side]['transfer_partition']['kind'] = kind
        save(tmp_path / (side + '_program.json'), bodies[side])
    reviewed = route(agent)['method_action_research']
    staged, = reviewed['staged_method_proposals']
    assert staged['status'] == 'unsupported_staged_method_pair'
    assert staged['partition_kinds'] == {'baseline': 'Pi1', 'continuation': 'Pi3'}
    assert staged['new_compute_cost'] == 0
    assert len(reviewed['paired_condition_research']) == 1
    assert reviewed['relative_labels_created'] > 0
    assert not list((agent.workspace / 'method_feedback').glob('staged_method_*.json'))
