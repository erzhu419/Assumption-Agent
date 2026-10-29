"""Condition selection is frozen before successor execution and never updated by it."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts import run_method_condition_cohort as cohort


def reference(path, value):
    return cohort.save(path, value)


@pytest.mark.parametrize('guard', [False, True])
def test_declared_roles_accumulate_then_freeze_without_successor_feedback(tmp_path, monkeypatch, guard):
    base = {'ordinary_source_ref': {'file': '/retained/ordinary.py'}, 'cells': {
        '00': {'id': 'P0', 'selection_plan': {'ordinary_method_path': 'ordinary.py'}, 'source_refs': []},
        '10': {'id': 'P1', 'selection_plan': {'ordinary_method_path': 'ordinary.py',
            'first_segment': {'current_option': {'id': 'whole', 'family': 'N'}}}, 'source_refs': []}}}
    bundle = reference(tmp_path / 'bundle.json', base)
    prior = [{'file': '/retained/pair.json', 'json_path': []}]
    seen = []

    def fake_source(request, output):
        output.mkdir()
        role = request['data_role']
        seen.append(deepcopy(request))
        paired = reference(output / 'pair.json', {'data_role': role})
        outcomes = {}
        for name in ('00', '10'):
            outcomes[name] = reference(output / f'outcome_{name}.json', {
                'loss': 0., 'completed': True, 'costs': {'compute_cost': 20, 'observation_cost': 0},
                'method_ref': {'id': name}})
        selection = {'local_program_supported': True,
            'general_guard_supported': guard and role == 'condition_validation',
            'candidates': [{'condition': {'learned': True}}] if guard else []}
        selected = reference(output / 'selection.json', selection)
        artifact = {'id': 'P1_guard', 'selection_plan': {
            'ordinary_method_path': 'shared/P1_ordinary.py', 'condition': {'learned': True},
            'when_true': base['cells']['10']['selection_plan']}, 'source_evidence_refs': prior}
        routed = reference(output / 'route.json', {'method_action_research': {'installed_candidate': artifact}})
        feedback = {'selection_ref': selected, 'research_route_ref': routed, 'new_compute_cost': 3}
        evaluation = reference(output / 'evaluation.json', {'new_response_rows': 512})
        study = reference(output / 'summary.json', {
            'pair_refs': [paired], 'outcome_refs': outcomes,
            'ordinary_use': feedback if role != 'independent_successor' else {'new_compute_cost': 0},
            'evaluation_lifecycle': {'labels_revealed': True, 'receipt_ref': evaluation}})
        result = {'status': 'completed', 'study_ref': study,
            'prefix_costs': {'compute_cost': 0, 'observation_cost': 213},
            'shared_comparison_costs': {'compute_cost': 10},
            'static_proposal_costs': {'compute_cost': 2}}
        reference(output / 'source_result.json', result)
        return result

    monkeypatch.setattr(cohort.source_entry, 'run', fake_source)
    requests = lambda role: [{'source_id': role, 'case': {'seed': role}}]
    registration = {'method_pair_ref': bundle, 'prior_pair_refs': prior,
        'phases': {role: requests(role) for role in (
            'development', 'condition_validation', 'independent_successor')}}
    result = cohort.run(registration, tmp_path / 'cohort')
    roles = [request['data_role'] for request in seen]
    assert roles == (['development', 'condition_validation', 'independent_successor'] if guard
                     else ['development', 'independent_successor'])
    assert seen[0]['prior_pair_refs'] == prior
    assert seen[-1]['prior_pair_refs'] == []
    assert len(result['condition_pair_refs']) == (3 if guard else 2)
    successor = cohort.read(seen[-1]['method_pair_ref'])
    assert successor['cells']['00'] == base['cells']['00']
    if guard:
        assert successor['cells']['10']['id'] == 'P1_guard'
        assert successor['cells']['10']['selection_plan']['ordinary_method_path'] == 'ordinary.py'
        assert cohort.read(successor['cells']['10']['selection_ref'])['general_guard_supported']
    else:
        assert successor == base
        assert result['phase_status']['condition_validation'] == 'not_run_no_development_condition'
    assert result['selected_policy']['selected_policy_kind'] == (
        'learned_condition' if guard else 'locally_supported_fixed_method')
    assert result['costs']['compute_cost'] == len(seen) * 52 + (len(seen) - 1) * 3
    assert result['costs']['observation_cost'] == len(seen) * 725
    assert cohort.read(bundle) == base


def test_solved_sources_costs_retained_without_fabricating_pairs(tmp_path, monkeypatch):
    selection = reference(tmp_path / 'selection.json', {'local_program_supported': True,
        'general_guard_supported': False, 'candidates': []})
    previous = reference(tmp_path / 'feedback.json', {'selection_ref': selection,
        'research_route_ref': {'file': '/retained/route.json'}})
    bundle = reference(tmp_path / 'bundle.json', {'cells': {'00': {}, '10': {}}})
    seen = []

    def solved(request, output):
        seen.append(request['data_role'])
        output.mkdir()
        result = {'status': 'retained_models_explain_paid_prefix_no_research',
            'prefix_costs': {'observation_cost': 213},
            'shared_comparison_costs': {'compute_cost': 48 * 213},
            'static_proposal_costs': {'compute_cost': 4}}
        reference(output / 'source_result.json', result)
        return result

    monkeypatch.setattr(cohort.source_entry, 'run', solved)
    result = cohort.run({'method_pair_ref': bundle, 'initial_method_feedback_ref': previous,
        'prior_pair_refs': [{'file': '/retained/pair.json'}], 'phases': {
            'development': [{'case': {'seed': 1}}],
            'condition_validation': [{'case': {'seed': 2}}],
            'independent_successor': [{'case': {'seed': 3}}]}}, tmp_path / 'cohort')
    assert seen == ['development', 'independent_successor']
    assert result['condition_pair_refs'] == [{'file': '/retained/pair.json'}]
    assert all(not source['pair_refs'] and not source['arms'] for source in result['sources'])
    assert result['costs']['compute_cost'] == 2 * (48 * 213 + 4)
    assert result['costs']['observation_cost'] == 426
    assert result['selected_policy']['selected_policy_kind'] == 'locally_supported_fixed_method'


def test_no_supported_method_does_not_start_successors(tmp_path, monkeypatch):
    def unexpected(*args):
        raise AssertionError('no unregistered continuation')

    monkeypatch.setattr(cohort.source_entry, 'run', unexpected)
    result = cohort.run({'method_pair_ref': {'file': '/unused/bundle.json'}, 'phases': {
        'independent_successor': [{'case': {'seed': 9}}]}}, tmp_path / 'cohort')
    assert result['phase_status']['independent_successor'] == 'not_run_no_supported_method'
    assert result['selected_policy'] is None


def test_missing_arm_receipt_preserves_comparison_cost_without_repaying_old_arms(tmp_path):
    outcomes = {}
    for name, cost in [('00', 100), ('10', 90), ('20', 80)]:
        value = {'costs': {'compute_cost': cost}, 'loss': 0.}
        if name != '10':
            value['new_stage_costs'] = {'compute_cost': 0}
        outcomes[name] = reference(tmp_path / f'{name}.json', value)
    evaluation = reference(tmp_path / 'evaluation.json', {'new_response_rows': 0})
    summary = reference(tmp_path / 'summary.json', {'outcome_refs': outcomes, 'pair_refs': [],
        'ordinary_use': {'new_compute_cost': 0},
        'evaluation_lifecycle': {'labels_revealed': True, 'receipt_ref': evaluation}})
    result = cohort.source_receipt({'status': 'completed', 'study_ref': summary}, tmp_path)
    assert result['costs']['compute_cost'] == 90
    assert result['costs']['observation_cost'] == 0
    assert [result['arms'][name]['costs']['compute_cost'] for name in ('00', '10', '20')] == [100, 90, 80]


def test_branch_attempt_limit_counts_failure_and_keeps_screening_cost(tmp_path, monkeypatch):
    seen = []

    def source(request, output):
        seen.append(request['source_id'])
        output.mkdir()
        matched = len(seen) > 1
        decision = reference(output / 'screen.json', {'matched': matched})
        return {'status': 'method_execution_failed' if matched else 'method_condition_other_branch',
            'prefix_costs': {'observation_cost': 213},
            'shared_comparison_costs': {'compute_cost': 34 * 213},
            'pre_use_condition_ref': decision}

    monkeypatch.setattr(cohort.source_entry, 'run', source)
    result = cohort.run_three_way_study({'supplements': [], 'method_bundle_ref': {'file': '/frozen'},
        'branch_sources': [{'source_id': f'source_{i}'} for i in range(5)],
        'max_branch_attempts': 2}, tmp_path / 'run')
    assert seen == ['source_0', 'source_1', 'source_2']
    assert result['branch_attempts'] == result['branch_candidates_not_opened'] == 2
    assert result['costs']['observation_cost'] == 3 * 213
    assert result['costs']['compute_cost'] == 3 * 34 * 213


def test_complete_factorial_panel_retains_solved_failed_and_completed_cells(tmp_path, monkeypatch):
    seen = []
    statuses = ['retained_models_explain_paid_prefix_no_research',
                'method_execution_failed', 'completed']

    def source(request, output):
        seen.append(deepcopy(request))
        output.mkdir()
        return {'status': statuses[len(seen) - 1],
            'prefix_reused': request.get('paid_prefix_ref') is not None,
            'prefix_costs': {'observation_cost': 213},
            'shared_comparison_costs': {'compute_cost': 10}}

    monkeypatch.setattr(cohort.source_entry, 'run', source)
    sources = [{'source_id': f'cell_{i}', 'factorial_cell': {'mature': i // 2, 'startup': i % 2},
                'evidence_role': 'registered_factorial_successor'} for i in range(3)]
    sources[0]['paid_prefix_ref'] = {'file': '/original/prefix.json'}
    template = reference(tmp_path / 'template.json', {'parents': [{'candidate_id': 'retained'}],
        'case': {'seed': 'must_be_overridden'}})
    for i, source in enumerate(sources):
        source['case'] = {'seed': i}
    result = cohort.run_three_way_study({'method_bundle_ref': {'file': '/frozen'},
        'source_template_ref': template, 'sources': sources}, tmp_path / 'run')
    assert [s['source_id'] for s in seen] == ['cell_0', 'cell_1', 'cell_2']
    assert all(s['data_role'] == 'independent_successor' and s['prior_pair_refs'] == [] for s in seen)
    assert [s['case']['seed'] for s in seen] == [0, 1, 2]
    assert all(s['parents'] == [{'candidate_id': 'retained'}] for s in seen)
    assert all('paid_prefix_ref' not in s for s in seen[1:])
    assert [s['status'] for s in result['sources']] == statuses
    assert [s['factorial_cell'] for s in result['sources']] == [s['factorial_cell'] for s in sources]
    assert result['source_candidates_not_opened'] == 0
    assert result['costs']['compute_cost'] == 30
    assert result['costs']['observation_cost'] == 426


def test_paid_profile_reconstruction_keeps_returns_and_uses_only_method_research(tmp_path, monkeypatch):
    from scripts import run_state_method_pairing as pairing
    from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
    from test_state_method_cohort import source_fixture, frozen_pair, run_source

    paid, case, values = source_fixture(tmp_path / 'source', 0)
    bundle = frozen_pair(tmp_path, paid, case)
    summary, outcome, study = run_source(
        tmp_path / 'study', paid, case, values, bundle, [], monkeypatch)
    pair_ref = summary['pair_refs'][0]
    pair = cohort.read(pair_ref)
    originals = {reference['file']: Path(reference['file']).read_bytes() for reference in
                 (pair_ref, pair['baseline_ref'], pair['challenger_ref'], outcome['profile_ref'])}
    registration_ref = {'file': str(tmp_path / 'study' / 'prepared' / 'registration.json'), 'json_path': []}
    feedback_ref = {'file': str(study / 'summary.json'), 'json_path': ['ordinary_use']}
    original_execute = CommonArchiveInquiryAgent._execute
    called = []

    def only_method_research(self, action, arguments, lab):
        assert action == 'run' and arguments['operator'] == 'research_route'
        assert arguments.get('method_action_revision')
        called.append(arguments)
        return original_execute(self, action, arguments, lab)

    def no_world_execution(*args, **kwargs):
        raise AssertionError('saved profiles must not trigger world fitting or readout')

    monkeypatch.setattr(CommonArchiveInquiryAgent, '_execute', only_method_research)
    monkeypatch.setattr(pairing, 'evaluate_terminal_model', no_world_execution)
    monkeypatch.setattr(pairing, 'run', no_world_execution)
    rebuilt = cohort.rebuild_paid_profiles({'sources': [{'pair_ref': pair_ref,
        'registration_ref': registration_ref}], 'method_feedback_ref': feedback_ref}, tmp_path / 'rebuilt')
    assert len(called) == 1
    assert rebuilt['new_observations'] == rebuilt['new_predictions'] == rebuilt['new_fits'] == 0
    assert rebuilt['new_compute_cost'] > 0
    old_allowance = cohort.read(registration_ref)['method_condition_compute_cap']
    assert rebuilt['remaining_original_compute_allowance'] == (
        old_allowance - summary['ordinary_use']['new_compute_cost'] - rebuilt['new_compute_cost'])
    derived = cohort.read(rebuilt['pair_refs'][0])
    assert derived['origin_ref'] == pair_ref and derived['original_pair_ref'] == pair_ref
    for side in ('baseline', 'challenger'):
        original, amended = (cohort.read(value[side + '_ref']) for value in (pair, derived))
        assert amended['original_profile_ref'] == original['profile_ref']
        assert amended['original_outcome_ref'] == pair[side + '_ref']
        for key in original:
            if key != 'profile_ref':
                assert amended[key] == original[key]
    assert all(Path(path).read_bytes() == content for path, content in originals.items())
