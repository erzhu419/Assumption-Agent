"""A sequential return trains its own next-stage question, without new fits."""
from copy import deepcopy
from pathlib import Path

from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.method_conditions import _conflict_predictions
from hegel_machine.cognition.staged_method_research import (
    baseline_return_profile, research_parent_bound_method_conditions)
from test_native_method_composition import measured_composition_pair
from test_native_method_return_route import route, save
from test_native_partition_proposals import forbidden
from test_staged_condition_revision import delivered_stage_pair


def saved_sources(tmp_path, losses):
    """Supply distinct measured-return fixtures with a real replay input graph.

    The reported losses below exercise condition learning, not a scientific
    experiment. The prefix/continuation binding comes from the runtime fixture.
    """
    agent, owner, _, actual = measured_composition_pair(tmp_path, attempts=48)
    proposal_ref = owner['method_use_plan']['staged_method_proposal_refs'][0]
    proposal = read_method_reference(proposal_ref, agent.workspace)
    body = proposal['proposal']['candidates'][0]['research_program']
    groups = []
    for index, (before, after) in enumerate(losses):
        root = tmp_path / ('measured_' + str(index))
        group = delivered_stage_pair(agent, root, body, before)
        artifact = read_method_reference(group['baseline_method_artifact_ref'], agent.workspace)
        save(Path(artifact['execution']['research_program_ref']['file']), body['staged_research']['baseline'])
        pair = read_method_reference(group['pair_refs'][0], agent.workspace)
        # Exposed independent evidence must not validate the new condition.
        pair['condition_validation_cohort_refs'] = [{'file': 'old_method_holdout.json'}]
        save(Path(group['pair_refs'][0]['file']), pair)
        output = read_method_reference(pair['challenger_ref'], agent.workspace)
        measured = deepcopy(actual)
        measured['baseline']['all_paid_loss'] = before
        measured['continuation']['all_paid_loss'] = after
        measured['guard']['profile'] = baseline_return_profile(
            {'fit_loss': before, 'execution_complete': True}, 1e-7)
        ref = output['actual_outcome_ref']
        save(Path(ref['file']), {**read_method_reference(ref, agent.workspace),
            'development_rows': 64, 'staged_research_result': measured})
        groups.append(group)
    return agent, owner, groups, proposal_ref


def test_actual_mixed_sequential_returns_synthesize_then_publish_without_changing_content(tmp_path, monkeypatch):
    agent, owner, groups, original = saved_sources(tmp_path, [(.4, 0.), (.2, 0.), (.01, .01)])
    frozen = Path(original['file']).read_bytes()
    monkeypatch.setattr('hegel_machine.cognition.research_programs._Fitter.fit', forbidden)
    learned, = research_parent_bound_method_conditions(agent, owner, groups, 100000)
    assert learned['status'] == 'staged_program_proposed'
    assert learned['condition_resynthesis'] and learned['usable_origins'] == 3
    assert learned['append_preferred'] == 2 and learned['retain_preferred'] == 1
    bundle = read_method_reference(learned['condition_research_ref'], agent.workspace)
    assert bundle['research']['group_validation_refs'] == []
    assert bundle['research']['condition_validation_cohort_refs'] == []
    assert not learned['general_guard_supported'] and not learned['independent_support']
    assert all(item['previous_data_role'] == 'condition_validation'
        and item['data_role'] == 'exposed_development' for item in bundle['baseline_evidence'])
    old = read_method_reference(original, agent.workspace)['proposal']['candidates'][0]['research_program']
    for candidate in bundle['proposal']['candidates']:
        stage = candidate['research_program']['staged_research']
        assert stage['operation'] == 'revise_then_reidentify'
        assert stage['baseline'] == old['staged_research']['baseline']
        assert stage['continuation'] == old['staged_research']['continuation']
        assert stage['condition_origin'] == 'learned_actual_parent_bound_returns'
        assert all(coordinate['path'][0] in {'residual', 'research'} for coordinate in stage['coordinate_paths'])
        votes = [_conflict_predictions({'coordinate_paths': stage['coordinate_paths'],
            'candidates': [{'condition': stage['condition']}]}, item['profile'])[0]['helpful']
            for item in bundle['baseline_evidence']]
        assert votes == [True, True, False]
    assert Path(original['file']).read_bytes() == frozen
    assert learned['proposal_ref'] in owner['method_use_plan']['staged_method_proposal_refs']
    paid = agent.costs['compute_cost']
    reused, = research_parent_bound_method_conditions(agent, owner, groups, 0)
    assert reused['reused'] and reused['new_compute_cost'] == 0
    assert agent.costs['compute_cost'] == paid


def test_three_positive_sources_remain_insufficient_and_zero_repeat(tmp_path):
    agent, owner, groups, _ = saved_sources(tmp_path, [(.4, 0.), (.2, 0.), (.01, 0.)])
    item, = research_parent_bound_method_conditions(agent, owner, groups, 100000)
    assert item['usable_origins'] == item['append_preferred'] == 3
    assert item['status'] == 'parent_bound_condition_evidence_insufficient'
    assert item['candidate_count'] == 0 and not item['condition_resynthesis']
    assert item['research_status'] == 'insufficient_paired_preferences'
    assert item['missing_source_count'] == 0 and item['missing_preference_directions'] == ['baseline']
    assert item['required_work'] == 'seek_actual_contrasting_return_or_retain_local_method'
    assert research_parent_bound_method_conditions(agent, owner, groups, 0)[0]['new_compute_cost'] == 0


def test_same_physical_source_cannot_supply_three_method_samples(tmp_path):
    agent, owner, groups, _ = saved_sources(tmp_path, [(.4, 0.), (.2, 0.), (.01, .01)])
    for group in groups:
        reference = group['pair_refs'][0]
        pair = read_method_reference(reference, agent.workspace)
        pair['physical_source_ref'] = {'file': 'one_actual_task.json'}
        save(Path(reference['file']), pair)
    item, = research_parent_bound_method_conditions(agent, owner, groups, 100000)
    assert item['usable_origins'] == 1 and item['candidate_count'] == 0
    assert item['retain_preferred'] == 1  # Latest return replaces, never multiplies, this source.


def test_mixed_but_identical_stage_observables_request_content_diagnosis(tmp_path):
    agent, owner, groups, _ = saved_sources(tmp_path, [(.2, 0.), (.2, 0.), (.2, .2)])
    item, = research_parent_bound_method_conditions(agent, owner, groups, 100000)
    assert item['usable_origins'] == 3 and item['missing_preference_directions'] == []
    assert item['research_status'] == 'conflicting_public_view'
    assert item['candidate_count'] == 0
    assert item['required_work'] == 'diagnose_retained_stage_observables_or_method_content'


def test_completed_successor_advances_common_to_evidence_not_repeat_validation(tmp_path):
    agent, owner, groups, _ = saved_sources(tmp_path, [(.4, 0.)])
    result = route(agent, allowance=1000000)['method_action_research']
    assert result['status'] == 'method_condition_evidence_insufficient'
    assert result['reason'] == 'parent_bound_condition_evidence_insufficient'
    assert result['required_work'] == 'collect_distinct_complete_sequential_sources'
    assert result['native_composition_condition_research'][0]['usable_origins'] >= 1
    paid = agent.costs['compute_cost']
    repeated = route(agent, allowance=1000000)
    assert repeated['new_compute_cost'] == 0 and agent.costs['compute_cost'] == paid


def test_ordinary_route_exposes_learned_executable_conditions(tmp_path):
    agent, owner, _, _ = saved_sources(tmp_path, [(.4, 0.), (.2, 0.), (.01, .01)])
    result = route(agent, allowance=1000000)['method_action_research']
    assert result['status'] == 'method_condition_revision_pending_validation'
    proposal = result['native_composition_condition_research'][0]
    assert proposal['candidate_count'] > 0
    assert result['required_work'] == 'freeze_learned_append_condition_before_independent_validation'
    assert proposal['proposal_ref'] in owner['method_use_plan']['staged_method_proposal_refs']
