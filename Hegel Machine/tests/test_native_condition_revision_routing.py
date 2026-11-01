"""Common revises conditions from paid returns without changing the methods."""

import json
from pathlib import Path

from hegel_machine.cognition import method_conditions, method_feedback
from hegel_machine.cognition.common_object_research import record_native_parent_research_return
from hegel_machine.cognition.method_action_research import (
    read_method_reference, review_native_method_returns,
)
from test_native_paired_method_feedback import common, deliver, route, save


def read(reference):
    value = json.loads(open(reference['file']).read())
    for part in reference.get('json_path', []):
        value = value[part]
    return value


def owner(agent):
    return next(question for question in agent.questions if question['id'] == 'task:method_choice')


def measured(agent, root, name, count, helpful):
    _, reference = deliver(agent, root, name, cheaper=helpful)
    pair = read(reference)
    profile_ref = read(pair['baseline_ref'])['profile_ref']
    profile = read(profile_ref)
    profile['observation_count'] = count
    save(Path(profile_ref['file']), profile)
    return reference


def freeze_returns(root, family, references, *, cohort):
    frozen_ref = save(root / 'frozen_conditions.json', [family])
    for index, reference in enumerate(references):
        pair = read(reference)
        profile = read(read(pair['baseline_ref'])['profile_ref'])
        prediction_ref = save(root / f'prediction_{index}.json', {
            'frozen_conditions_ref': frozen_ref, 'profile': profile,
            'before_method_execution': True,
            'predictions': [{'candidate_predictions': method_conditions._conflict_predictions(
                family, profile)}]})
        pair.update(data_role='condition_validation', frozen_conditions_ref=frozen_ref,
            condition_prediction_ref=prediction_ref, condition_validation_cohort_refs=cohort)
        save(Path(reference['file']), pair)
    return frozen_ref


def completed_revision(tmp_path, *, allowance=100000):
    agent = common(tmp_path)
    development = [measured(agent, tmp_path, name, count, helpful) for name, count, helpful in (
        ('d0', 10, True), ('d1', 20, True), ('d2', 40, False))]
    initial = route(agent)['method_action_research']['paired_condition_research'][0]
    assert initial['candidates'] and not initial['general_guard_supported']
    validation = [measured(agent, tmp_path, name, count, False)
        for name, count in (('v0', 25), ('v1', 35))]
    cohort = [read(reference)['physical_source_ref'] for reference in validation]
    freeze_returns(tmp_path / 'first_frozen', initial, validation, cohort=cohort)
    original = {reference['file']: open(reference['file'], 'rb').read()
        for reference in [*development, *validation]}
    reviewed = route(agent, allowance=allowance)['method_action_research']
    revision, = reviewed['native_condition_revisions']
    return agent, development, validation, original, initial, reviewed, revision


def test_common_forms_a_new_condition_generation_and_preserves_the_failed_validation(tmp_path):
    agent, development, validation, original, initial, reviewed, revision = completed_revision(tmp_path)
    failed, = reviewed['paired_condition_research']
    assert failed['status'] == 'development_condition_without_group_support'
    assert failed['group_validation_refs'] == validation
    assert failed['candidates'] == initial['candidates']
    assert revision['version'] == 1 and revision['new_compute_cost'] > 0
    assert revision['new_fits'] == revision['new_predictions'] == revision['new_observations'] == 0
    assert revision['original_pair_refs'] == [*development, *validation]
    assert owner(agent)['method_use_plan']['native_condition_revision_refs'] == [revision['revision_ref']]
    learned = read_method_reference(revision['research_ref'], agent.workspace)
    assert learned['paired_invocations'] == 5 and learned['candidates']
    assert not learned['general_guard_supported'] and not learned['new_version_supported']
    assert learned['candidates'] != initial['candidates']
    for field in ('method_ref', 'baseline_method_ref', 'method_artifact_ref', 'baseline_method_artifact_ref'):
        assert learned[field] == initial[field]
    for reference in revision['development_pair_refs']:
        pair = read(reference)
        assert pair['data_role'] == 'development'
        assert not pair.get('condition_validation_cohort_refs')
        assert not pair.get('frozen_conditions_ref')
    assert {path: open(path, 'rb').read() for path in original} == original


def test_originals_and_reference_views_still_count_five_invocations(tmp_path):
    agent, _, _, _, _, _, revision = completed_revision(tmp_path)
    # Ordinary feedback can reference the same physical invocation from its
    # retained development view. This is not five additional research uses.
    for index, reference in enumerate(revision['development_pair_refs']):
        pair = read(reference)
        challenger = read(pair['challenger_ref'])
        receipt = save(tmp_path / f'view_return_{index}.json', {
            'status': 'paired_completed', 'method_pair_ref': reference,
            'method_ref': challenger['method_ref'],
            'method_artifact_ref': challenger['method_artifact_ref']})
        record_native_parent_research_return(agent,
            source_task_ref=pair['physical_source_ref'], result_ref=receipt,
            mission={'purpose': 'method_learning', 'allocation': {'compute': 100000}})
    result = route(agent)['method_action_research']
    learned, = result['paired_condition_research']
    assert learned['paired_invocations'] == 5
    assert len(learned['preferences']) == result['relative_labels_created'] == 5
    assert len(owner(agent)['method_use_plan']['native_condition_revision_refs']) == 1


def test_complete_condition_revision_is_reused_without_another_condition_fit(tmp_path, monkeypatch):
    agent, _, _, _, _, _, revision = completed_revision(tmp_path)
    def prohibited(*args, **kwargs):
        raise AssertionError('The completed development generation must be reused')
    monkeypatch.setattr(method_conditions, 'research_method_conflict', prohibited)
    monkeypatch.setattr(method_feedback, 'revise_paired_method_conditions', prohibited)
    # Bypass the routing-result cache to exercise the method-topic cache itself.
    result = review_native_method_returns(agent, owner(agent), compute_limit=100000)
    learned, = result['paired_condition_research']
    assert learned['reused'] and learned['new_compute_cost'] == 0
    assert learned['condition_revision_ref'] == revision['revision_ref']
    assert learned['paired_invocations'] == 5 and result['native_condition_revisions'] == []


def test_common_resumes_an_unfinished_condition_revision_using_its_paid_projection(tmp_path):
    # Six units review the five native returns; six project their immutable
    # pairs. This deliberately leaves no allowance for condition learning.
    agent, development, validation, original, _, first, pending = completed_revision(
        tmp_path, allowance=12)
    assert first['status'] == 'scoped_unknown'
    assert pending['budget_incomplete'] and pending['new_compute_cost'] == 6
    assert len(pending['development_pair_refs']) == 5
    assert pending['projection_reused'] is False
    assert owner(agent)['method_use_plan']['native_condition_revision_refs'] == [pending['revision_ref']]
    repeated = review_native_method_returns(agent, owner(agent), compute_limit=12)
    unchanged, = repeated['paired_condition_research']
    assert unchanged['pair_refs'] == [*development, *validation]
    assert unchanged['condition_validation_mode'] == 'frozen_before_method_call'
    assert unchanged['group_validation_refs'] == validation
    repeated_revision, = repeated['native_condition_revisions']
    assert repeated_revision['cache_hit'] and repeated_revision['new_compute_cost'] == 0
    assert repeated_revision['revision_ref'] == pending['revision_ref']
    resumed = route(agent)['method_action_research']
    completed, = resumed['native_condition_revisions']
    assert not completed['budget_incomplete'] and completed['projection_reused']
    assert completed['development_pair_refs'] == pending['development_pair_refs']
    assert completed['previous_revision_ref'] == pending['revision_ref']
    assert completed['version'] == pending['version'] == 1
    assert completed['research']['paired_invocations'] == 5
    assert completed['new_compute_cost'] == completed['research']['compute_cost']
    assert not completed['general_guard_supported'] and completed['research']['candidates']
    assert {path: open(path, 'rb').read() for path in original} == original


def test_new_frozen_validation_uses_the_revision_without_old_cohort_or_old_family(tmp_path):
    agent, _, old_validation, original, _, _, revision = completed_revision(tmp_path)
    learned = read(revision['research_ref'])
    first = measured(agent, tmp_path, 'next0', 12, True)
    second_source = {'file': str(tmp_path / 'next1/source.json')}
    cohort = [read(first)['physical_source_ref'], second_source]
    freeze_returns(tmp_path / 'second_frozen', learned, [first], cohort=cohort)
    partial = route(agent)['method_action_research']['paired_condition_research'][0]
    assert partial['status'] == 'awaiting_registered_condition_validation'
    assert partial['missing_condition_validation_source_refs'] == [second_source]
    second = measured(agent, tmp_path, 'next1', 28, False)
    freeze_returns(tmp_path / 'second_frozen', learned, [first, second], cohort=cohort)
    result = route(agent)['method_action_research']['paired_condition_research'][0]
    assert result['status'] == 'paired_condition_supported_on_declared_groups'
    assert result['paired_invocations'] == 7
    assert result['group_validation_refs'] == [first, second]
    assert result['condition_validation_cohort_refs'] == cohort
    assert result['candidates'] == learned['candidates']
    assert not any(reference in result['group_validation_refs'] for reference in old_validation)
    assert {path: open(path, 'rb').read() for path in original} == original


def test_second_revision_keeps_original_origin_mapping_on_another_common_review(tmp_path):
    agent, _, _, _, _, _, first_revision = completed_revision(tmp_path)
    family = read(first_revision['research_ref'])
    validation = [measured(agent, tmp_path, name, count, False)
        for name, count in (('next0', 21), ('next1', 28))]
    cohort = [read(reference)['physical_source_ref'] for reference in validation]
    freeze_returns(tmp_path / 'second_frozen', family, validation, cohort=cohort)
    result = route(agent)['method_action_research']
    second_revision, = result['native_condition_revisions']
    assert second_revision['version'] == 2
    assert second_revision['previous_revision_ref'] == first_revision['revision_ref']
    assert second_revision['research']['paired_invocations'] == 7
    repeated = review_native_method_returns(agent, owner(agent), compute_limit=100000)
    learned, = repeated['paired_condition_research']
    assert learned['reused'] and learned['paired_invocations'] == 7
    assert learned['condition_revision_ref'] == second_revision['revision_ref']
    assert repeated['native_condition_revisions'] == []
