"""Actual native returns create addressed content, not a method victory."""

from copy import deepcopy
from pathlib import Path

from hegel_machine.cognition.common_object_research import (
    CommonObjectDependencyMethods, record_native_parent_research_return,
)
from hegel_machine.cognition.method_action_research import (
    quote_native_partition_analysis, quote_existing_interval_return,
    read_method_reference, record_native_partition_study,
)
from hegel_machine.cognition.method_feedback import (
    ensure_python_method_topic, export_python_method_topic,
)
from hegel_machine.cognition.models import SearchBudget
from test_native_method_return_route import route, save
from test_transfer_partitions import plan


def scene(tmp_path, *, attach=True, changed=True, allowance=10000):
    source = {'physical_source': 'one_actual_research'}
    before, after = plan(dormant=True), plan(.85, weight=3., dormant=True)
    if not changed:
        after = deepcopy(before)
    before_ref = save(tmp_path / 'before.json', {'research_program': before})
    after_ref = save(tmp_path / 'after.json', {'research_program': after})
    evidence = save(tmp_path / 'revision.json', {
        'source_task_ref': source, 'before_entry_ref': before_ref,
        'after_entry_ref': after_ref, 'source_operation': 'paid_joint_structure_search',
        'comparison_ref': {'file': 'retained_full_pool.json'},
        'data_role': 'exposed_development', 'cost_refs': [{'file': 'original_paid_account.json'}]})
    actual = save(tmp_path / 'returned.json', {'status': 'confirmed',
        'execution_complete': True, **({'revision_evidence_ref': evidence} if attach else {})})
    common = CommonObjectDependencyMethods(tmp_path / 'common', [],
        budget=SearchBudget(10000), actions=[]).common
    returned = record_native_parent_research_return(common, source_task_ref=source,
        result_ref=actual, mission={'purpose': 'method_learning', 'allocation': {'compute': allowance}})
    return common, returned, evidence, actual, before, after


def forbidden(*args, **kwargs):
    raise AssertionError('A saved method proposal must not refit or regenerate')


def test_actual_common_route_compiles_partition_without_fitting_or_support(tmp_path, monkeypatch):
    common, returned, evidence, _, before, after = scene(tmp_path)
    from hegel_machine.cognition import research_programs
    monkeypatch.setattr(research_programs, 'replay_research_program', forbidden)
    result = route(common, allowance=10000)
    reviewed = result['method_action_research']
    assert result['status'] == 'method_content_proposed_pending_validation'
    assert result['new_compute_cost'] == 2 + quote_native_partition_analysis(before, after)
    published, = reviewed['transfer_partition_proposals']
    bundle = read_method_reference(published['proposal_ref'], common.workspace)
    candidate = read_method_reference(published['candidate_refs'][0], common.workspace)
    assert bundle['basis']['revision_evidence_ref'] == evidence
    assert bundle['feedback_ref'] == returned['feedback_ref']
    assert bundle['source_operation'] == 'paid_joint_structure_search'
    assert not bundle['source_support_inherited'] and bundle['paired_effect'] is None
    assert candidate['research_program']['transfer_partition']['kind'] == 'Pi1'
    assert candidate['research_program']['selection_steps'][0]['parameters'][0]['radius'] == .4
    assert reviewed['distinct_source_refs'] == [{'physical_source': 'one_actual_research'}]
    assert reviewed['relative_labels_created'] == 0 and reviewed['installed_candidate'] is None
    assert not common.theory_consolidation.get('candidates')
    assert common.costs['model_calls'] == common.costs['observation_cost'] == 0
    repeated = route(common, allowance=10000)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0


def test_late_reference_keeps_one_return_and_restored_topic_reuses_compilation(tmp_path, monkeypatch):
    common, returned, evidence, actual, _, _ = scene(tmp_path, attach=False)
    route(common, allowance=10000)
    original = Path(returned['feedback_ref']['file']).read_text()
    attached = record_native_parent_research_return(common,
        source_task_ref={'physical_source': 'one_actual_research'}, result_ref=actual,
        revision_evidence_ref=evidence)
    assert attached['status'] == 'reused_native_return'
    assert Path(returned['feedback_ref']['file']).read_text() == original
    first = route(common, allowance=10000)
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    assert len(owner['method_use_plan']['native_return_refs']) == 1
    frozen = export_python_method_topic(common, owner)
    restored = CommonObjectDependencyMethods(tmp_path / 'restored', [],
        budget=SearchBudget(10000), actions=[]).common
    continued = ensure_python_method_topic(restored,
        parent_question_ref={'question_id': 'task'}, topic_ref=frozen)
    continued.pop('routing_execution', None)
    from hegel_machine.cognition import transfer_partitions
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', forbidden)
    resumed = route(restored, allowance=10000)
    assert resumed['new_compute_cost'] == 2
    assert resumed['method_action_research']['transfer_partition_proposals'] == (
        first['method_action_research']['transfer_partition_proposals'])
    assert len(continued['method_use_plan']['native_return_refs']) == 1


def test_complete_analysis_quote_precedes_generator_and_unfunded_is_unresolved(tmp_path, monkeypatch):
    common, _, _, _, before, after = scene(tmp_path, allowance=2)
    from hegel_machine.cognition import transfer_partitions
    original = transfer_partitions.propose_transfer_partitions
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', forbidden)
    held = route(common, allowance=2)
    assert held['status'] == 'scoped_unknown'
    assert held['method_action_research']['required_compute'] == 2 + quote_native_partition_analysis(before, after)
    assert not list((common.workspace / 'method_feedback').glob('transfer_partition_*.json'))
    assert common.costs['compute_cost'] == 2
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    owner['mission']['allocation']['compute'] = 10000
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', original)
    assert route(common, allowance=10000)['status'] == 'method_content_proposed_pending_validation'


def test_unchanged_paid_return_finishes_without_inventing_method_content(tmp_path):
    common, _, _, _, _, _ = scene(tmp_path, changed=False)
    result = route(common, allowance=10000)
    assert result['status'] == 'complete_native_return_review'
    published, = result['method_action_research']['transfer_partition_proposals']
    assert published['status'] == 'complete_no_partition_proposal'
    assert published['candidate_refs'] == []
    assert route(common, allowance=10000)['new_compute_cost'] == 0


def interval_execution_scene(tmp_path, *, changed_readout=False):
    common, returned, evidence_ref, _, before, _ = scene(tmp_path)
    from hegel_machine.cognition.method_action_research import partition_followup_body
    before['transfer_partition'] = {'kind': 'Pi1', 'released_groups': ['r'],
        'max_attempts': 48, 'max_candidates': 12}
    before = partition_followup_body(before)
    after = plan(.85, weight=3. if changed_readout else 1., dormant=True)
    after['transfer_partition'] = deepcopy(before['transfer_partition'])
    evidence = read_method_reference(evidence_ref, common.workspace)
    save(Path(evidence['before_entry_ref']['file']), {'research_program': before})
    save(Path(evidence['after_entry_ref']['file']), {'research_program': after})
    evidence.update(executed_method_ref={**evidence['before_entry_ref'], 'json_path': ['research_program']},
        method_execution_ref=save(tmp_path / 'interval_execution.json', {
            'status': 'paid_interval_supported', 'execution_complete': True,
            'fixed_readout_only': True, 'new_fits': 0}))
    save(Path(evidence_ref['file']), evidence)
    return common, returned, evidence_ref, before, after


def test_existing_interval_method_return_keeps_second_stage_without_new_proposal(tmp_path, monkeypatch):
    common, _, evidence_ref, before, after = interval_execution_scene(tmp_path)
    from hegel_machine.cognition import transfer_partitions
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', forbidden)
    result = route(common, allowance=10000)
    reviewed = result['method_action_research']
    bundle = read_method_reference(reviewed['transfer_partition_proposals'][0]['proposal_ref'], common.workspace)
    assert bundle['status'] == 'existing_method_execution_returned'
    assert bundle['basis']['revision_evidence_ref'] == evidence_ref
    assert bundle['proposal']['baseline_research_program'] == before
    assert bundle['proposal']['candidates'] == [] and bundle['required_work'] is None
    assert reviewed['existing_method_executions'][0]['execution_complete']
    assert reviewed['relative_labels_created'] == 0
    assert reviewed['independent_method_gain'] is None
    assert reviewed['distinct_source_refs'] == [{'physical_source': 'one_actual_research'}]
    assert result['new_compute_cost'] == 2 + quote_existing_interval_return(before, after)
    assert bundle['analysis_compute_cost'] == quote_existing_interval_return(before, after)
    assert route(common, allowance=10000)['new_compute_cost'] == 0
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    frozen = export_python_method_topic(common, owner)
    restored = CommonObjectDependencyMethods(tmp_path / 'restored', [],
        budget=SearchBudget(10000), actions=[]).common
    continued = ensure_python_method_topic(restored,
        parent_question_ref={'question_id': 'task'}, topic_ref=frozen)
    continued.pop('routing_execution', None)
    resumed = route(restored, allowance=10000)
    assert resumed['new_compute_cost'] == 2
    assert resumed['method_action_research']['existing_method_executions'] == reviewed['existing_method_executions']


def test_changed_readout_is_not_misreported_as_fixed_readout_execution(tmp_path):
    common, _, _, _, _ = interval_execution_scene(tmp_path, changed_readout=True)
    reviewed = route(common, allowance=10000)['method_action_research']
    assert not reviewed.get('existing_method_executions')
    published, = reviewed['transfer_partition_proposals']
    assert published['status'] == 'pending_validation' and published['candidate_refs']


def test_interval_execution_uses_actual_research_parent_not_pool_winner(tmp_path, monkeypatch):
    common, _, evidence_ref, before, _ = interval_execution_scene(tmp_path)
    evidence = read_method_reference(evidence_ref, common.workspace)
    evidence['research_parent_ref'] = evidence['before_entry_ref']
    evidence['before_entry_ref'] = save(tmp_path / 'pool_winner.json', {
        'research_program': plan(.2, weight=4., dormant=True)})
    save(Path(evidence_ref['file']), evidence)
    from hegel_machine.cognition import transfer_partitions
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', forbidden)
    reviewed = route(common, allowance=10000)['method_action_research']
    published, = reviewed['transfer_partition_proposals']
    assert published['status'] == 'existing_method_execution_returned'
    assert read_method_reference(published['baseline_ref'], common.workspace) == before


def paid_study(tmp_path, common, *, incomplete=False):
    published, = route(common, allowance=10000)['method_action_research']['transfer_partition_proposals']
    bundle = read_method_reference(published['proposal_ref'], common.workspace)
    study = {'proposal': bundle['proposal'], 'basis': {'minimum_improvement': 1e-7,
        'source_refs': [{'file': 'already_paid_training_and_validation.json'}]},
        'methods': [{'index': 0, 'execution_complete': True, 'validation_loss': .013,
                     'new_compute_cost': 0, 'reused': True},
                    {'index': 1, 'execution_complete': not incomplete,
                     'validation_loss': None if incomplete else .0000347}],
        'pairs': [{'method_index': 1, 'execution_complete': not incomplete,
                   'status': 'incomplete_not_negative' if incomplete else 'local_step_improvement',
                   'validation_loss_change': None if incomplete else -.0129653}],
        'budget_incomplete': incomplete, 'time_incomplete': False}
    study_ref = save(tmp_path / 'paid_study.json', study)
    comparison_ref = save(tmp_path / 'full_pool.json', {
        'full_comparison': not incomplete, 'parent_changed': False,
        'selected_ref': {'file': 'confirmed_parent.json'}, 'paid_rows': 195})
    return published['proposal_ref'], study_ref, comparison_ref


def test_paid_local_improvement_returns_to_ordinary_topic_with_remaining_residual(tmp_path, monkeypatch):
    common, _, _, _, _, _ = scene(tmp_path)
    proposal_ref, study_ref, comparison_ref = paid_study(tmp_path, common)
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    original = [Path(ref['file']).read_text() for ref in (proposal_ref, study_ref)]
    from hegel_machine.cognition import transfer_partitions
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', forbidden)
    monkeypatch.setattr(transfer_partitions, 'replay_transfer_partition', forbidden)
    returned = record_native_partition_study(common, owner, proposal_ref=proposal_ref,
        study_ref=study_ref, comparison_ref=comparison_ref, compute_limit=10)
    assert returned['status'] == 'local_development_improvement'
    assert returned['new_compute_cost'] == 6 and returned['remaining_validation_residual']
    assert not returned['full_pool_comparison']['parent_changed']
    assert not returned['independent_support'] and returned['installed_candidate'] is None
    assert returned['counterexample_refs'] == []
    assert returned['method_outcomes'][0]['validation_passed'] is False
    assert [Path(ref['file']).read_text() for ref in (proposal_ref, study_ref)] == original
    actual = route(common, allowance=10000)
    assert actual['status'] == 'complete_native_partition_development_review'
    proposal, = actual['method_action_research']['transfer_partition_proposals']
    assert proposal['study_return_ref'] == returned['return_ref']
    assert proposal['remaining_validation_residual']
    repeated = record_native_partition_study(common, owner, proposal_ref=proposal_ref,
        study_ref=study_ref, comparison_ref=comparison_ref, compute_limit=0)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0
    assert len(owner['method_use_plan']['transfer_partition_return_refs']) == 1


def test_incomplete_study_remains_unknown_and_unfunded_return_does_not_publish(tmp_path):
    common, _, _, _, _, _ = scene(tmp_path)
    proposal_ref, study_ref, comparison_ref = paid_study(tmp_path, common, incomplete=True)
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    held = record_native_partition_study(common, owner, proposal_ref=proposal_ref,
        study_ref=study_ref, comparison_ref=comparison_ref, compute_limit=1)
    assert held['status'] == 'scoped_unknown' and held['new_compute_cost'] == 0
    assert not owner['method_use_plan'].get('transfer_partition_return_refs')
    returned = record_native_partition_study(common, owner, proposal_ref=proposal_ref,
        study_ref=study_ref, comparison_ref=comparison_ref, compute_limit=6)
    assert returned['status'] == 'incomplete_not_negative'
    assert not returned['execution_complete'] and not returned['independent_support']
    actual = route(common, allowance=10000)
    assert actual['status'] == 'scoped_unknown'
    assert actual['method_action_research']['reason'] == 'partition_development_incomplete'


def returned_fit_with_counterexample(tmp_path, common, *, training_loss=1e-20):
    proposal_ref, study_ref, comparison_ref = paid_study(tmp_path, common)
    study = read_method_reference(study_ref, common.workspace)
    study['methods'][1].update(training_loss=training_loss,
        research_program=plan(.85, weight=3., dormant=True))
    save(Path(study_ref['file']), study)
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    result = record_native_partition_study(common, owner, proposal_ref=proposal_ref,
        study_ref=study_ref, comparison_ref=comparison_ref, compute_limit=6)
    return owner, proposal_ref, study_ref, result


def test_actual_fit_success_and_return_failure_proposes_second_stage_without_target_answer(tmp_path):
    common, *_ = scene(tmp_path)
    _, parent_ref, study_ref, returned = returned_fit_with_counterexample(tmp_path, common)
    old = Path(parent_ref['file']).read_bytes()
    actual = route(common, allowance=10000)
    original, followup = actual['method_action_research']['transfer_partition_proposals']
    assert actual['status'] == 'method_content_proposed_pending_validation'
    assert original['status'] == 'local_development_improvement'
    assert followup['followup_of'] == parent_ref
    bundle = read_method_reference(followup['proposal_ref'], common.workspace)
    candidate = bundle['proposal']['candidates'][0]['research_program']
    baseline = bundle['proposal']['baseline_research_program']
    assert candidate['transfer_partition']['second_stage'] == {
        'operation': 'resolve_relation_intervals',
        'trigger': 'fitted_training_with_unexplained_return',
        'evidence_role': 'paid_return_after_first_fit', 'compute_limit': 15000}
    without_stage = deepcopy(candidate)
    del without_stage['transfer_partition']['second_stage']
    assert without_stage == baseline
    assert candidate['selection_steps'][0]['parameters'][0]['radius'] == .4
    assert study_ref in bundle['source_refs'] and returned['return_ref'] in bundle['source_refs']
    assert not bundle['independent_support'] and bundle['installed_candidate'] is None
    assert Path(parent_ref['file']).read_bytes() == old
    assert actual['method_action_research']['relative_labels_created'] == 0


def test_saved_followup_restores_without_requoting_or_recreating(tmp_path, monkeypatch):
    common, *_ = scene(tmp_path)
    owner, *_ = returned_fit_with_counterexample(tmp_path, common)
    first = route(common, allowance=10000)
    frozen = export_python_method_topic(common, owner)
    restored = CommonObjectDependencyMethods(tmp_path / 'continued', [],
        budget=SearchBudget(10000), actions=[]).common
    question = ensure_python_method_topic(restored, parent_question_ref={'question_id': 'task'}, topic_ref=frozen)
    question.pop('routing_execution', None)
    from hegel_machine.cognition import method_action_research
    monkeypatch.setattr(method_action_research, 'quote_partition_followup_analysis', forbidden)
    resumed = route(restored, allowance=10000)
    assert resumed['new_compute_cost'] == 2
    assert resumed['method_action_research']['transfer_partition_proposals'] == (
        first['method_action_research']['transfer_partition_proposals'])
    assert len(question['method_use_plan']['native_return_refs']) == 1
    assert len(question['method_use_plan']['transfer_partition_proposal_refs']) == 2


def test_unexplained_training_does_not_relabel_itself_as_boundary_identification(tmp_path):
    common, *_ = scene(tmp_path)
    returned_fit_with_counterexample(tmp_path, common, training_loss=.02)
    actual = route(common, allowance=10000)
    assert actual['status'] == 'complete_native_partition_development_review'
    assert len(actual['method_action_research']['transfer_partition_proposals']) == 1
