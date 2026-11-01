"""Learn a sequential dependency from paid execution, not parallel preference."""

from copy import deepcopy
from pathlib import Path

import pytest

from hegel_machine.cognition.common_object_research import record_native_parent_research_return
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.research_programs import replay_research_program
from hegel_machine.cognition.staged_method_research import (
    record_staged_method_practice, revise_staged_method_conditions)
from hegel_machine.cognition.transfer_partitions import (
    prepare_parent_parameter_revision, quote_transfer_partition, replay_transfer_partition)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_native_method_return_route import route, save
from test_native_partition_proposals import scene, forbidden
from test_parent_bound_staged_replay import fixture
from test_staged_condition_revision import delivered_stage_pair


def composition_source(tmp_path, *, producer=True, pool_winner=True, complete=True, produced=True, alias=False):
    common, _, evidence_ref, _, _, _ = scene(tmp_path, allowance=1000000)
    evidence = read_method_reference(evidence_ref, common.workspace)
    staged, training, feedback = fixture()
    baseline, continuation = staged['staged_research']['baseline'], staged['staged_research']['continuation']
    parent = replay_transfer_partition(baseline, training, SearchBudget(100000))
    parent_ref = save(tmp_path / 'actual_parent.json', encode_entry(parent))
    source_ref = save(tmp_path / 'source_pi1.json', continuation)
    (tmp_path / 'producer').mkdir()
    producer_ref = save(tmp_path / 'producer/outcome.json', {'execution_complete': True,
        'full_comparison': True, 'selected_ref': parent_ref if pool_winner else {'file': 'better_old_pool.json'}})
    delivered_ref = parent_ref
    if alias:
        delivered_ref = {**save(tmp_path / 'producer/new_entries.json', [encode_entry(parent)]),
                         'json_path': [0]}
    if not produced:
        from hegel_machine.cognition.models import Candidate
        from hegel_machine.cognition.structural_research_program import research_program_fit_contract
        different = {'candidate': Candidate('other', research_program_fit_contract(baseline)['program'], 'new'),
                     'research_program': baseline}
        delivered_ref = save(tmp_path / 'producer/different_new_entry.json', encode_entry(different))
    save(tmp_path / 'producer/comparison.json', {'full_comparison': True,
        'new_scores': [{'candidate_ref': delivered_ref, 'loss': parent['training_loss']}]})
    method = prepare_parent_parameter_revision(parent, continuation,
        source_method_ref=source_ref, parent_ref=parent_ref)['research_program']
    quote = quote_transfer_partition(method, training, feedback_rows=feedback)
    actual = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))._revise_paid_structure(
        TaskSpec('composition', '', tuple(training[0].features), (), ()),
        parent, training + feedback, SearchBudget(quote['required_compute'] + 10000), [],
        finalists=[parent], paid_scores={parent['candidate'].program: loss(parent['candidate'].program,
            training + feedback, 0., .5)}, research_action='relation_parameter_revision', research_method=method,
        fit_rows=training, feedback_rows=feedback,
        paid_initial={'entry': parent, 'training': training, 'risk_quantile': .5, 'cost_ref': producer_ref})
    assert actual['full_comparison'] and actual['execution_complete']
    action = deepcopy(actual['action'])
    if not complete:
        action['search_complete'] = action['research']['execution_complete'] = False
    save(Path(evidence['after_entry_ref']['file']), encode_entry(actual['new_claims'][0]))
    evidence.update(research_parent_ref=parent_ref,
        executed_method_ref=save(tmp_path / 'method.json', method),
        method_execution_ref=save(tmp_path / 'action.json', action))
    save(Path(evidence_ref['file']), evidence)
    if producer:
        # Reuse the historical paired-return fixture only for producer identity.
        # Its outcome preference must never become a sequence training label.
        group = delivered_stage_pair(common, tmp_path / 'producer_pair', staged, .3)
        artifact = read_method_reference(group['baseline_method_artifact_ref'], common.workspace)
        save(Path(artifact['execution']['research_program_ref']['file']), baseline)
        pair = read_method_reference(group['pair_refs'][0], common.workspace)
        saved = read_method_reference(pair['baseline_ref'], common.workspace)
        save(Path(pair['baseline_ref']['file']), {**saved, 'actual_outcome_ref': producer_ref})
    return common, evidence, baseline, continuation, parent


@pytest.mark.parametrize(('pool_winner', 'alias'), [(True, False), (False, False), (True, True)])
def test_ordinary_review_forms_composition_using_producer_input_not_returned_graph(tmp_path, monkeypatch, pool_winner, alias):
    common, evidence, baseline, continuation, parent = composition_source(tmp_path,
        pool_winner=pool_winner, alias=alias)
    assert baseline != parent['research_program']
    from hegel_machine.cognition import research_programs, transfer_partitions
    monkeypatch.setattr(research_programs._Fitter, 'fit', forbidden)
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', forbidden)
    result = route(common, allowance=1000000)['method_action_research']
    existing = next(item for item in result['transfer_partition_proposals']
        if item['status'] == 'existing_method_execution_returned')
    assert read_method_reference(existing['proposal_ref'], common.workspace)['proposal']['candidates'] == []
    proposed = next(item for item in result['staged_method_proposals'] if item.get('candidate_count') == 1)
    bundle = read_method_reference(proposed['proposal_ref'], common.workspace)
    body = read_method_reference(proposed['research_program_refs'][0], common.workspace)
    stage = body['staged_research']
    assert stage['operation'] == 'revise_then_reidentify'
    assert stage['baseline'] == baseline and stage['continuation'] == continuation
    assert stage['condition_origin'] == 'unlearned_append_candidate'
    assert stage['condition'] == Program('const', value=1.).to_dict()
    assert bundle['basis']['research_parent_ref'] == evidence['research_parent_ref']
    assert bundle['baseline_evidence'] == [] and bundle['research']['preferences'] == []
    assert not bundle['source_support_inherited'] and not bundle['independent_support']
    assert proposed['paired_labels_created'] == 0
    paid = common.costs['compute_cost']
    again = route(common, allowance=1000000)
    assert again['new_compute_cost'] == 0 and common.costs['compute_cost'] == paid


@pytest.mark.parametrize(('kwargs', 'reason'), [
    ({'producer': False}, 'actual_baseline_execution_input_missing'),
    ({'produced': False}, 'actual_baseline_execution_input_missing'),
    ({'complete': False}, 'complete_parent_bound_research_required'),
])
def test_missing_or_incomplete_dependency_stays_unresolved(tmp_path, kwargs, reason):
    common, *_ = composition_source(tmp_path, **kwargs)
    reviewed = route(common, allowance=1000000)['method_action_research']
    unresolved = [item for item in reviewed['staged_method_proposals'] if item.get('reason') == reason]
    assert len(unresolved) == 1 and unresolved[0]['candidate_count'] == 0


def test_existing_verified_return_forms_new_composition_without_rechecking_old_research(tmp_path, monkeypatch):
    common, evidence, *_ = composition_source(tmp_path)
    from hegel_machine.cognition import method_action_research, staged_method_research
    native = staged_method_research.propose_native_method_compositions
    monkeypatch.setattr(staged_method_research, 'propose_native_method_compositions', lambda *args: [])
    first = route(common, allowance=1000000)['method_action_research']
    previous = next(item['proposal_ref'] for item in first['transfer_partition_proposals']
                    if item['status'] == 'existing_method_execution_returned')
    saved = Path(previous['file']).read_bytes()
    monkeypatch.setattr(staged_method_research, 'propose_native_method_compositions', native)
    monkeypatch.setattr(method_action_research, '_existing_method_execution_return', forbidden)
    current = save(tmp_path / 'new_complete_method_return.json', {'status': 'unresolved',
        'execution_complete': True})
    record_native_parent_research_return(common, source_task_ref=evidence['source_task_ref'],
        result_ref=current, mission={'purpose': 'method_learning', 'allocation': {'compute': 1000000}})
    result = route(common, allowance=1000000)['method_action_research']
    proposed = next(item for item in result['staged_method_proposals'] if item.get('candidate_count') == 1)
    assert read_method_reference(proposed['proposal_ref'], common.workspace)['source_execution_return_ref'] == previous
    assert Path(previous['file']).read_bytes() == saved


def test_new_content_keeps_its_agenda_when_an_older_staged_revision_is_reused(tmp_path, monkeypatch):
    from test_staged_condition_revision import setup
    from hegel_machine.cognition import staged_method_research
    older = tmp_path / 'older'
    older.mkdir()
    old_agent, old_owner, old_groups, *_ = setup(older)
    saved_revision = revise_staged_method_conditions(old_agent, old_owner, old_groups, 1000)
    assert saved_revision['revision_ref'] and saved_revision['status'] == 'staged_conditions_revised'
    current = tmp_path / 'current'
    current.mkdir()
    common, *_ = composition_source(current)
    cached = {**saved_revision, 'new_compute_cost': 0, 'reused': True}
    monkeypatch.setattr(staged_method_research, 'revise_staged_method_conditions', lambda *args: cached)
    result = route(common, allowance=1000000)['method_action_research']
    assert result['staged_method_revision'] == cached
    assert any(item.get('candidate_count') == 1 for item in result['staged_method_proposals'])
    assert result['status'] == 'method_content_proposed_pending_validation'
    assert result['required_work'] == 'execute_staged_method_from_same_start_then_freeze_before_new_source_validation'


@pytest.mark.parametrize('complete', [True, False])
def test_actual_proposal_practice_returns_to_same_topic_without_repeating_research(tmp_path, monkeypatch, complete):
    common, evidence, _, _, parent = composition_source(tmp_path)
    reviewed = route(common, allowance=1000000)['method_action_research']
    proposed = next(item for item in reviewed['staged_method_proposals'] if item.get('candidate_count') == 1)
    body_ref = proposed['research_program_refs'][0]
    body = read_method_reference(body_ref, common.workspace)
    _, training, feedback = fixture()
    quote = quote_transfer_partition(body, training, feedback_rows=feedback)
    executed = replay_research_program(body, training, SearchBudget(quote['required_compute']),
        feedback_rows=feedback, tolerance=1e-7)
    actual = executed['staged_research_result']
    assert actual['execution_complete']
    # Evaluate both entries of this small retained pool on the same rows.
    scores = [loss(item['candidate'].program, training + feedback, 0., .5)
              for item in (parent, executed)]
    root = tmp_path / 'practice'
    root.mkdir()
    save(root / 'registration.json', {'method_source_ref': body_ref,
        'method_ref': save(root / 'method.json', body),
        'sources': [{'source_task_ref': evidence['source_task_ref'], 'pool_size': 1}]})
    practice_ref = save(root / 'receipt.json', {'status': 'paid_stage_paths_completed'
        if complete else 'paid_stage_path_incomplete', 'sources': [{
            'source_task_ref': evidence['source_task_ref'], 'old_pool_size': 1,
            'execution_complete': complete, 'full_comparison': len(scores) == 2 and complete,
            'scores': scores, 'staged_research_result': actual}]})
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    result = record_staged_method_practice(common, owner, proposal_ref=proposed['proposal_ref'],
        practice_ref=practice_ref, compute_limit=3)
    assert result['execution_complete'] is complete and result['new_compute_cost'] == 3
    from hegel_machine.cognition import research_programs
    monkeypatch.setattr(research_programs._Fitter, 'fit', forbidden)
    current = route(common, allowance=1000000)['method_action_research']
    item = next(item for item in current['staged_method_proposals']
                if item.get('proposal_ref') == proposed['proposal_ref'])
    assert item['status'] == ('staged_method_path_verified' if complete else 'staged_method_path_incomplete')
    assert not item['independent_support'] and not item['general_guard_supported']
    assert current['required_work'] == ('assess_new_source_validation_for_verified_composition'
        if complete else 'resolve_retained_incomplete_staged_study')
    assert record_staged_method_practice(common, owner, proposal_ref=proposed['proposal_ref'],
        practice_ref=practice_ref, compute_limit=0)['new_compute_cost'] == 0


def measured_composition_pair(tmp_path, *, attempts=1, missing=False, corrupt=False, reused=False):
    common, *_ = scene(tmp_path, allowance=1000000)
    body, training, feedback = fixture()
    body['staged_research']['condition'] = Program('const', value=1.).to_dict()
    body['staged_research']['continuation']['transfer_partition']['max_attempts'] = attempts
    quote = quote_transfer_partition(body, training, feedback_rows=feedback)
    entry = replay_research_program(body, training, SearchBudget(quote['required_compute']),
        feedback_rows=feedback, tolerance=1e-7)
    actual = deepcopy(entry['staged_research_result'])
    assert actual['execution_complete'] and actual['continuation']['execution_complete']
    assert actual['baseline']['all_paid_loss'] > 1e-7
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    proposal = save(tmp_path / 'proposal.json', {'basis': {'operation': 'revise_then_reidentify'},
        'baseline_evidence': [], 'research': {'preferences': []},
        'proposal': {'baseline_research_program': body['staged_research']['baseline'], 'candidates': [
            {'research_program': body, 'condition_source': {'predicted_returns': []}}]}})
    owner['method_use_plan']['staged_method_proposal_refs'] = [proposal]
    group = delivered_stage_pair(common, tmp_path / 'paired', body, actual['baseline']['all_paid_loss'])
    artifact = read_method_reference(group['baseline_method_artifact_ref'], common.workspace)
    save(Path(artifact['execution']['research_program_ref']['file']), body['staged_research']['baseline'])
    pair = read_method_reference(group['pair_refs'][0], common.workspace)
    outcome = read_method_reference(pair['challenger_ref'], common.workspace)
    raw_ref = outcome['actual_outcome_ref']
    if missing:
        actual.pop('continuation')
    elif corrupt:
        actual['continuation']['replay_inputs']['relation_binding']['parameters'][0]['radius'] += .1
    elif reused:
        actual['continuation'].update(reused=True, compute_cost=0,
            original_cost_ref={'file': 'original-scientific-cost.json'})
    save(Path(raw_ref['file']), {**read_method_reference(raw_ref, common.workspace),
        'development_rows': len(training) + len(feedback), 'staged_research_result': actual})
    # The parallel fixed arm misleadingly has a perfect new score. It must
    # never supply the continuation outcome or cost for this sequence.
    return common, owner, [group], actual


@pytest.mark.parametrize(('attempts', 'status'), [(1, 'staged_conditions_exhausted'),
                                                (48, 'staged_conditions_retained')])
def test_revision_uses_real_same_run_continuation_not_parallel_pi1(tmp_path, attempts, status):
    common, owner, groups, actual = measured_composition_pair(tmp_path, attempts=attempts)
    result = revise_staged_method_conditions(common, owner, groups, 1000)
    assert result['status'] == status
    bundle = read_method_reference(result['revision_ref'], common.workspace)
    measured, = bundle['paid_stage_evidence']
    assert measured['actual_parent_bound']
    assert measured['incremental_costs']['append_continuation']['compute_cost'] == actual['continuation']['compute_cost']
    assert measured['continuation_score_ref']['json_path'][-3:] == ['staged_research_result', 'continuation', 'all_paid_loss']
    assert result['new_predictions'] == result['new_fits'] == 0
    assert not result['general_guard_supported'] and not result['condition_resynthesis']
    assert revise_staged_method_conditions(common, owner, groups, 0)['new_compute_cost'] == 0


@pytest.mark.parametrize('change', ['missing', 'corrupt', 'reused'])
def test_unexecuted_or_wrong_parent_stage_cannot_borrow_parallel_success(tmp_path, change):
    common, owner, groups, _ = measured_composition_pair(tmp_path, **{change: True})
    result = revise_staged_method_conditions(common, owner, groups, 1000)
    assert result['status'] == 'staged_condition_evidence_incomplete'
    assert result['usable_sources'] == 0 and result['refuted_indices'] == []
