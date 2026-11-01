"""Restore actual object-record semantics without inventing missing history."""

from dataclasses import asdict
from pathlib import Path

import pytest

from scripts import review_paid_object_pool as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition import parallel_prediction
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry


def records():
    return [
        {'probe_id': 'first', 'position': 0, 'controls': {'c': 1.}, 'history': [],
         'before': {'m1_x': 1.}, 'after': {'m1_x': 101.}, 'target': 1.},
        {'probe_id': 'first', 'position': 1, 'controls': {'c': 2.}, 'history': [{'c': 1.}],
         'before': {'m1_x': 2.}, 'after': {'m1_x': 102.}, 'target': 2.},
        {'probe_id': 'reset', 'position': 0, 'controls': {'c': 5.}, 'history': [],
         'before': {'m1_x': 5.}, 'after': {'m1_x': 105.}, 'target': 5.},
    ]


def public():
    return TaskSpec('paid', 'Saved anonymous sensors', ('m1_x',), (), ())


def test_before_sensor_history_target_and_probe_reset_are_preserved():
    rows, probes = runner.restore_object_rows(records(), public())
    assert [row.features for row in rows] == [{'m1_x': 1.}, {'m1_x': 2.}, {'m1_x': 5.}]
    assert rows[0].history == rows[2].history == ()
    assert rows[1].history == ({'m1_x': 1., '__target__': 1.},)
    assert probes == [{'probe_id': 'first', 'start': 0, 'rows': 2},
                      {'probe_id': 'reset', 'start': 2, 'rows': 1}]


def test_incomplete_prefix_does_not_become_zero_filled_sensor_history():
    with pytest.raises(ValueError, match='complete observed prefix'):
        runner.restore_object_rows(records()[1:], public())


def test_mixed_control_prefix_and_changed_sensor_interface_are_rejected():
    wrong = records()
    wrong[1]['history'] = []
    with pytest.raises(ValueError, match='control prefix'):
        runner.restore_object_rows(wrong, public())
    wrong = records()
    wrong[0]['before'] = {'other_sensor': 1.}
    with pytest.raises(ValueError, match='sensor view'):
        runner.restore_object_rows(wrong, public())


def scene(tmp_path, monkeypatch, *, available=50):
    output = tmp_path / 'review'
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args: None)
    monkeypatch.setattr(runner, 'write_text_incrementally', lambda *args, **kwargs: None)
    save(tmp_path / 'paid.json', records())
    save(tmp_path / 'public.json', asdict(public()))
    save(tmp_path / 'ledger.json', {'remaining_compute': 50})
    programs = [Program('feature', name='m1_x'), Program('feature', name='m1_x'),
                Program('const', value=0.)]
    refs = []
    for index, program in enumerate(programs):
        location = tmp_path / f'candidate{index}.json'
        save(location, encode_entry({'candidate': Candidate(f'old{index}', program, 'retained'),
            'training_loss': 0., 'validation_loss': 0.}))
        refs.append(runner.ref(location))
    cfg = {'paid_records_ref': runner.ref(tmp_path / 'paid.json'),
        'public_spec_ref': runner.ref(tmp_path / 'public.json'), 'candidate_refs': refs,
        'ledger_ref': runner.ref(tmp_path / 'ledger.json', 'remaining_compute'),
        'available_compute': available, 'source_cost_refs': [runner.ref(tmp_path / 'ledger.json')],
        'quality_threshold': 1e-10, 'runtime_source': str(tmp_path / 'old')}
    request = tmp_path / 'request.json'
    save(request, cfg)
    return request, output


def test_all_references_retained_first_readout_charged_once_per_program(tmp_path, monkeypatch):
    request, output = scene(tmp_path, monkeypatch)
    cfg = runner.register(request, output)
    assert cfg['maximum_compute'] == 6
    assert cfg['candidate_occurrences'] == 3 and cfg['unique_programs'] == 2
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    calls, predict = [], parallel_prediction.predict_rows
    monkeypatch.setattr(parallel_prediction, 'predict_rows',
        lambda program, rows: (calls.append(len(rows)) or predict(program, rows)))
    result = runner.run(output)
    assert calls == [3, 3]
    assert result['selected_mse'] == 0. and result['status'] == 'paid_pool_already_explains'
    assert len(result['scores'][0]['candidate_refs']) == 2
    assert sum(len(item['candidate_refs']) for item in result['scores']) == 3
    assert result['scores'][1]['mse'] == 10.  # source validation_loss=0 is not reused
    assert result['compute_cost'] == 6 and result['remaining_compute'] == 44
    assert result['new_fits'] == result['new_observations'] == result['method_labels_created'] == 0
    assert not result['independent_support']
    assert runner.run(output)['new_compute_cost'] == 0 and calls == [3, 3]


def test_quote_cannot_increase_or_exceed_original_balance(tmp_path, monkeypatch):
    request, output = scene(tmp_path, monkeypatch, available=5)
    with pytest.raises(ValueError, match='complete paid pool'):
        runner.register(request, output)
    cfg = load(request)
    cfg['available_compute'] = 51
    save(request, cfg)
    with pytest.raises(ValueError, match='increase'):
        runner.register(request, output)
    assert not output.exists()


def test_prediction_failure_retains_cost_and_completed_scores(tmp_path, monkeypatch):
    request, output = scene(tmp_path, monkeypatch)
    runner.register(request, output)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    predict = parallel_prediction.predict_rows

    def fail(program, rows):
        if program.op == 'const':
            raise RuntimeError('actual readout interrupted')
        return predict(program, rows)

    monkeypatch.setattr(parallel_prediction, 'predict_rows', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        runner.run(output)
    failure = load(output / 'failure.json')
    assert failure['compute_cost'] == 6 and len(failure['completed_scores']) == 1
    assert not failure['full_comparison'] and not (output / 'receipt.json').exists()


def relation_scene(tmp_path, monkeypatch):
    from hegel_machine.cognition import transfer_partitions

    request, pool = scene(tmp_path, monkeypatch)
    paid = records()
    for record in paid:
        record['target'] += 1.
    save(tmp_path / 'paid.json', paid)
    runner.register(request, pool)
    monkeypatch.setattr(runner, 'ROOT', pool / 'runtime')
    runner.run(pool)
    method = {'transfer_partition': {'kind': 'Pi1', 'max_attempts': 48,
        'max_candidates': 12, 'second_stage': {'operation': 'resolve_relation_intervals',
            'compute_limit': 15000}}}
    save(tmp_path / 'method.json', {'P0': method})
    method_ref = runner.ref(tmp_path / 'method.json', 'P0')
    monkeypatch.setattr(transfer_partitions, 'quote_transfer_partition',
        lambda body, training, **kwargs: {'status': 'quoted', 'required_compute': 9,
            'followup_compute_upper': 5, 'fit_rows': len(training), 'attempt_limit': 48})
    relation_request = tmp_path / 'relation_request.json'
    save(relation_request, {'pool': str(pool), 'method_ref': method_ref,
        'ledger_ref': runner.ref(tmp_path / 'ledger.json', 'remaining_compute'),
        'available_compute': 50, 'method_compute_limit': 15, 'active_wall_seconds': 30,
        'source_cost_refs': [runner.ref(pool / 'receipt.json')]})
    output = tmp_path / 'relation'
    monkeypatch.setattr(runner, 'ROOT', Path(__file__).resolve().parents[1])
    return relation_request, output, method


def test_relation_quotes_full_body_and_compares_only_new_programs(tmp_path, monkeypatch):
    from hegel_machine.cognition import research_programs, unified_research

    request, output, method = relation_scene(tmp_path, monkeypatch)
    cfg = runner.register_relation(request, output)
    assert cfg['training_rows'] == 2 and cfg['feedback_rows'] == 1
    assert cfg['total_compute_upper'] == 27 and cfg['equal_future_method_compute_limit'] == 15
    assert cfg['max_candidates'] == 4 and cfg['method_quote']['attempt_limit'] == 48

    def replay(body, training, budget, *, feedback_rows, **kwargs):
        assert body == method  # internal 12 retained; no mutation to mimic the public four-draft cap
        assert len(training) == 2 and len(feedback_rows) == 1
        assert not feedback_rows[0].history
        assert budget.spend(7)
        program = Program('add', children=(Program('feature', name='m1_x'), Program('const', value=1.)))
        return {'candidate': Candidate('new', program, 'replayed'), 'training_loss': 0.,
            'transfer_partition_search': {'completed': True, 'attempts': 48},
            'transfer_partition_followup': {'execution_complete': True, 'status': 'paid_interval_supported'}}

    monkeypatch.setattr(research_programs, 'replay_research_program', replay)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    calls, loss = [], unified_research.loss
    monkeypatch.setattr(unified_research, 'loss',
        lambda program, rows, *args: (calls.append(len(rows)) or loss(program, rows, *args)))
    result = runner.run_relation(output)
    assert calls == [3] and result['comparison_compute_cost'] == 3
    assert result['old_candidate_occurrences'] == 3 and result['new_candidates'] == 1
    assert result['selected_loss'] == 0. and result['prior_selected_loss'] == 1.
    assert result['status'] == 'relation_supported' and result['execution_complete']
    assert result['compute_cost'] == 10 and result['remaining_compute'] == 40
    assert result['method_labels_created'] == result['new_observations'] == 0
    assert runner.run_relation(output)['new_compute_cost'] == 0 and calls == [3]


def test_followup_not_applicable_is_distinct_from_incomplete_compute(tmp_path, monkeypatch):
    from hegel_machine.cognition import research_programs

    request, output, _ = relation_scene(tmp_path, monkeypatch)
    runner.register_relation(request, output)

    def replay(body, training, budget, **kwargs):
        assert budget.spend(7)
        return {'candidate': Candidate('same', Program('feature', name='m1_x'), 'replayed'),
            'training_loss': 1., 'transfer_partition_search': {'completed': True, 'attempts': 48},
            'transfer_partition_followup': {'execution_complete': False,
                'status': 'first_stage_training_unexplained'}}

    monkeypatch.setattr(research_programs, 'replay_research_program', replay)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_relation(output)
    assert result['status'] == 'relation_followup_not_applicable'
    assert result['bounded_first_search_complete'] and not result['execution_complete']
    assert not result['budget_denied'] and result['full_comparison']
    assert result['comparison_compute_cost'] == 0 and result['compute_cost'] == 7
    assert result['method_labels_created'] == 0


@pytest.mark.parametrize('produce_candidate', [False, True])
def test_structure_uses_original_pool_partial_target_and_source_only_as_candidate(
        tmp_path, monkeypatch, produce_candidate):
    from hegel_machine.cognition import research_programs, unified_research

    request, relation, _ = relation_scene(tmp_path, monkeypatch)
    runner.register_relation(request, relation)

    def replay(body, training, budget, **kwargs):
        budget.spend(7)
        return {'candidate': Candidate('same', Program('feature', name='m1_x'), 'replayed'),
            'training_loss': 1., 'transfer_partition_search': {'completed': True, 'attempts': 48},
            'transfer_partition_followup': {'execution_complete': False,
                'status': 'first_stage_training_unexplained'}}

    monkeypatch.setattr(research_programs, 'replay_research_program', replay)
    monkeypatch.setattr(runner, 'ROOT', relation / 'runtime')
    original_return = runner.run_relation(relation)
    target = {'schema': None, 'complete': False, 'raw_sensor_features': ['m1_x'],
        'raw_control_features': [], 'groups': []}
    source = {**target, 'schema': {'old_assignment': 'candidate_only'}, 'complete': True}
    save(tmp_path / 'target_model.json', target)
    save(tmp_path / 'source_model.json', source)
    request = tmp_path / 'structure_request.json'
    save(request, {'relation': str(relation), 'ledger_ref': runner.ref(relation / 'receipt.json', 'remaining_compute'),
        'available_compute': 43, 'active_wall_seconds': 30,
        'target_model_ref': runner.ref(tmp_path / 'target_model.json'),
        'source_model_ref': runner.ref(tmp_path / 'source_model.json'),
        'source_cost_refs': [runner.ref(relation / 'receipt.json')]})
    output = tmp_path / 'structure'
    monkeypatch.setattr(runner, 'ROOT', Path(__file__).resolve().parents[1])
    cfg = runner.register_structure(request, output)
    assert cfg['method_compute_limit'] == 15 and cfg['comparison_compute_reserve'] == 12

    def search(agent, spec, training, budget, *, actions, parent, new_observations, transitions, tolerance):
        assert spec.context == {'object_model': target}
        assert agent.library.object_models == [source]
        assert agent.config.max_candidates == 4 and budget.limit == 15
        assert len(training) == len(transitions) == 2 and len(new_observations) == 1
        assert transitions[0] == {'before': {'m1_x': 1.}, 'after': {'m1_x': 101.}}
        assert parent['candidate'].program == Program('feature', name='m1_x')
        budget.spend(5)
        agent._object_competitions[tuple(spec.feature_names)] = {
            'status': 'supported' if produce_candidate else 'no_supported_program',
            'searches': [{'schema_index': 0, 'search_complete': False, 'compute_cost': 5}]}
        if not produce_candidate:
            return []
        program = Program('add', children=(Program('feature', name='m1_x'), Program('const', value=1.)))
        return [{'candidate': Candidate('structure_new', program, 'native'), 'training_loss': 0.}]

    monkeypatch.setattr(unified_research.UnifiedResearchAgent, '_object_explanations', search)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_structure(output)
    assert result['invocation_complete'] and not result['execution_complete']
    assert result['old_candidate_occurrences'] == 3 and result['full_comparison']
    assert result['method_compute_cost'] == 5
    assert result['comparison_compute_cost'] == (3 if produce_candidate else 0)
    assert result['selected_loss'] == (0. if produce_candidate else 1.)
    assert result['relation_return_scope']['followup_applicability'] == 'refuted_by_paid_training'
    assert not result['relation_return_scope']['two_segment_execution_complete']
    assert load(output / 'competition.json')['searches'][0]['search_complete'] is False
    assert load(relation / 'receipt.json') == original_return
    assert result['method_labels_created'] == result['new_observations'] == 0
    assert runner.run_structure(output)['new_compute_cost'] == 0
    if not produce_candidate:
        # Continue the same Common owner; incomplete returns remain UNKNOWN.
        from hegel_machine.cognition.common_object_research import (
            CommonObjectDependencyMethods, record_native_parent_research_return)
        from hegel_machine.cognition.models import SearchBudget

        prior = tmp_path / 'prior_review'
        common = CommonObjectDependencyMethods(prior / 'common', [],
            budget=SearchBudget(100), actions=[]).common
        source_ref = runner.ref(tmp_path / 'physical_task.json')
        save(tmp_path / 'physical_task.json', {'seed': 123})
        old_ref = runner.ref(tmp_path / 'old_native_return.json')
        save(tmp_path / 'old_native_return.json', {'status': 'unresolved'})
        record_native_parent_research_return(common, source_task_ref=source_ref,
            result_ref=old_ref, mission={'purpose': 'method_learning', 'allocation': {'compute': 100}})
        save(prior / 'receipt.json', {'compute_cost': 0})
        questions_before = load(prior / 'common/questions.json')
        before = next(item for item in questions_before if item['id'] == 'task:method_choice')
        request = tmp_path / 'return_request.json'
        save(request, {'structure': str(output), 'prior_review': str(prior),
            'source_task_ref': source_ref, 'maximum_compute': 30})
        returned = runner.return_choice(request, tmp_path / 'returned')
        assert returned['new_fits'] == returned['new_predictions'] == returned['new_observations'] == 0
        assert returned['compute_cost'] <= 30
        assert returned['current_pair_preferences'][0]['preference'] == 'UNKNOWN'
        assert returned['current_pair_preferences'][0]['reason'] == 'non_delivery_or_censoring_is_not_a_prediction_label'
        continued = load(tmp_path / 'returned/method_return/common/questions.json')
        owner = next(item for item in continued if item['id'] == before['id'])
        assert owner['meaning_version'] == before['meaning_version']
        assert owner['method_use_plan']['native_return_refs'][:1] == before['method_use_plan']['native_return_refs']
        assert len(owner['method_use_plan']['native_return_refs']) == 2
        assert load(prior / 'common/questions.json') == questions_before
        assert load(relation / 'receipt.json') == original_return
        assert load(output / 'receipt.json') == result
        assert runner.return_choice(request, tmp_path / 'returned')['new_compute_cost'] == 0


def test_structure_cannot_start_on_unfinished_relation(tmp_path, monkeypatch):
    request, relation, _ = relation_scene(tmp_path, monkeypatch)
    runner.register_relation(request, relation)
    save(relation / 'receipt.json', {'status': 'relation_unresolved',
        'bounded_first_search_complete': False, 'budget_denied': True, 'full_comparison': True})
    request = tmp_path / 'structure_request.json'
    save(request, {'relation': str(relation)})
    with pytest.raises(ValueError, match='actual completed'):
        runner.register_structure(request, tmp_path / 'structure')


def parent_scene(tmp_path, monkeypatch):
    request, pool = scene(tmp_path, monkeypatch)
    paid = records()
    for row in paid:
        row['target'] += 1.
    save(tmp_path / 'paid.json', paid)
    runner.register(request, pool)
    monkeypatch.setattr(runner, 'ROOT', pool / 'runtime')
    runner.run(pool)
    source = tmp_path / 'parent_source'
    source.mkdir()
    program = Program('add', children=(Program('feature', name='m1_x'), Program('const', value=.5)))
    parent = {'candidate': Candidate('paid_relation', program, 'native'),
        'training_loss': .25, 'validation_loss': .25,
        'research_program': {'kind': 'retained', 'term_nodes': ['complete_column']}}
    save(source / 'relation/new_entries.json', [encode_entry(parent)])
    save(source / 'relation/selected_entry.json', encode_entry(parent))
    rows, probes = runner.restore_object_rows(paid, public())
    save(source / 'relation/registration.json', {'training_rows': 2, 'feedback_rows': 1,
        'comparison_rows': 3, 'probes': probes, 'method_compute_limit': 65000,
        'comparison_compute_reserve': 12, 'active_wall_seconds': 30})
    save(source / 'relation/comparison.json', {'full_comparison': True,
        'new_scores': [{'candidate_ref': runner.ref(source / 'relation/new_entries.json', 0), 'loss': .25}]})
    save(source / 'structure/comparison.json', {'full_comparison': True, 'new_scores': []})
    save(source / 'retained_frontier.json', {'full_comparison': True,
        'original_pool_ref': runner.ref(pool / 'receipt.json'),
        'appended_comparison_refs': [runner.ref(source / 'relation/comparison.json'),
            runner.ref(source / 'structure/comparison.json')],
        'complete_current_pool_occurrences': 4,
        'selected_ref': runner.ref(source / 'relation/selected_entry.json'), 'selected_mse': .25})
    save(source / 'choice_return/receipt.json', {'remaining_compute': 67000,
        'source_task_ref': runner.ref(tmp_path / 'physical_task.json')})
    save(source / 'source_costs.json', {'compute_cost': 100})
    request = tmp_path / 'parent_request.json'
    save(request, {'source': str(source)})
    monkeypatch.setattr(runner, 'ROOT', Path(__file__).resolve().parents[1])
    return request, source, parent


def test_parent_continuation_reuses_union_scores_and_real_fit_split(tmp_path, monkeypatch):
    from hegel_machine.cognition import unified_research
    request, source, expected = parent_scene(tmp_path, monkeypatch)
    output = tmp_path / 'continued'
    cfg = runner.register_parent(request, output)
    frontier_before = load(source / 'retained_frontier.json')
    assert cfg['prior_selected_loss'] == .25
    assert cfg['total_compute_upper'] == 65012 and cfg['available_compute'] == 67000

    def revised(agent, spec, parent, rows, budget, actions, *, finalists, paid_scores, fit_rows, research_action):
        assert len(finalists) == 4 and len(paid_scores) == 3
        assert parent is finalists[-1] and parent['research_program'] == expected['research_program']
        assert len(fit_rows) == 2 and len(rows) == 3
        assert [row.target for row in rows] == [2., 3., 6.]
        assert research_action == 'relation_form_revision' and budget.limit == 65012
        assert paid_scores[parent['candidate'].program] == .25
        assert budget.spend(5)
        action = {'search_complete': True, 'reused_scores': len(paid_scores), 'old_scoring_compute': 0}
        actions.append(action)
        return {'status': 'retained', 'execution_complete': True, 'full_comparison': True,
            'new_claims': [], 'compared_claims': finalists, 'selected': parent, 'action': action}

    monkeypatch.setattr(unified_research.UnifiedResearchAgent, '_revise_paid_structure', revised)
    monkeypatch.setattr(parallel_prediction, 'predict_rows',
        lambda *args: pytest.fail('A completed old pool must not be scored again'))
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    receipt = runner.run_parent(output)
    assert receipt['selected_ref'] == frontier_before['selected_ref']
    assert receipt['selected_loss'] == .25 and receipt['old_candidate_occurrences'] == 4
    assert receipt['compute_cost'] == 5 and receipt['remaining_compute'] == 66995
    assert receipt['execution_complete'] and receipt['full_comparison']
    assert load(source / 'retained_frontier.json') == frontier_before
    assert not (output / 'selected_entry.json').exists()
    assert runner.run_parent(output)['new_compute_cost'] == 0


def test_parent_register_keeps_current_frontier_and_balance(tmp_path, monkeypatch):
    request, source, _ = parent_scene(tmp_path, monkeypatch)
    ledger = load(source / 'choice_return/receipt.json')
    ledger['remaining_compute'] = 65011
    save(source / 'choice_return/receipt.json', ledger)
    with pytest.raises(ValueError, match='unchanged remaining account'):
        runner.register_parent(request, tmp_path / 'unfunded')
    frontier = load(source / 'retained_frontier.json')
    frontier['complete_current_pool_occurrences'] = 3
    save(source / 'retained_frontier.json', frontier)
    with pytest.raises(ValueError, match='complete retained frontier'):
        runner.parent_inputs(source)


@pytest.mark.parametrize('confirmation', [False, True])
def test_sequential_parent_return_continues_topic_without_old_pair(tmp_path, monkeypatch, confirmation):
    from hegel_machine.cognition.common_object_research import (
        CommonObjectDependencyMethods, record_native_parent_research_return)
    from hegel_machine.cognition.models import SearchBudget

    request, source, parent = parent_scene(tmp_path, monkeypatch)
    output = tmp_path / 'continued'
    cfg = runner.register_parent(request, output)
    prior = Path(cfg['prior_method_review'])
    common = CommonObjectDependencyMethods(prior / 'common', [], budget=SearchBudget(50), actions=[]).common
    old_ref = runner.ref(tmp_path / 'old_return.json')
    save(Path(old_ref['file']), {'status': 'unresolved'})
    record_native_parent_research_return(common, source_task_ref=cfg['source_task_ref'], result_ref=old_ref,
        mission={'purpose': 'method_learning', 'allocation': {'compute': 50}})
    old_questions = load(prior / 'common/questions.json')
    save(output / 'comparison.json', {'full_comparison': True, 'new_scores': [],
        'selected_ref': cfg['parent_ref'], 'selected_loss': .25})
    actual = {'status': 'retained', 'execution_complete': False, 'full_comparison': True,
        'selected_ref': cfg['parent_ref'], 'selected_loss': .25, 'remaining_compute': 30,
        'comparison_ref': runner.ref(output / 'comparison.json'), 'compute_cost': 5}
    if confirmation:
        cfg['source_frontier_ref'] = cfg.pop('frontier_ref')
        save(output / 'registration.json', cfg)
        actual.pop('full_comparison')
        actual.pop('comparison_ref')
        actual.update(status='confirmation_failed',
            confirmation_ref=runner.ref(output / 'fixed_readout.json'))
        save(output / 'fixed_readout.json', {'arms': {'selected': {'passed': False, 'mse': .25}}})
    save(output / 'receipt.json', actual)
    returned = runner.return_parent(output, tmp_path / 'returned_parent', maximum_compute=30)
    assert not returned['method_pair_created'] and returned['revision_evidence_ref'] is None
    assert returned['new_observations'] == returned['new_fits'] == returned['new_predictions'] == 0
    assert returned['compute_cost'] <= 30
    assert load(prior / 'common/questions.json') == old_questions
    assert load(output / 'receipt.json') == actual
    after = next(q for q in load(tmp_path / 'returned_parent/common/questions.json') if q['id'] == 'task:method_choice')
    before = next(q for q in old_questions if q['id'] == 'task:method_choice')
    assert after['meaning_version'] == before['meaning_version']
    assert after['method_use_plan']['native_return_refs'][:1] == before['method_use_plan']['native_return_refs']
    assert len(after['method_use_plan']['native_return_refs']) == 2
    assert not (tmp_path / 'returned_parent/pair.json').exists()
    if confirmation:
        record = load(Path(returned['feedback']['feedback_ref']['file']))
        assert record['confirmation_ref'] == actual['confirmation_ref']
        assert record['status'] == 'confirmation_failed'
        assert 'method_pair_ref' not in record


@pytest.mark.parametrize('missing_question_checkpoint', [False, True])
def test_parent_return_resume_retains_paid_proposals_and_condition_groups_without_resetting_account(
        tmp_path, monkeypatch, missing_question_checkpoint):
    from copy import deepcopy
    from hegel_machine.cognition import question_routing
    from hegel_machine.cognition.common_object_research import (
        CommonObjectDependencyMethods, record_native_parent_research_return)
    from hegel_machine.cognition.models import SearchBudget

    request, _, _ = parent_scene(tmp_path, monkeypatch)
    source = tmp_path / 'completed_research'
    cfg = runner.register_parent(request, source)
    actual = {'status': 'retained', 'full_comparison': True, 'selected_ref': cfg['parent_ref'],
        'remaining_compute': 3000, 'compute_cost': 7}
    actual_ref = runner.ref(source / 'receipt.json')
    save(Path(actual_ref['file']), actual)
    prior, partial = Path(cfg['prior_method_review']), tmp_path / 'partial_return'
    common = CommonObjectDependencyMethods(prior / 'common', [], budget=SearchBudget(50), actions=[]).common
    feedback = record_native_parent_research_return(common, source_task_ref=cfg['source_task_ref'],
        result_ref=actual_ref, mission={'purpose': 'method_learning', 'allocation': {'compute': 50}})
    original_owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    owner = deepcopy(original_owner)
    saved_refs = {}
    for name in ('transfer_partition_proposal_refs', 'transfer_partition_return_refs',
                 'staged_method_proposal_refs'):
        location = runner.ref(partial / f'{name}.json')
        save(Path(location['file']), {'status': 'paid', 'kind': name})
        owner['method_use_plan'][name] = saved_refs[name] = [location]
    owner['method_content_proposal_refs'] = saved_refs['transfer_partition_proposal_refs']
    completed_prior = runner.ref(prior / 'receipt.json', 'return', 'research_result',
        'method_action_research', 'paired_condition_research', 0)
    completed_partial = runner.ref(partial / 'receipt.json', 'return', 'research_result',
        'method_action_research', 'paired_condition_research', 0)
    owner['method_use_plan']['paired_condition_research_refs'] = [completed_prior]
    owner.update(routing_execution={'old': True}, method_use_result={'old': True},
        completed_agenda_basis={'old': True})
    if missing_question_checkpoint:
        save(partial / 'previous_topic.json', {'question': owner, 'questions': [owner], 'methods': []})
    else:
        save(partial / 'common/questions.json', [owner])
    for directory, group in ((prior, 'first_group'), (partial, 'second_group')):
        save(directory / 'receipt.json', {'status': 'scoped_unknown',
            'source_return_ref': actual_ref, 'remaining_compute': 648, 'compute_cost': 2352,
            'return': {'research_result': {'method_action_research': {
                'paired_condition_research': [
                    {'method_ref': {'id': group}, 'budget_incomplete': False},
                    {'method_ref': {'id': 'unfinished'}, 'budget_incomplete': True}]}}}})
    protected = {path: path.read_bytes() for directory in (prior, partial, source)
                 for path in directory.rglob('*.json')}
    calls = []

    def consume(restored, returned, public_spec, *, compute_limit):
        calls.append(returned)
        current = next(q for q in restored.questions if q['id'] == 'task:method_choice')
        plan = current['method_use_plan']
        assert compute_limit == 648
        assert returned['status'] == 'reused_native_return'
        assert returned['feedback_ref'] == feedback['feedback_ref']
        assert plan['native_return_refs'] == original_owner['method_use_plan']['native_return_refs']
        assert plan['paired_condition_research_refs'] == [completed_prior, completed_partial]
        assert all(plan[name] == refs for name, refs in saved_refs.items())
        assert current['method_content_proposal_refs'] == owner['method_content_proposal_refs']
        assert all(key not in current for key in ('routing_execution', 'method_use_result',
                                                 'completed_agenda_basis'))
        # The all-reused path performs no internal question save. Its updated
        # topic must still survive the return boundary for the next episode.
        assert not (restored.workspace / 'questions.json').exists()
        current['method_use_result'] = {'status': 'cached_paid_method_review', 'new_compute_cost': 7}
        restored.costs['compute_cost'] += 7
        return {'status': 'method_content_proposed_pending_validation'}

    monkeypatch.setattr(question_routing, 'consume_method_return_agenda', consume)
    output = tmp_path / 'continued_return'
    result = runner.return_parent(source, output, previous_return=partial)
    expected_ledger = runner.ref(partial / 'receipt.json', 'remaining_compute')
    assert result['ledger_ref'] == expected_ledger
    assert result['remaining_compute'] == 641 and result['compute_cost'] == 7
    assert load(output / 'registration.json')['maximum_compute'] == 648
    assert load(output / 'registration.json')['ledger_ref'] == expected_ledger
    assert result['new_fits'] == result['new_predictions'] == result['new_observations'] == 0
    assert not result['method_pair_created'] and not result['independent_support']
    after = next(q for q in runner.read_method_reference(result['continued_topic_ref'], output)
                 if q['id'] == 'task:method_choice')
    assert after['method_use_result'] == {'status': 'cached_paid_method_review', 'new_compute_cost': 7}
    assert after['method_use_plan']['paired_condition_research_refs'] == [completed_prior, completed_partial]
    expected_topic = (runner.ref(partial / 'previous_topic.json', 'questions')
                      if missing_question_checkpoint else runner.ref(partial / 'common/questions.json'))
    assert result['prior_topic_ref'] == expected_topic
    assert runner.read_method_reference(result['prior_topic_ref'], output) == [owner]
    if missing_question_checkpoint:
        assert not (partial / 'common/questions.json').exists()
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert runner.return_parent(source, output, previous_return=partial)['new_compute_cost'] == 0
    assert len(calls) == 1
    wrong = tmp_path / 'unrelated_return'
    save(wrong / 'receipt.json', {**load(partial / 'receipt.json'),
        'source_return_ref': runner.ref(tmp_path / 'another_science_result.json')})
    with pytest.raises(ValueError, match='same completed research'):
        runner.return_parent(source, tmp_path / 'wrong_continuation', previous_return=wrong)
    assert not (tmp_path / 'wrong_continuation').exists()


@pytest.mark.parametrize('historical_alias', [False, True])
def test_parent_return_resume_consumes_paid_revision_bundle_without_new_evidence_or_analysis(
        tmp_path, monkeypatch, historical_alias):
    from copy import deepcopy
    from hegel_machine.cognition import method_action_research, question_routing
    from hegel_machine.cognition.common_object_research import (
        CommonObjectDependencyMethods, record_native_parent_research_return)
    from hegel_machine.cognition.models import SearchBudget

    request, _, parent = parent_scene(tmp_path, monkeypatch)
    source = tmp_path / 'completed_research'
    cfg = runner.register_parent(request, source)
    new_parent = {**parent, 'candidate': Candidate('new_parent', Program('add', children=(
        Program('feature', name='m1_x'), Program('const', value=1.))), 'native')}
    selected_ref = runner.ref(source / 'selected_entry.json')
    save(Path(selected_ref['file']), encode_entry(new_parent))
    actual_ref = runner.ref(source / 'receipt.json')
    comparison_ref = runner.ref(source / 'comparison.json')
    save(Path(comparison_ref['file']), {'full_comparison': True, 'selected_ref': selected_ref})
    save(Path(actual_ref['file']), {'status': 'revised', 'full_comparison': True,
        'selected_ref': selected_ref, 'comparison_ref': comparison_ref, 'remaining_compute': 3000})
    prior, partial = Path(cfg['prior_method_review']), tmp_path / 'partial_return'
    evidence_ref = runner.ref(prior / 'revision_evidence.json')
    evidence = {'source_task_ref': cfg['source_task_ref'], 'before_entry_ref': cfg['parent_ref'],
        'after_entry_ref': selected_ref, 'source_operation': cfg['research_action'],
        'comparison_ref': comparison_ref, 'cost_refs': [actual_ref],
        'data_role': 'exposed_development', 'independent_support': False}
    save(Path(evidence_ref['file']), evidence)
    common = CommonObjectDependencyMethods(prior / 'common', [], budget=SearchBudget(50), actions=[]).common
    feedback = record_native_parent_research_return(common, source_task_ref=cfg['source_task_ref'],
        result_ref=actual_ref, revision_evidence_ref=evidence_ref,
        mission={'purpose': 'method_learning', 'allocation': {'compute': 50}})
    owner = deepcopy(next(q for q in common.questions if q['id'] == 'task:method_choice'))
    bundle_ref = runner.ref(prior / 'common/method_feedback/paid_bundle.json')
    save(Path(bundle_ref['file']), {'status': 'complete_no_partition_proposal',
        'basis': {'actual_result_ref': actual_ref, 'revision_evidence_ref': evidence_ref},
        'source_task_ref': cfg['source_task_ref'], 'analysis_compute_cost': 1586,
        'proposal': {'baseline_research_program': parent['research_program'], 'candidates': []}})
    followup_ref = runner.ref(prior / 'common/method_feedback/old_followup.json')
    save(Path(followup_ref['file']), {'status': 'pending_validation',
        'basis': {'parent_proposal_ref': runner.ref(prior / 'older_proposal.json'),
            'study_return_ref': runner.ref(prior / 'older_study_return.json'), 'method_index': 1},
        'followup_of': runner.ref(prior / 'older_proposal.json')})
    owner['method_use_plan']['transfer_partition_proposal_refs'] = [followup_ref, bundle_ref]
    latest_evidence = evidence_ref
    if historical_alias:
        latest_evidence = runner.ref(partial / 'revision_evidence.json')
        save(Path(latest_evidence['file']), evidence)
        owner['method_use_plan']['native_revision_evidence'] = [
            {'feedback_ref': feedback['feedback_ref'], 'revision_evidence_ref': latest_evidence}]
    save(partial / 'common/questions.json', [owner])
    save(partial / 'receipt.json', {'status': 'scoped_unknown', 'source_return_ref': actual_ref,
        'remaining_compute': 590, 'revision_evidence_ref': latest_evidence})
    protected = {path: path.read_bytes() for directory in (prior, partial, source)
                 for path in directory.rglob('*.json')}
    monkeypatch.setattr(method_action_research, 'quote_native_partition_analysis',
        lambda *args: pytest.fail('An equal addressed paid revision must not be analysed again'))

    def consume(restored, returned, public_spec, *, compute_limit):
        current = next(q for q in restored.questions if q['id'] == 'task:method_choice')
        assert current['method_use_plan']['transfer_partition_proposal_refs'] == [followup_ref, bundle_ref]
        records = [(reference, runner.read_method_reference(reference, restored.workspace))
                   for reference in current['method_use_plan']['native_return_refs']]
        research = {'status': 'complete_native_return_review', 'new_compute_cost': 0}
        method_action_research._propose_native_transfer_partitions(
            restored, current, records, research, compute_limit)
        assert compute_limit == 590 and research['new_compute_cost'] == 0
        assert [item['proposal_ref'] for item in research['transfer_partition_proposals']] == [bundle_ref]
        assert returned['status'] == 'reused_native_return'
        return {'status': research['status'], 'research_result': {'method_action_research': research}}

    monkeypatch.setattr(question_routing, 'consume_method_return_agenda', consume)
    output = tmp_path / 'continued_return'
    result = runner.return_parent(source, output, previous_return=partial)
    assert result['revision_evidence_ref'] == latest_evidence
    assert result['remaining_compute'] == 590 and result['compute_cost'] == 0
    assert not (output / 'revision_evidence.json').exists()
    assert not (output / 'common/method_feedback').exists()
    assert all(path.read_bytes() == value for path, value in protected.items())


def test_parent_return_failure_persists_consumed_compute_against_partial_ledger(tmp_path, monkeypatch):
    from hegel_machine.cognition import question_routing
    from hegel_machine.cognition.common_object_research import (
        CommonObjectDependencyMethods, record_native_parent_research_return)
    from hegel_machine.cognition.models import SearchBudget

    request, _, _ = parent_scene(tmp_path, monkeypatch)
    source, partial = tmp_path / 'completed_research', tmp_path / 'partial_return'
    cfg = runner.register_parent(request, source)
    actual_ref = runner.ref(source / 'receipt.json')
    save(Path(actual_ref['file']), {'status': 'retained', 'full_comparison': True,
        'selected_ref': cfg['parent_ref'], 'remaining_compute': 3000})
    common = CommonObjectDependencyMethods(partial / 'common', [], budget=SearchBudget(50), actions=[]).common
    record_native_parent_research_return(common, source_task_ref=cfg['source_task_ref'],
        result_ref=actual_ref, mission={'purpose': 'method_learning', 'allocation': {'compute': 50}})
    save(partial / 'receipt.json', {'status': 'scoped_unknown', 'source_return_ref': actual_ref,
        'remaining_compute': 590})
    original_science, original_return = load(source / 'receipt.json'), load(partial / 'receipt.json')

    def fail_after_paid_review(restored, *args, compute_limit):
        assert compute_limit == 200
        assert not (restored.workspace / 'questions.json').exists()
        owner = next(q for q in restored.questions if q['id'] == 'task:method_choice')
        owner['method_use_result'] = {'status': 'partial_paid_review', 'new_compute_cost': 17}
        restored.costs['compute_cost'] += 17
        raise RuntimeError('failed after consuming paid review budget')

    monkeypatch.setattr(question_routing, 'consume_method_return_agenda', fail_after_paid_review)
    output = tmp_path / 'failed_review'
    with pytest.raises(RuntimeError, match='after consuming'):
        runner.return_parent(source, output, maximum_compute=200, previous_return=partial)
    failure = load(output / 'failure.json')
    assert failure['compute_cost'] == 17 and failure['remaining_compute'] == 573
    assert failure['ledger_ref'] == runner.ref(partial / 'receipt.json', 'remaining_compute')
    assert failure['source_return_ref'] == actual_ref
    assert not (output / 'receipt.json').exists()
    owner = next(q for q in load(output / 'common/questions.json') if q['id'] == 'task:method_choice')
    assert owner['method_use_result'] == {'status': 'partial_paid_review', 'new_compute_cost': 17}
    assert owner['method_use_plan']['native_return_refs'] == [
        runner.ref(partial / 'common/native_method_returns/return_0000.json')]
    assert load(source / 'receipt.json') == original_science
    assert load(partial / 'receipt.json') == original_return


def parameter_scene(tmp_path, monkeypatch):
    from hegel_machine.cognition import transfer_partitions
    request, source, parent = parent_scene(tmp_path, monkeypatch)
    previous = load(source / 'relation/registration.json')
    save(source / 'registration.json', previous)
    frontier = load(source / 'retained_frontier.json')
    save(source / 'receipt.json', {'selected_ref': frontier['selected_ref'],
        'source_task_ref': runner.ref(tmp_path / 'physical_task.json'), 'compute_cost': 5})
    save(source / 'method_return/receipt.json', {'remaining_compute': 62000})
    save(source / 'diagnostics.json', {'selected_predictions': [1.5, 2.5, 5.5]})
    source_method = {'transfer_partition': {'kind': 'Pi1', 'released_groups': ['saved_group'],
        'max_attempts': 48, 'second_stage': {'operation': 'resolve_relation_intervals'}}}
    save(tmp_path / 'source_method.json', source_method)
    cfg = load(request)
    cfg['source_method_ref'] = runner.ref(tmp_path / 'source_method.json')
    save(request, cfg)

    def bind(current, method, **references):
        assert current['candidate'].program == parent['candidate'].program
        assert method == source_method
        return {'status': 'bound', 'research_program': {**current['research_program'],
            'transfer_partition': method['transfer_partition']}, **references}

    def quote(body, training, *, feedback_rows):
        assert body['term_nodes'] == ['complete_column']
        assert len(training) == 2 and len(feedback_rows) == 1
        return {'status': 'quoted', 'required_compute': 55636}

    monkeypatch.setattr(transfer_partitions, 'prepare_parent_parameter_revision', bind)
    monkeypatch.setattr(transfer_partitions, 'quote_transfer_partition', quote)
    return request, source, parent


def test_parameter_continuation_charges_saved_residuals_and_reuses_current_fit(tmp_path, monkeypatch):
    from hegel_machine.cognition import unified_research
    request, source, parent = parameter_scene(tmp_path, monkeypatch)
    output = tmp_path / 'parameter'
    old_ledger = load(source / 'method_return/receipt.json')
    monkeypatch.setattr(parallel_prediction, 'predict_rows',
        lambda *args: pytest.fail('Saved diagnostics and paid scores require no predictions'))
    cfg = runner.register_parameter(request, output)
    assert cfg['available_compute'] == 61997
    assert cfg['method_compute_limit'] == 55636 and cfg['total_compute_upper'] == 55648
    assert cfg['method_return_compute_limit'] == 3000
    assert cfg['prior_method_review'] == str(source / 'method_return')
    assert cfg['parent_ref'] == load(source / 'retained_frontier.json')['selected_ref']
    diagnosis = load(output / 'paid_residuals.json')
    assert diagnosis['compute_cost'] == 3 and diagnosis['mse'] == .25
    assert [item['rows'] for item in diagnosis['probes']] == [2, 1]
    assert all(item['mean_signed_residual'] == -.5 for item in diagnosis['probes'])
    assert diagnosis['new_predictions'] == diagnosis['new_fits'] == 0

    def revised(agent, spec, actual, rows, budget, actions, *, finalists, paid_scores,
                fit_rows, research_action, research_method, feedback_rows, paid_initial):
        assert research_action == 'relation_parameter_revision'
        assert research_method['transfer_partition']['kind'] == 'Pi1'
        assert research_method['term_nodes'] == parent['research_program']['term_nodes']
        assert len(finalists) == 4 and len(paid_scores) == 3
        assert paid_initial['entry'] is actual and paid_initial['training'] == fit_rows
        assert paid_initial['cost_ref'] == runner.ref(source / 'receipt.json')
        assert paid_initial['risk_quantile'] == spec.risk_quantile
        assert len(fit_rows) == 2 and len(feedback_rows) == 1 and len(rows) == 3
        assert budget.spend(7)
        action = {'search_complete': True, 'old_scoring_compute': 0,
            'research': {'initializer_reused': True, 'new_fits': 1}}
        return {'status': 'retained', 'execution_complete': True, 'full_comparison': True,
            'new_claims': [], 'compared_claims': finalists, 'selected': actual, 'action': action}

    monkeypatch.setattr(unified_research.UnifiedResearchAgent, '_revise_paid_structure', revised)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    actual = runner.run_parent(output)
    assert actual['compute_cost'] == 7 and actual['remaining_compute'] == 61990
    assert actual['selected_ref'] == cfg['parent_ref']
    assert load(source / 'method_return/receipt.json') == old_ledger
    assert runner.run_parent(output)['new_compute_cost'] == 0


def test_parameter_registration_reserves_return_and_requires_actual_predictions(tmp_path, monkeypatch):
    request, source, _ = parameter_scene(tmp_path, monkeypatch)
    save(source / 'method_return/receipt.json', {'remaining_compute': 58650})
    with pytest.raises(ValueError, match='research and ordinary return'):
        runner.register_parameter(request, tmp_path / 'unfunded_parameter')
    assert not (tmp_path / 'unfunded_parameter').exists()
    save(source / 'method_return/receipt.json', {'remaining_compute': 62000})
    save(source / 'diagnostics.json', {'selected_predictions': None})
    with pytest.raises(ValueError, match='saved complete predictions'):
        runner.register_parameter(request, tmp_path / 'unmeasured_parameter')
    assert not (tmp_path / 'unmeasured_parameter').exists()


def paired_parameter_scene(tmp_path, monkeypatch, *, repeats=1, training_rows=2):
    """The successor layout keeps paid evidence in common and its new parent in pair."""
    from hegel_machine.cognition import transfer_partitions

    request, old_source, parent = parameter_scene(tmp_path, monkeypatch)
    frontier, _, rows, probes, references, _, _, scores = runner.parent_inputs(old_source)
    source = tmp_path / 'paired_source'
    common, pair = source / 'common', source / 'pair'
    locations = []
    for repeat in range(repeats):
        for index, probe in enumerate(probes):
            location = runner.ref(common / f'batch_{repeat}_{index}.json')
            batch = rows[probe['start']:probe['start'] + probe['rows']]
            save(Path(location['file']), {'probe': {'probe_id': f'{repeat}:{probe["probe_id"]}'},
                'batch': {'observations': [asdict(row) for row in batch]}})
            locations.append(location)
    task_ref = runner.ref(tmp_path / 'physical_task.json')
    registration = {'evaluation_role': 'paid_development_preparation',
        'training_rows': training_rows, 'working_rows': len(rows) * repeats - training_rows,
        'quality_threshold': 1e-10, 'active_wall_seconds': 30, 'source_task_ref': task_ref}
    save(common / 'registration.json', registration)
    save(common / 'public_spec.json', asdict(public()))
    save(common / 'receipt.json', {'full_comparison': True, 'rows': len(rows) * repeats,
        'data_refs': locations, 'selected_ref': references[0][0],
        'scores': [{'candidate_refs': [location], 'mse': measured}
                   for location, measured in references[:-1]]})
    save(pair / 'registration.json', registration)
    selected_ref = runner.ref(pair / 'Pi3/selected_entry.json')
    save(Path(selected_ref['file']), encode_entry(parent))
    save(pair / 'Pi3/comparison.json', {'full_comparison': True,
        'new_scores': [{'candidate_ref': selected_ref, 'loss': .25}]})
    save(pair / 'Pi3/outcome.json', {'full_comparison': True, 'execution_complete': True,
        'selected_ref': selected_ref, 'selected_loss': .25, 'compute_cost': 7})
    save(source / 'retained_frontier.json', {**frontier,
        'original_pool_ref': runner.ref(common / 'receipt.json'),
        'appended_comparison_refs': [runner.ref(pair / 'Pi3/comparison.json')],
        'selected_ref': selected_ref})
    # An allocation within the existing account; registration must neither top
    # it up nor charge a second prediction column for these already scored rows.
    allocation = tmp_path / 'allocation.json'
    save(allocation, {'remaining_compute': 55636 + 4 * len(rows) * repeats + 3000})
    cfg = load(request)
    cfg.update(source=str(source), previous_registration_ref=runner.ref(pair / 'registration.json'),
        previous_result_ref=runner.ref(pair / 'Pi3/outcome.json'),
        ledger_ref=runner.ref(allocation, 'remaining_compute'),
        prior_method_review=str(tmp_path / 'latest_method_return'),
        source_cost_refs=[runner.ref(allocation), runner.ref(pair / 'Pi3/outcome.json')],
        runtime_source=str(old_source))
    save(request, cfg)

    def quote(body, training, *, feedback_rows):
        assert body['term_nodes'] == parent['research_program']['term_nodes']
        assert list(training) == (rows * repeats)[:training_rows]
        assert list(feedback_rows) == (rows * repeats)[training_rows:]
        return {'status': 'quoted', 'required_compute': 55636}

    monkeypatch.setattr(transfer_partitions, 'quote_transfer_partition', quote)
    return request, source, parent, rows * repeats, scores


@pytest.mark.parametrize(('repeats', 'training_rows'), [(1, 2), (40, 96)])
def test_parameter_pair_layout_reuses_full_pool_and_returns_actual_bound_method(
        tmp_path, monkeypatch, repeats, training_rows):
    from hegel_machine.cognition import unified_research
    from hegel_machine.cognition.common_object_research import (
        CommonObjectDependencyMethods, record_native_parent_research_return)
    from hegel_machine.cognition.models import SearchBudget

    request, source, parent, rows, scores = paired_parameter_scene(
        tmp_path, monkeypatch, repeats=repeats, training_rows=training_rows)
    supplied = load(request)
    protected = {path: path.read_bytes() for path in source.rglob('*.json')}
    allocation = Path(supplied['ledger_ref']['file'])
    ledger_before = load(allocation)
    output = tmp_path / 'bound_parameter'
    monkeypatch.setattr(parallel_prediction, 'predict_rows',
        lambda *args: pytest.fail('Paid canonical scores must not trigger another readout'))
    cfg = runner.register_parameter(request, output)
    assert set(path.name for path in source.iterdir()) == {'common', 'pair', 'retained_frontier.json'}
    assert not (output / 'paid_residuals.json').exists()
    assert 'paid_records_ref' not in cfg and cfg['data_refs'] == load(source / 'common/receipt.json')['data_refs']
    assert cfg['preparation_compute'] == 0 and cfg['available_compute'] == ledger_before['remaining_compute']
    assert cfg['total_compute_upper'] + cfg['method_return_compute_limit'] == cfg['available_compute']
    assert cfg['training_rows'] == training_rows and cfg['feedback_rows'] == len(rows) - training_rows
    assert cfg['parent_ref'] != load(source / 'common/receipt.json')['selected_ref']
    assert cfg['paid_initializer_cost_ref'] == supplied['previous_result_ref']

    def revised(agent, spec, actual, observed, budget, actions, *, finalists, paid_scores,
                fit_rows, research_action, research_method, feedback_rows, paid_initial):
        assert len(finalists) == 4 and paid_scores == scores
        assert observed == rows and fit_rows == rows[:training_rows]
        assert feedback_rows == rows[training_rows:]
        assert actual['candidate'].program == parent['candidate'].program
        assert paid_initial['entry'] is actual and paid_initial['training'] == fit_rows
        assert paid_initial['cost_ref'] == supplied['previous_result_ref']
        assert research_method == runner.read_method_reference(cfg['research_method_ref'], output)
        assert research_action == 'relation_parameter_revision' and budget.spend(7)
        return {'status': 'retained', 'execution_complete': True, 'full_comparison': True,
            'new_claims': [], 'compared_claims': finalists, 'selected': actual,
            'action': {'search_complete': True, 'research': {'new_fits': 1}}}

    monkeypatch.setattr(unified_research.UnifiedResearchAgent, '_revise_paid_structure', revised)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_parent(output)
    assert result['remaining_compute'] == cfg['available_compute'] - 7
    assert result['method_invoked'] and result['executed_method_ref'] == cfg['research_method_ref']
    assert result['method_execution_ref'] == runner.ref(output / 'diagnostics.json', 'native_action')
    assert load(output / 'comparison.json')['data_refs'] == cfg['data_refs']
    assert runner.run_parent(output)['new_compute_cost'] == 0

    prior = Path(cfg['prior_method_review'])
    common = CommonObjectDependencyMethods(prior / 'common', [], budget=SearchBudget(50), actions=[]).common
    old_ref = runner.ref(tmp_path / 'old_return.json')
    save(Path(old_ref['file']), {'status': 'unresolved'})
    record_native_parent_research_return(common, source_task_ref=cfg['source_task_ref'], result_ref=old_ref,
        mission={'purpose': 'method_learning', 'allocation': {'compute': 50}})
    returned = runner.return_parent(output, tmp_path / 'ordinary_return', maximum_compute=30)
    record = runner.read_method_reference(returned['feedback']['feedback_ref'], tmp_path)
    assert record['method_refs'] == [cfg['research_method_ref']]
    assert returned['executed_method_ref'] == cfg['research_method_ref']
    assert returned['method_execution_ref'] == result['method_execution_ref']
    assert returned['remaining_compute'] == result['remaining_compute'] - returned['compute_cost']
    assert not returned['method_pair_created'] and not returned['independent_support']
    assert returned['new_fits'] == returned['new_predictions'] == returned['new_observations'] == 0
    assert load(allocation) == ledger_before
    assert all(path.read_bytes() == value for path, value in protected.items())


def test_parameter_pair_layout_cannot_increase_account_or_restore_pre_call_winner(tmp_path, monkeypatch):
    request, source, _, _, _ = paired_parameter_scene(tmp_path, monkeypatch)
    supplied = load(request)
    allocation = Path(supplied['ledger_ref']['file'])
    original = load(allocation)
    save(allocation, {'remaining_compute': original['remaining_compute'] - 1})
    with pytest.raises(ValueError, match='research and ordinary return'):
        runner.register_parameter(request, tmp_path / 'unfunded_pair')
    assert not (tmp_path / 'unfunded_pair').exists()
    assert load(allocation)['remaining_compute'] == original['remaining_compute'] - 1
    save(allocation, original)
    supplied['previous_result_ref'] = runner.ref(source / 'common/receipt.json')
    save(request, supplied)
    with pytest.raises(ValueError, match='current selected parent'):
        runner.register_parameter(request, tmp_path / 'wrong_parent')
    assert not (tmp_path / 'wrong_parent').exists() and load(allocation) == original


def confirmation_scene(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hegel_machine.cognition import anonymous_object_lab
    from hegel_machine.cognition.models import Probe

    _, source, _ = parent_scene(tmp_path, monkeypatch)
    frontier = load(source / 'retained_frontier.json')
    incumbent_ref = frontier['selected_ref']
    selected = {'candidate': Candidate('new', Program('add', children=(
        Program('feature', name='m1_x'), Program('const', value=1.))), 'native'),
        'training_loss': 0., 'research_program': {'kind': 'retained'}}
    selected_ref = runner.ref(source / 'new_selected.json')
    save(Path(selected_ref['file']), encode_entry(selected))
    save(source / 'new_comparison.json', {'full_comparison': True,
        'new_scores': [{'candidate_ref': selected_ref, 'loss': 0.}]})
    frontier['appended_comparison_refs'].append(runner.ref(source / 'new_comparison.json'))
    frontier.update(selected_ref=selected_ref, selected_mse=0., complete_current_pool_occurrences=5)
    save(source / 'retained_frontier.json', frontier)
    cohort = tmp_path / 'cohort'
    task = {'seed': 73, 'objects': 4}
    protocol_ref = runner.ref(tmp_path / 'protocol.json', 'allocations', 'abc')
    save(tmp_path / 'protocol.json', {'authorized_limits': {'observation_cost': 8000},
        'allocations': {'abc': {}}})
    save(cohort / 'registration.json', {'tasks': [task], 'arms': ['A', 'B', 'C'],
        'terminal': {'length': 48, 'trajectories': 1, 'random_seed_offset': 1000003},
        'budget_contract_ref': protocol_ref})
    terminal_ref = cohort / 'terminal/73/summary.json'
    save(terminal_ref, {'shared_actual_costs': {'compute_cost': 0, 'observation_cost': 0},
        'arms': {name: {'prediction_measured': False} for name in ('A', 'B', 'C')}})
    for arm in ('A', 'B', 'C'):
        save(cohort / 'runs/73' / arm / 'failure.json', {'new_acquisitions': [
            {'probe_id': 'missing_labels', 'sequence': [{'c': float(i)} for i in range(24)]}]})
    save(source / 'registration.json', {'source_task_ref': runner.ref(cohort / 'registration.json', 'tasks', 0),
        'quality_threshold': 1e-10, 'parent_ref': incumbent_ref, 'active_wall_seconds': 30})
    save(source / 'receipt.json', {'execution_complete': True, 'full_comparison': True,
        'selected_ref': selected_ref, 'selected_loss': 0.})
    save(source / 'method_return/receipt.json', {'remaining_compute': 42966})
    save(source / 'source_costs.json', {'cumulative_protocol_observations': 3921})

    class Lab:
        channels = [SimpleNamespace(name='fine', cost_per_row=12)]

        def __init__(self, **settings):
            assert settings == task

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def random_probe(self, seed, *, length, name):
            assert seed == 1000076 and length == 48
            return Probe(name, tuple({'c': float(100 + i)} for i in range(length)))

        def acquire(self, *args, **kwargs):
            pytest.fail('Preparation must not acquire responses')

    monkeypatch.setattr(anonymous_object_lab, 'AnonymousObjectLab', Lab)
    return source, cohort


def test_confirmation_freezes_actual_parents_and_uses_shared_old_protocol(tmp_path, monkeypatch):
    from scripts import run_learned_structure_successor as historical
    source, cohort = confirmation_scene(tmp_path, monkeypatch)
    output = tmp_path / 'confirmation'
    cfg = runner.register_confirmation(source, output)
    assert cfg['confirmation_seed'] == 1000076 and cfg['confirmation_rows'] == 48
    assert cfg['confirmation_cost_quote'] == {'compute_cost': 96, 'observation_cost': 48,
        'measurement_cost': 576, 'queries': 1, 'new_fits': 0, 'model_calls': 0}
    assert cfg['incumbent_ref'] == load(source / 'registration.json')['parent_ref']
    assert cfg['available_compute'] == 42966 and cfg['previous_observations'] == 3921
    assert not cfg['method_pair_created'] and not cfg['independent_task_support']
    assert load(output / 'prepared_state.json')['fallback'] == 11 / 3
    frozen = load(output / 'frozen_entries.json')
    assert frozen['selected']['source_ref'] == load(source / 'receipt.json')['selected_ref']
    assert not load(output / 'confirmation_plan.json')['responses_obtained']

    def confirm(directory, registration):
        assert load(directory / 'frozen_entries.json') == frozen
        assert set(registration['operations']) == {'incumbent', 'selected'}
        result = {'arms': {'incumbent': {'passed': False}, 'selected': {'passed': True}},
            'shared_actual_costs': {'compute_cost': 96, 'observation_cost': 48, 'measurement_cost': 576}}
        save(directory / 'confirmation/receipt.json', result)
        return result

    monkeypatch.setattr(historical, 'confirm_pair', confirm)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_confirmation(output)
    assert result['status'] == 'confirmed' and result['remaining_compute'] == 42870
    assert result['cumulative_protocol_observations'] == 3969
    assert result['new_fits'] == 0 and not (output / 'pair.json').exists()
    assert runner.run_confirmation(output)['new_compute_cost'] == 0


def canonical_confirmation_scene(tmp_path, monkeypatch):
    source, cohort = confirmation_scene(tmp_path, monkeypatch)
    frontier, _, rows, probes, references, _, _, _ = runner.parent_inputs(source)
    common = source / 'common'
    data_refs = []
    for probe in probes:
        location = runner.ref(common / f"{probe['probe_id']}.json")
        batch = rows[probe['start']:probe['start'] + probe['rows']]
        controls = [row['controls'] for row in records() if row['probe_id'] == probe['probe_id']]
        save(Path(location['file']), {'probe': {'probe_id': probe['probe_id'], 'sequence': controls},
            'batch': {'observations': [asdict(row) for row in batch]}})
        data_refs.append(location)
    save(common / 'registration.json', {'evaluation_role': 'paid_development_preparation',
        'training_rows': 2, 'working_rows': 1, 'quality_threshold': 1e-10,
        'task': {'seed': 73, 'objects': 4}})
    save(common / 'public_spec.json', asdict(public()))
    save(common / 'receipt.json', {'full_comparison': True, 'rows': len(rows), 'data_refs': data_refs,
        'scores': [{'candidate_refs': [location], 'mse': measured} for location, measured in references]})
    save(source / 'retained_frontier.json', {**frontier,
        'original_pool_ref': runner.ref(common / 'receipt.json'), 'appended_comparison_refs': []})
    previous = load(source / 'registration.json')
    method_ref = runner.ref(source / 'method_instance.json', 'research_program')
    save(Path(method_ref['file']), {'research_program': {
        'saved_complete_columns': ['first', 'second'],
        'transfer_partition': {'kind': 'Pi1', 'second_stage': {'operation': 'resolve_relation_intervals'}}}})
    save(source / 'registration.json', {**previous,
        'source_task_ref': runner.ref(common / 'registration.json', 'task'),
        'research_method_ref': method_ref, 'max_candidates': 4,
        'training_rows': 2, 'feedback_rows': 1})
    latest = tmp_path / 'latest_return'
    save(latest / 'receipt.json', {'remaining_compute': 42758})
    save(latest / 'source_costs.json', {'cumulative_protocol_observations': 4689})
    save(latest / 'protocol.json', {'length': 48, 'trajectories': 1, 'random_seed_offset': 1000003})
    save(latest / 'authorization.json', {'observation_limit': 8000})
    request = tmp_path / 'canonical_confirmation_request.json'
    save(request, {'confirmation_protocol_ref': runner.ref(latest / 'protocol.json'),
        'ledger_ref': runner.ref(latest / 'receipt.json', 'remaining_compute'),
        'source_costs_ref': runner.ref(latest / 'source_costs.json'),
        'observation_limit_ref': runner.ref(latest / 'authorization.json', 'observation_limit'),
        'prior_method_review': str(latest)})
    # These files do not exist in the supported successor-pair layout.
    for path in (cohort / 'registration.json', cohort / 'terminal/73/summary.json',
                 source / 'method_return/receipt.json', source / 'source_costs.json', tmp_path / 'paid.json'):
        path.unlink()
    return source, request, data_refs, latest


def test_confirmation_from_canonical_pair_freezes_current_method_without_legacy_cohort(tmp_path, monkeypatch):
    from hegel_machine.cognition import research_programs
    source, request, data_refs, latest = canonical_confirmation_scene(tmp_path, monkeypatch)
    protected = [source / 'registration.json', source / 'receipt.json', source / 'retained_frontier.json',
        source / 'method_instance.json', latest / 'receipt.json', latest / 'source_costs.json',
        *[Path(location['file']) for location in data_refs]]
    original = {path: path.read_bytes() for path in protected}
    monkeypatch.setattr(parallel_prediction, 'predict_rows',
        lambda *args, **kwargs: pytest.fail('Registration cannot repeat paid predictions'))
    monkeypatch.setattr(research_programs._Fitter, 'fit',
        lambda *args, **kwargs: pytest.fail('Frozen confirmation cannot fit either predictor'))
    output = tmp_path / 'canonical_confirmation'
    cfg = runner.register_confirmation(source, output, request=request)
    supplied, previous = load(request), load(source / 'registration.json')
    assert cfg['available_compute'] == 42758 and cfg['previous_observations'] == 4689
    assert cfg['ledger_ref'] == supplied['ledger_ref']
    assert cfg['source_cost_refs'] == [supplied['source_costs_ref']]
    assert cfg['allocation_ref'] == supplied['observation_limit_ref']
    assert cfg['confirmation_protocol_ref'] == supplied['confirmation_protocol_ref']
    assert cfg['prior_method_review'] == str(latest)
    assert cfg['frozen_research_method_ref'] == previous['research_method_ref']
    assert cfg['method_execution_contract_ref'] == runner.ref(source / 'registration.json')
    assert runner.read_method_reference(cfg['method_execution_contract_ref'], output)['max_candidates'] == 4
    assert cfg['confirmation_rows'] == 48 and cfg['confirmation_seed'] == 1000076
    assert cfg['quality_threshold'] == previous['quality_threshold']
    assert cfg['confirmation_cost_quote']['compute_cost'] == 96
    assert not cfg['independent_task_support'] and not cfg['method_pair_created']
    frozen, actual = load(output / 'frozen_entries.json'), load(source / 'receipt.json')
    assert frozen['incumbent']['source_ref'] == previous['parent_ref']
    assert frozen['selected']['source_ref'] == actual['selected_ref']
    for arm in frozen:
        expected = runner.read_method_reference(frozen[arm]['source_ref'], source)
        assert frozen[arm]['candidate']['program'] == expected['candidate']['program']
    plan = load(output / 'confirmation_plan.json')
    assert plan['data_refs'] == data_refs and plan['checked_control_sequences'] == 2
    assert not plan['responses_obtained'] and not plan['old_acquisition_refs']
    assert 'paid_records_ref' not in plan and 'original_unpurchased_terminal_ref' not in cfg
    assert {path: path.read_bytes() for path in protected} == original


@pytest.mark.parametrize('change', ['paid_prefix', 'observation_limit', 'ledger'])
def test_canonical_confirmation_respects_paid_controls_and_latest_account(tmp_path, monkeypatch, change):
    source, request, data_refs, latest = canonical_confirmation_scene(tmp_path, monkeypatch)
    if change == 'paid_prefix':
        document = load(Path(data_refs[0]['file']))
        document['probe']['sequence'] = [{'c': 100.}, {'c': 101.}]
        save(Path(data_refs[0]['file']), document)
        reason = 'already paid control prefix'
    elif change == 'observation_limit':
        save(latest / 'authorization.json', {'observation_limit': 4689 + 48 - 1})
        reason = 'cannot fund confirmation'
    else:
        save(latest / 'receipt.json', {'remaining_compute': 96 + 3000 - 1})
        reason = 'cannot fund confirmation'
    output = tmp_path / 'rejected_confirmation'
    with pytest.raises(ValueError, match=reason):
        runner.register_confirmation(source, output, request=request)
    assert not output.exists()


def test_confirmation_rejects_paid_missing_label_prefix_or_used_panel(tmp_path, monkeypatch):
    source, cohort = confirmation_scene(tmp_path, monkeypatch)
    failure = cohort / 'runs/73/A/failure.json'
    save(failure, {'new_acquisitions': [{'probe_id': 'missing_labels',
        'sequence': [{'c': float(100 + i)} for i in range(24)]}]})
    with pytest.raises(ValueError, match='already paid control prefix'):
        runner.register_confirmation(source, tmp_path / 'collision')
    assert not (tmp_path / 'collision').exists()
    summary = cohort / 'terminal/73/summary.json'
    measured = load(summary)
    measured['shared_actual_costs']['observation_cost'] = 48
    save(summary, measured)
    with pytest.raises(ValueError, match='already been measured'):
        runner.register_confirmation(source, tmp_path / 'old_panel')


def test_confirmation_keeps_failed_costs_and_does_not_restart(tmp_path, monkeypatch):
    from scripts import run_learned_structure_successor as historical
    source, _ = confirmation_scene(tmp_path, monkeypatch)
    output = tmp_path / 'confirmation_failure'
    runner.register_confirmation(source, output)

    def fail(directory, registration):
        save(directory / 'confirmation/failure.json', {'costs': {
            'compute_cost': 3, 'observation_cost': 48, 'measurement_cost': 576}})
        raise RuntimeError('actual frozen readout failure')

    monkeypatch.setattr(historical, 'confirm_pair', fail)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    with pytest.raises(RuntimeError, match='actual frozen readout'):
        runner.run_confirmation(output)
    failure = load(output / 'failure.json')
    assert failure['remaining_compute'] == 42963
    assert failure['actual_costs']['observation_cost'] == 48
    with pytest.raises(ValueError, match='once'):
        runner.run_confirmation(output)


def interval_scene(tmp_path, monkeypatch, *, already_explained=False):
    from hegel_machine.cognition.models import Observation

    source, _ = confirmation_scene(tmp_path, monkeypatch)
    cfg = load(source / 'registration.json')
    cfg['training_rows'] = 2
    save(source / 'registration.json', cfg)
    actual = load(source / 'receipt.json')
    parent = load(Path(actual['selected_ref']['file']))
    parent['research_program'].update(fit_rows=2, transfer_partition={'kind': 'Pi1',
        'released_groups': ['saved_group'], 'second_stage': {
            'operation': 'resolve_relation_intervals', 'compute_limit': 15000}})
    save(Path(actual['selected_ref']['file']), parent)
    bonus = 1. if already_explained else 2.
    suffix = [Observation({'m1_x': 9.}, 9. + bonus),
        Observation({'m1_x': 10.}, 10. + bonus, ({'m1_x': 9., '__target__': 9. + bonus},))]
    save(source / 'frozen_confirmation/confirmation/observations.json', {
        'probe': {'probe_id': 'saved_confirmation', 'sequence': [{'c': 9.}, {'c': 10.}]},
        'batch': {'observations': [asdict(row) for row in suffix]}})
    incumbent = runner.retained_entry(cfg['parent_ref'], source, 0)
    current = runner.retained_entry(actual['selected_ref'], source, 0)
    frozen, arms = {}, {}
    for name, item in [('incumbent', incumbent), ('selected', current)]:
        program = item['candidate'].program
        values = [program.predict(row) for row in suffix]
        frozen[name] = {'program': program.to_dict()}
        arms[name] = {'predictions': values, 'mse': sum((value - row.target) ** 2
            for value, row in zip(values, suffix)) / len(suffix)}
    save(source / 'frozen_confirmation/confirmation/frozen_predictors.json', frozen)
    save(source / 'frozen_confirmation/confirmation/receipt.json', {'arms': arms})
    save(source / 'diagnostics.json', {'selected_predictions': [2., 3., 6.]})
    save(source / 'confirmation_feedback/method_return/receipt.json', {'remaining_compute': 42135})
    save(source / 'final_source_costs.json', {'separate_suballocation_remaining': 42135})
    return source


def test_interval_appends_only_missing_scores_preserves_groups_and_restores_full_view(tmp_path, monkeypatch):
    from scripts import continue_successor_relation_intervals as historical
    from hegel_machine.cognition import transfer_partitions
    source = interval_scene(tmp_path, monkeypatch)
    output = tmp_path / 'interval'
    cfg = runner.register_interval(source, output)
    assert (cfg['old_candidate_occurrences'], cfg['old_unique_programs'], cfg['paid_suffix_programs']) == (5, 4, 2)
    assert cfg['comparison_compute_upper'] == 4 and cfg['interval_compute_limit'] == 15000
    assert cfg['prior_method_review'] == str(source / 'confirmation_feedback/method_return')
    old = runner.parent_inputs(source)
    paid_programs = set(runner.interval_inputs(source)['paid_suffix'])
    calls, original_loss = [], historical.loss

    def suffix_loss(program, rows, fallback, quantile):
        assert program not in paid_programs and len(rows) == 2
        calls.append(program)
        return original_loss(program, rows, fallback, quantile)

    monkeypatch.setattr(historical, 'loss', suffix_loss)
    delivered = []

    def continued(parent, training, feedback, budget, *, tolerance, feedback_loss,
                  feedback_experiments, feedback_squared_errors, fit_source_ref):
        assert len(training) == 2 and len(feedback) == 3
        assert [len(group) for group in feedback_experiments] == [1, 2]
        assert [row for group in feedback_experiments for row in group] == feedback
        assert feedback_squared_errors == [0., 1., 1.] and feedback_loss == pytest.approx(2 / 3)
        assert parent['training_loss'] == 0. and fit_source_ref == cfg['parent_ref']
        assert budget.spend(17)
        for index, cut in enumerate([6., 7., 8., 8.5, 8.8]):
            feature = Program('feature', name='m1_x')
            program = Program('add', children=(feature, Program('if', children=(
                Program('lt', children=(feature, Program('const', value=cut))),
                Program('const', value=1.), Program('const', value=2.)))))
            delivered.append({'candidate': Candidate(f'interval{index}', program, 'native'),
                'research_program': {**parent['research_program'], 'fit_rows': 2, 'evaluation_rows': 5},
                'training_loss': 0., 'validation_loss': 0., 'all_paid_loss': 0.,
                'evaluation_rows': 5, 'paid_row_squared_errors': [0.] * 5})
        return {'status': 'paid_interval_supported', 'execution_complete': True,
            'candidates': delivered, 'selected': delivered[-1], 'new_fits': 0,
            'fixed_readout_only': True, 'new_observations': 0}

    monkeypatch.setattr(transfer_partitions, 'continue_transfer_partition', continued)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_interval(output)
    assert result['compute_cost'] == 21 and result['remaining_compute'] == 42114
    assert result['new_fits'] == result['new_observations'] == 0 and len(calls) == 2
    assert result['new_candidates'] == 3 and result['actual_interval_assessments'] == 5
    assert result['method_invoked'] and result['executed_method_ref'] == cfg['executed_method_ref']
    exported = load(output / 'new_entries.json')
    assert exported[0]['candidate']['candidate_id'] == 'interval4'
    assert len(load(output / 'interval_assessments.json')['assessments']) == 5
    restored = runner.parent_inputs(output)
    assert len(restored[2]) == 5 and len(restored[5]) == 8
    assert restored[1]['training_rows'] == 2 and restored[1]['feedback_rows'] == 3
    assert restored[3][-1]['probe_id'] == 'saved_confirmation'
    assert restored[6]['research_program']['fit_rows'] == 2 and restored[6]['validation_loss'] == 0.
    assert restored[7][old[5][0]['candidate'].program] == pytest.approx(2.2)
    assert runner.run_interval(output)['new_compute_cost'] == 0 and len(calls) == 2


def test_interval_stops_after_complete_pool_already_explains(tmp_path, monkeypatch):
    from hegel_machine.cognition import transfer_partitions
    source = interval_scene(tmp_path, monkeypatch, already_explained=True)
    output = tmp_path / 'explained'
    runner.register_interval(source, output)
    monkeypatch.setattr(transfer_partitions, 'continue_transfer_partition',
        lambda *args, **kwargs: pytest.fail('Do not repeat already explained research'))
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_interval(output)
    assert result['new_candidates'] == 0 and result['compute_cost'] == 4
    assert not result['method_invoked']
    assert len(runner.parent_inputs(output)[2]) == 5


def test_interval_failure_accounts_local_spend_once_and_retains_paid_comparison(tmp_path, monkeypatch):
    from hegel_machine.cognition import transfer_partitions
    source = interval_scene(tmp_path, monkeypatch)
    output = tmp_path / 'failed_interval'
    runner.register_interval(source, output)

    def fail(parent, training, feedback, budget, **kwargs):
        assert budget.spend(19)
        raise RuntimeError('interval execution cutoff')

    monkeypatch.setattr(transfer_partitions, 'continue_transfer_partition', fail)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    with pytest.raises(RuntimeError, match='execution cutoff'):
        runner.run_interval(output)
    assert load(output / 'failure.json')['compute_cost'] == 23
    assert load(output / 'failure.json')['remaining_compute'] == 42112
    assert load(output / 'expanded_pool_comparison.json')['compute_cost'] == 4
    with pytest.raises(ValueError, match='once'):
        runner.run_interval(output)


def canonical_interval_scene(tmp_path, monkeypatch):
    from hegel_machine.cognition.models import Observation

    source, _, data_refs, latest = canonical_confirmation_scene(tmp_path, monkeypatch)
    previous, actual = load(source / 'registration.json'), load(source / 'receipt.json')
    selected_file = Path(actual['selected_ref']['file'])
    selected = load(selected_file)
    selected['research_program'].update(fit_rows=2,
        saved_complete_columns=[f'column_{index}' for index in range(8)],
        saved_coefficients=[float(index) for index in range(9)],
        transfer_partition={'kind': 'Pi1', 'released_groups': ['saved_group'],
            'second_stage': {'operation': 'resolve_relation_intervals', 'compute_limit': 15000}})
    save(selected_file, selected)
    confirmation = tmp_path / 'external_confirmation/run'
    suffix = [Observation({'m1_x': 9.}, 11.),
        Observation({'m1_x': 10.}, 12., ({'m1_x': 9., '__target__': 11.},))]
    save(confirmation / 'confirmation/observations.json', {
        'probe': {'probe_id': 'new_confirmation', 'sequence': [{'c': 9.}, {'c': 10.}]},
        'batch': {'observations': [asdict(row) for row in suffix]}})
    frozen, arms = {}, {}
    for name, location in [('incumbent', previous['parent_ref']), ('selected', actual['selected_ref'])]:
        program = runner.retained_entry(location, source, 0)['candidate'].program
        values = [program.predict(row) for row in suffix]
        frozen[name] = {'program': program.to_dict()}
        arms[name] = {'predictions': values, 'mse': sum((value - row.target) ** 2
            for value, row in zip(values, suffix)) / len(suffix), 'passed': False}
    save(confirmation / 'confirmation/frozen_predictors.json', frozen)
    save(confirmation / 'confirmation/receipt.json', {'arms': arms, 'rows': len(suffix)})
    save(confirmation / 'registration.json', {'source': str(source),
        'source_selected_ref': actual['selected_ref'], 'source_task_ref': previous['source_task_ref']})
    save(confirmation / 'receipt.json', {'status': 'confirmation_failed',
        'source_task_ref': previous['source_task_ref'], 'selected_ref': actual['selected_ref'],
        'confirmation_ref': runner.ref(confirmation / 'confirmation/receipt.json')})
    save(source / 'diagnostics.json', {'selected_predictions': [2., 3., 6.]})
    save(latest / 'receipt.json', {'remaining_compute': 42603})
    request = tmp_path / 'canonical_interval_request.json'
    save(request, {'confirmation_source': str(confirmation),
        'ledger_ref': runner.ref(latest / 'receipt.json', 'remaining_compute'),
        'prior_method_review': str(latest),
        'source_cost_refs': [runner.ref(latest / 'source_costs.json'), runner.ref(latest / 'receipt.json')]})
    return source, confirmation, request, data_refs, latest


def test_canonical_interval_consumes_external_paid_return_and_restores_complete_pool(tmp_path, monkeypatch):
    from copy import deepcopy
    from scripts import continue_successor_relation_intervals as historical
    from hegel_machine.cognition import research_programs, transfer_partitions

    source, confirmation, request, data_refs, latest = canonical_interval_scene(tmp_path, monkeypatch)
    protected = {path: path.read_bytes() for directory in (source, confirmation, latest)
                 for path in directory.rglob('*.json')}
    old = runner.parent_inputs(source)
    body = deepcopy(old[6]['research_program'])
    output = tmp_path / 'canonical_interval'
    cfg = runner.register_interval(source, output, request=request)
    assert cfg['confirmation_source'] == str(confirmation)
    assert cfg['data_refs'] == data_refs and 'paid_records_ref' not in cfg
    assert cfg['appended_observation_refs'] == [runner.ref(confirmation / 'confirmation/observations.json')]
    assert cfg['ledger_ref'] == runner.ref(latest / 'receipt.json', 'remaining_compute')
    assert cfg['available_compute'] == 42603 and cfg['prior_method_review'] == str(latest)
    assert cfg['source_cost_refs'] == load(request)['source_cost_refs']
    assert (cfg['training_rows'], cfg['feedback_rows'], cfg['comparison_rows']) == (2, 3, 5)
    assert (cfg['old_candidate_occurrences'], cfg['old_unique_programs'], cfg['paid_suffix_programs']) == (5, 4, 2)
    assert cfg['comparison_compute_upper'] == 4
    paid = runner.interval_inputs(source, confirmation_source=confirmation)['paid_suffix']
    calls, original_loss = [], historical.loss

    def suffix_loss(program, rows, fallback, quantile):
        assert program not in paid and len(rows) == 2
        calls.append(program)
        return original_loss(program, rows, fallback, quantile)

    def continued(parent, training, feedback, budget, *, tolerance, feedback_loss,
                  feedback_experiments, feedback_squared_errors, fit_source_ref):
        assert parent['research_program'] == body
        assert training == old[2][:2]
        assert feedback[0] == old[2][2] and feedback[1].history == ()
        assert feedback[2].history == ({'m1_x': 9., '__target__': 11.},)
        assert [len(group) for group in feedback_experiments] == [1, 2]
        assert [row for group in feedback_experiments for row in group] == feedback
        assert feedback_squared_errors == [0., 1., 1.] and feedback_loss == pytest.approx(2 / 3)
        assert fit_source_ref == cfg['parent_ref'] and budget.spend(17)
        feature = Program('feature', name='m1_x')
        program = Program('add', children=(feature, Program('if', children=(
            Program('lt', children=(feature, Program('const', value=8.5))),
            Program('const', value=1.), Program('const', value=2.)))))
        candidate = {'candidate': Candidate('paid_interval_new', program, 'native'),
            'research_program': {**deepcopy(body), 'evaluation_rows': 5},
            'training_loss': 0., 'validation_loss': 0., 'all_paid_loss': 0.,
            'evaluation_rows': 5, 'paid_row_squared_errors': [0.] * 5}
        return {'status': 'paid_interval_supported', 'execution_complete': True,
            'candidates': [candidate], 'selected': candidate,
            'new_fits': 0, 'new_observations': 0, 'fixed_readout_only': True}

    monkeypatch.setattr(historical, 'loss', suffix_loss)
    monkeypatch.setattr(research_programs._Fitter, 'fit',
        lambda *args, **kwargs: pytest.fail('An interval continuation must keep the original fitted readout'))
    monkeypatch.setattr(transfer_partitions, 'continue_transfer_partition', continued)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_interval(output)
    assert result['compute_cost'] == 21 and result['remaining_compute'] == 42582
    assert result['new_fits'] == result['new_observations'] == 0
    assert result['full_comparison'] and result['new_candidates'] == 1 and len(calls) == 2
    restored = runner.parent_inputs(output)
    assert len(restored[2]) == 5 and len(restored[5]) == 6
    assert restored[1]['data_refs'] == data_refs and 'paid_records_ref' not in restored[1]
    assert restored[3][-1] == {'probe_id': 'new_confirmation', 'start': 3, 'rows': 2}
    restored_body = restored[6]['research_program']
    assert restored_body['fit_rows'] == 2 and restored_body['evaluation_rows'] == 5
    assert restored_body['saved_complete_columns'] == body['saved_complete_columns']
    assert restored_body['saved_coefficients'] == body['saved_coefficients']
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert runner.run_interval(output)['new_compute_cost'] == 0 and len(calls) == 2


@pytest.mark.parametrize('change', ['source', 'selected', 'ledger'])
def test_canonical_interval_requires_same_confirmed_parent_and_latest_balance(tmp_path, monkeypatch, change):
    source, confirmation, request, _, latest = canonical_interval_scene(tmp_path, monkeypatch)
    if change == 'source':
        cfg = load(confirmation / 'registration.json')
        cfg['source'] = str(tmp_path / 'another_source')
        save(confirmation / 'registration.json', cfg)
    elif change == 'selected':
        result = load(confirmation / 'receipt.json')
        result['selected_ref'] = load(source / 'registration.json')['parent_ref']
        save(confirmation / 'receipt.json', result)
    else:
        save(latest / 'receipt.json', {'remaining_compute': 4 + 15000 + 3000 - 1})
    output = tmp_path / 'unregistered_interval'
    with pytest.raises(ValueError):
        runner.register_interval(source, output, request=request)
    assert not output.exists()


def parent_method_pair_scene(tmp_path, monkeypatch):
    from hegel_machine.cognition import structural_research_program, transfer_partitions

    _, source, parent = parent_scene(tmp_path, monkeypatch)
    baseline = {'transfer_partition': {'kind': 'Pi1', 'second_stage': {
        'operation': 'resolve_relation_intervals', 'compute_limit': 15000}}}
    challenger = {'transfer_partition': {'kind': 'Pi3'}, 'declared_parent': 'paid_relation'}
    save(source / 'baseline.json', baseline)
    save(source / 'proposal.json', {'proposal': {'candidates': [{'research_program': challenger}]}})
    relation = load(source / 'relation/registration.json')
    relation['method_ref'] = runner.ref(source / 'baseline.json')
    save(source / 'relation/registration.json', relation)
    allocation = {'source_frontier_ref': runner.ref(source / 'retained_frontier.json'),
        'maximum_compute': 10000, 'new_observation_limit': 0, 'allocation_id': 'explicit-paid-pair'}
    save(tmp_path / 'allocation.json', allocation)
    save(tmp_path / 'latest_ledger.json', {'remaining_compute': 29798})
    save(tmp_path / 'protected_frontier.json', {'complete_current_pool_occurrences': 255})
    request = tmp_path / 'method_pair_request.json'
    supplied = {'source': str(source), 'challenger_method_ref': runner.ref(source / 'proposal.json',
        'proposal', 'candidates', 0, 'research_program'),
        'allocation_ref': runner.ref(tmp_path / 'allocation.json'),
        'prior_ledger_ref': runner.ref(tmp_path / 'latest_ledger.json'),
        'prior_method_review': str(tmp_path / 'latest_common'),
        'protected_frontier_ref': runner.ref(tmp_path / 'protected_frontier.json')}
    completed = tmp_path / 'paid_two_directions'
    save(completed / 'registration.json', {**relation,
        'paid_records_ref': runner.ref(tmp_path / 'paid.json')})
    save(completed / 'new_entries.json', [encode_entry(parent), encode_entry(parent)])
    save(completed / 'receipt.json', {'compute_cost': 17})
    save(completed / 'comparison.json', {'full_comparison': True, 'new_scores': [
        {'candidate_ref': runner.ref(completed / 'new_entries.json', index), 'loss': .25}
        for index in range(2)]})
    supplied['paid_fit_refs'] = {'Pi3': [{'entry_ref': runner.ref(completed / 'new_entries.json', index),
        'cost_ref': runner.ref(completed / 'receipt.json')} for index in range(2)]}
    supplied['shared_paid_comparison_refs'] = [runner.ref(completed / 'comparison.json')]
    save(request, supplied)

    def bind(actual, method, *, source_method_ref, parent_ref):
        assert actual['candidate'] == parent['candidate']
        assert parent_ref == runner.ref(source / 'relation/selected_entry.json')
        assert method == baseline and source_method_ref == relation['method_ref']
        return {'status': 'ready', 'research_program': {**method, 'bound_parent': parent_ref}}

    monkeypatch.setattr(transfer_partitions, 'prepare_parent_parameter_revision', bind)
    monkeypatch.setattr(structural_research_program, 'research_program_fit_contract',
        lambda body: {'program': parent['candidate'].program})

    def quote(body, training, *, feedback_rows):
        assert [row.target for row in training] == [2., 3.]
        assert [row.target for row in feedback_rows] == [6.]
        return {'status': 'quoted', 'required_compute': 50 if body['transfer_partition']['kind'] == 'Pi1' else 20}

    monkeypatch.setattr(transfer_partitions, 'quote_transfer_partition', quote)
    monkeypatch.setattr(parallel_prediction, 'predict_rows',
        lambda *args: pytest.fail('Preparing the paid method pair must not predict'))
    return request, source


def test_parent_method_pair_keeps_historical_union_paid_split_and_distinct_bodies(tmp_path, monkeypatch):
    request, source = parent_method_pair_scene(tmp_path, monkeypatch)
    output = tmp_path / 'method_pair'
    old_frontier = load(source / 'retained_frontier.json')
    cfg = runner.register_parent_method_pair(request, output)
    assert cfg['evaluation_arm_names'] == {'baseline': 'Pi1', 'challenger': 'Pi3'}
    assert cfg['equal_method_compute_limit'] == 50 and cfg['per_arm_comparison_reserve'] == 12
    assert cfg['total_new_compute_upper'] == 3076  # fixed two bodies, not two maximum caps
    assert cfg['source_pool_size'] == 4 and cfg['max_candidates'] == 4
    assert cfg['paid_fit_inputs']['Pi1']['initial']['entry_ref'] == old_frontier['selected_ref']
    assert len(cfg['paid_fit_inputs']['Pi3']['fits']) == len(cfg['shared_paid_scores']) == 2
    assert all(item['training_refs'] == cfg['training_refs'] for item in cfg['paid_fit_inputs']['Pi3']['fits'])
    assert cfg['remaining_compute'] == 10000 and cfg['original_ledger']['remaining_compute'] == 29798
    assert cfg['retained_method_pair_ref'] is None and not cfg['independent_support']
    assert cfg['prior_method_review'] == str(tmp_path / 'latest_common')
    methods = load(output / 'frozen_methods.json')
    assert methods['Pi1']['bound_parent'] == old_frontier['selected_ref']
    assert methods['Pi1']['transfer_partition']['second_stage']['compute_limit'] == 15000
    assert methods['Pi3']['declared_parent'] == 'paid_relation'
    for arm, location in cfg['native_method_artifact_refs'].items():
        artifact = runner.read_method_reference(location, output)
        assert artifact['id'] == f'native:parent_partition:method_pair:{arm}'
        assert runner.read_method_reference(artifact['execution']['research_program_ref'], output) == methods[arm]
    views = load(output / 'paid_observations.json')
    assert [len(view['observations']) for view in views] == [2, 1]
    assert views[0]['probe']['sequence'] == [{'c': 1.}, {'c': 2.}]
    assert views[1]['observations'][0]['history'] == []
    assert cfg['training_refs'] == [cfg['data_refs'][0]] and cfg['feedback_refs'] == [cfg['data_refs'][1]]
    paid = load(output / 'source_pool.json')
    assert len(paid['scores']) == 3 and sum(len(item['candidate_refs']) for item in paid['scores']) == 4
    assert paid['compute_cost'] == paid['new_predictions'] == 0
    assert load(source / 'retained_frontier.json') == old_frontier
    assert load(tmp_path / 'protected_frontier.json')['complete_current_pool_occurrences'] == 255


def test_parent_method_pair_never_borrows_fits_from_an_appended_data_view(tmp_path, monkeypatch):
    request, _ = parent_method_pair_scene(tmp_path, monkeypatch)
    changed = tmp_path / 'paid_two_directions/registration.json'
    registration = load(changed)
    registration['appended_observation_refs'] = [runner.ref(tmp_path / 'later48.json')]
    save(changed, registration)
    output = tmp_path / 'wrong_cached_view'
    with pytest.raises(ValueError, match='same complete paid view'):
        runner.register_parent_method_pair(request, output)
    assert not output.exists()


def test_parent_method_pair_quotes_before_allocation_and_refuses_partial_funding(tmp_path, monkeypatch):
    request, source = parent_method_pair_scene(tmp_path, monkeypatch)
    supplied = load(request)
    allocation_ref = supplied.pop('allocation_ref')
    save(request, supplied)
    output = tmp_path / 'quoted_only'
    result = runner.register_parent_method_pair(request, output)
    assert result['status'] == 'parent_method_pair_quote_only' and result['total_new_compute_upper'] == 3076
    assert not output.exists()
    supplied['allocation_ref'] = allocation_ref
    save(request, supplied)
    allocation = load(Path(allocation_ref['file']))
    allocation['maximum_compute'] = 3075
    save(Path(allocation_ref['file']), allocation)
    result = runner.register_parent_method_pair(request, output)
    assert result['status'] == 'complete_paired_unit_unfunded_not_method_negative'
    assert not output.exists()
    allocation.update(maximum_compute=10000, source_frontier_ref=runner.ref(source / 'wrong_pool.json'))
    save(Path(allocation_ref['file']), allocation)
    with pytest.raises(ValueError, match='complete historical frontier'):
        runner.register_parent_method_pair(request, output)
    assert not output.exists()


@pytest.mark.parametrize('changed_interface', [False, True])
def test_new_development_keeps_exact_methods_all_old_candidates_and_no_old_scores(tmp_path, monkeypatch, changed_interface):
    _, knowledge, _ = parent_scene(tmp_path, monkeypatch)
    canonical, output = tmp_path / 'frozen_pair', tmp_path / 'new_source'
    old_task = {'seed': 1, 'objects': 4, 'name_seed': 11, 'wiring_seed': 12,
        'actuator_curvature': .1, 'variant': 'exchange', 'profile': 'confirmation',
        'readout': 'window_total_event', 'relation_direction': 'right'}
    save(canonical / 'task.json', old_task)
    methods = {'Pi1': {'frozen_body': 'full_parameter_stage'}, 'Pi3': {'frozen_body': 'two_alternatives'}}
    save(canonical / 'frozen_methods.json', methods)
    identities = {arm: runner.ref(canonical / (arm + '_method.json')) for arm in methods}
    save(canonical / 'registration.json', {'source_task_ref': runner.ref(canonical / 'task.json'),
        'source_model_ref': runner.ref(canonical / 'model.json'),
        'evaluation_arm_names': {'baseline': 'Pi1', 'challenger': 'Pi3'},
        'native_method_artifact_refs': identities, 'quality_threshold': 1e-7,
        'research_config': {'max_candidates': 4}, 'active_wall_seconds': 60})
    save(tmp_path / 'allocation.json', {'source_frontier_ref': runner.ref(knowledge / 'retained_frontier.json'),
        'maximum_compute': 1000, 'new_observation_limit': 5, 'allocation_id': 'new-development'})
    save(tmp_path / 'ledger.json', {'remaining_compute': 43385})
    task = {**old_task, 'seed': 2, 'relation_direction': 'symmetric'}
    if changed_interface:
        task['wiring_seed'] = 99
    request = tmp_path / 'new_request.json'
    save(request, {'canonical_source': str(canonical), 'knowledge_source': str(knowledge), 'task': task,
        'allocation_ref': runner.ref(tmp_path / 'allocation.json'),
        'prior_ledger_ref': runner.ref(tmp_path / 'ledger.json'),
        'prior_source_costs_ref': runner.ref(tmp_path / 'source_costs.json'),
        'prior_method_review': str(canonical / 'method_return'),
        'training_probe_rows': [2, 2], 'working_rows': 1})
    if changed_interface:
        with pytest.raises(ValueError, match='measurement interface'):
            runner.register_frozen_method_development(request, output)
        assert not output.exists()
        return
    original_refs = runner.parent_inputs(knowledge)[4]
    cfg = runner.register_frozen_method_development(request, output)
    assert load(output / 'frozen_methods.json') == methods
    assert cfg['native_method_artifact_refs'] == identities
    assert cfg['retained_method_pair_ref'] == runner.ref(canonical / 'method_return/pair.json')
    pool = load(output / 'source_pool.json')
    assert pool['scores'] == [{'candidate_ref': location} for location, _ in original_refs]
    assert not pool['target_scores_available']
    assert cfg['confirmation_rows'] == 0 and cfg['feature_mappings'] == [{}]
    assert cfg['original_ledger']['remaining_compute'] == 43385
    assert 'paid_fit_inputs' not in cfg and 'shared_paid_scores' not in cfg


def test_new_public_preparation_frontier_restores_saved_probe_histories_and_full_pool(tmp_path, monkeypatch):
    _, source, _ = parent_scene(tmp_path, monkeypatch)
    frontier, original, rows, probes, references, entries, parent, _ = runner.parent_inputs(source)
    common, output = tmp_path / 'common', tmp_path / 'new_frontier'
    locations = []
    for index, probe in enumerate(probes):
        location = runner.ref(common / f'batch{index}.json')
        batch = rows[probe['start']:probe['start'] + probe['rows']]
        save(Path(location['file']), {'probe': {'probe_id': probe['probe_id']},
            'batch': {'observations': [asdict(row) for row in batch]}})
        locations.append(location)
    save(common / 'registration.json', {'evaluation_role': 'paid_development_preparation',
        'training_rows': 2, 'working_rows': len(rows) - 2})
    save(common / 'receipt.json', {'full_comparison': True, 'rows': len(rows), 'data_refs': locations,
        'scores': [{'candidate_refs': [location], 'mse': measured} for location, measured in references]})
    save(output / 'retained_frontier.json', {**frontier,
        'original_pool_ref': runner.ref(common / 'receipt.json'), 'appended_comparison_refs': []})
    restored = runner.parent_inputs(output)
    assert restored[2] == rows and restored[3] == probes
    assert restored[4] == references and len(restored[5]) == len(entries)
    assert restored[6]['candidate'].program == parent['candidate'].program
    assert restored[1]['training_rows'] == 2
    assert restored[1]['comparison_rows'] == len(rows)
