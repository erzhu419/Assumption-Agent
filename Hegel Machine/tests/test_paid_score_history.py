"""Paid score payloads stay recoverable without accumulating in comparison history."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.paid_score_history import (
    compact_agent_comparisons, retain_comparison_history,
)
from hegel_machine.cognition.research_episodes import load_source_values, resolve_program_sources
from test_inquiry_revision import action
from test_python_paid_score_reuse import (
    ChoiceLab, MODELS, choose, no_execution, scene, scoring_costs,
)


def test_complete_new_and_reused_scores_keep_exact_old_field_addresses(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    first = choose(agent, lab)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    second = choose(agent, lab)
    assert first['new_scored_count'] == second['reused_score_count'] == 2
    agent._prepare_program_state()
    for index, expected in enumerate((first, second)):
        saved = agent.operator_results[index]
        assert saved['paid_score_history'] == 'assessment_references'
        assert 'examples' not in saved['evaluations'][1]['paid_residuals']
        assert first['evaluations'][1]['paid_residuals']['examples']
        for path, expected_value in (
                (['evaluations', 1], expected['evaluations'][1]),
                (['evaluations', 1, 'paid_residuals'], expected['evaluations'][1]['paid_residuals']),
                (['evaluations', 1, 'paid_residuals', 'examples', 0],
                 expected['evaluations'][1]['paid_residuals']['examples'][0]),
                (['evaluations', 1, 'representation_diagnostic'],
                 expected['evaluations'][1]['representation_diagnostic'])):
            reference = {'file': 'research_state.json',
                         'json_path': ['operator_results', index, *path]}
            expected_value = json.loads(json.dumps(expected_value))
            assert load_source_values([reference], agent.workspace)[0]['value'] == expected_value
            assert resolve_program_sources(agent, [reference]) == [expected_value]
    # Reading a whole retained state does not resurrect all repeated payloads.
    raw = load_source_values([{'file': 'research_state.json', 'json_path': []}], agent.workspace)[0]['value']
    assert 'examples_ref' in raw['operator_results'][0]['evaluations'][1]['paid_residuals']


def test_old_history_and_all_persistent_comparison_owners_are_compacted(tmp_path):
    agent, lab = scene(tmp_path)
    live = choose(agent, lab)
    agent.operator_results[0].update(deepcopy(live))
    agent.operator_results[0].pop('paid_score_history')
    agent.events.append({'action': 'run', 'feedback': live})
    agent.method_results.append({'path': 'fixture.py', 'feedback': deepcopy(live)})
    agent.questions[-1]['after_observation_results'] = [deepcopy(live)]
    original = deepcopy(live)
    compact_agent_comparisons(agent)
    assert live == original  # Caller-owned feedback is not rewritten.
    for retained in (agent.operator_results[0], agent.events[0]['feedback'],
                     agent.method_results[0]['feedback'],
                     agent.questions[-1]['after_observation_results'][0]):
        assert retained['paid_score_history'] == 'assessment_references'
        assert retained['evaluations'][1]['retention_decision'] == original['evaluations'][1]['retention_decision']
        assert 'examples' not in retained['evaluations'][1]['paid_residuals']
    first_objects = (agent.operator_results[0], agent.events[0]['feedback'], agent.method_results[0]['feedback'])
    compact_agent_comparisons(agent)
    assert all(a is b for a, b in zip(first_objects, (
        agent.operator_results[0], agent.events[0]['feedback'], agent.method_results[0]['feedback'])))


def test_alias_and_occurrence_decisions_restore_without_replaying_retention(tmp_path):
    agent, lab = scene(tmp_path)
    first = choose(agent, lab)
    alias = deepcopy(first)
    evaluation = alias['evaluations'][1]
    evaluation['path'] = evaluation['executed_path'] = 'renamed.py'
    evaluation['paid_residuals']['model'] = 'renamed.py'
    evaluation['retention_decision']['status'] = 'incumbent_retained'
    evaluation['paid_score_reuse']['matched_model'] = MODELS[1]
    saved = retain_comparison_history(alias)
    agent.operator_results.append(saved)
    agent._prepare_program_state()
    reference = {'file': 'research_state.json',
                 'json_path': ['operator_results', 1, 'evaluations', 1]}
    assert load_source_values([reference], agent.workspace)[0]['value'] == json.loads(json.dumps(evaluation))


def test_nested_incumbent_scores_are_referenced_and_restore_the_exact_paid_value(tmp_path):
    agent, lab = scene(tmp_path)
    choose(agent, lab)
    agent._execute('query', {'actions': [action(3.)]}, lab)
    live = choose(agent, lab)
    expected = deepcopy(live)
    agent._prepare_program_state()
    retained = agent.operator_results[-1]
    assert any('retained_comparison' in item for item in retained['evaluations'])
    for index, evaluation in enumerate(retained['evaluations']):
        incumbent = evaluation.get('retained_comparison')
        if incumbent is None:
            continue
        assert incumbent['score_result_ref'] == incumbent['paid_score_reuse']['assessment_ref']
        assert 'representation_diagnostic' not in incumbent
        assert 'examples' not in incumbent['paid_residuals']
        for suffix, original in (([], expected['evaluations'][index]),
                (['retained_comparison'], expected['evaluations'][index]['retained_comparison']),
                (['retained_comparison', 'paid_residuals', 'examples'],
                 expected['evaluations'][index]['retained_comparison']['paid_residuals']['examples'])):
            reference = {'file': 'research_state.json', 'json_path': [
                'operator_results', len(agent.operator_results) - 1, 'evaluations', index, *suffix]}
            assert load_source_values([reference], agent.workspace)[0]['value'] == json.loads(json.dumps(original))
    assert live == expected


def test_recovery_upgrades_old_nested_payload_once_and_keeps_unpaid_failures(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hegel_machine.cognition import paid_score_history

    agent, lab = scene(tmp_path)
    choose(agent, lab)
    agent._execute('query', {'actions': [action(3.)]}, lab)
    live = choose(agent, lab)
    retained = retain_comparison_history(live)
    for current, original in zip(retained['evaluations'], live['evaluations']):
        if 'retained_comparison' in original:
            current['retained_comparison'] = deepcopy(original['retained_comparison'])
    failure = {'returncode': 1, 'stderr': 'original incomplete comparison',
               'paid_residuals': {'examples': [{'row_index': 0}]}}
    retained['evaluations'].append({'retained_comparison': deepcopy(failure)})
    restored = SimpleNamespace(operator_results=[retained], events=[], method_results=[], questions=[])
    paid_score_history.restore_comparison_history(restored)
    assert restored.operator_results[0]['evaluations'][-1]['retained_comparison'] == failure
    good = next(item['retained_comparison'] for item in restored.operator_results[0]['evaluations']
                if item.get('retained_comparison', {}).get('paid_score_reuse', {}).get('assessment_ref'))
    assert good['score_result_ref'] and 'representation_diagnostic' not in good
    monkeypatch.setattr(paid_score_history, '_compact_retained_comparisons', no_execution)
    paid_score_history.compact_agent_comparisons(restored)


@pytest.mark.parametrize('failure', ['exception', 'truncated', 'unstable_context'])
def test_nonreusable_scores_preserve_complete_failure_payload(tmp_path, monkeypatch, failure):
    agent, lab = scene(tmp_path)
    if failure == 'exception':
        agent._save(MODELS[0], "def predict(row): raise ValueError('retained fixture failure')\n")
    elif failure == 'truncated':
        monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model',
            lambda *args, **kwargs: {'returncode': 0, 'truncated': True, 'stderr': 'partial fixture'})
    else:
        agent._save(MODELS[0], "from pathlib import Path\n"
            "def fit(rows):\n p=Path('counter'); p.write_text(p.read_text()+'x')\n"
            "def predict(row): return 0.\n")
        agent._save('old/_evaluated_model/counter', '')
        agent._save('old/_evaluated_model/model_observations.json', '[]')
    live = choose(agent, lab, MODELS[:1])
    assert 'assessment_ref' not in live['evaluations'][0]['paid_score_reuse']
    assert agent.operator_results[-1]['evaluations'] == live['evaluations']


def test_cold_common_restore_keeps_compact_history_and_reuses_paid_scores(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path / 'source')
    choose(agent, lab)
    agent._prepare_program_state()
    previous = agent.result('researching')
    cold, _ = scene(tmp_path / 'cold')
    import shutil
    shutil.rmtree(cold.workspace)
    cold.workspace.mkdir()
    cold_lab = ChoiceLab()
    cold_lab.costs = {'observation_cost': 0}
    cold.restore_practice_research(previous, agent.workspace, cold_lab, confirmation_seed=0)
    before = scoring_costs(cold)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    result = choose(cold, cold_lab)
    assert result['reused_score_count'] == 2 and scoring_costs(cold) == before
    assert 'examples' not in cold.operator_results[0]['evaluations'][1]['paid_residuals']
    snapshot = cold.result('researching')
    snapshot['events'].append({'fixture': 'owned snapshot'})
    assert not cold.events


def test_provider_context_keeps_the_saved_counterexample_reference(tmp_path):
    from hegel_machine.cognition.code_provider_context import _compact_practice_feedback

    agent, lab = scene(tmp_path)
    choose(agent, lab)
    retained = agent.operator_results[-1]['evaluations'][1]
    feedback = _compact_practice_feedback(retained, ['events', 0, 'feedback'])
    residuals = feedback['paid_residuals']
    assert 'examples' not in residuals and 'example_count' not in residuals
    examples = load_source_values([residuals['examples_ref']], agent.workspace)[0]['value']
    assert examples and examples[0]['squared_error'] > 0


def test_actual_failure_consumer_reads_retained_compact_score_and_witness(tmp_path, monkeypatch):
    from hegel_machine.cognition import common_archive_inquiry
    from hegel_machine.cognition.language import Program
    from test_revision_comparison_failures import saved_unconsumed_comparison

    agent, lab, owner, identifier, live, reference = saved_unconsumed_comparison(tmp_path, monkeypatch)
    agent._prepare_program_state()
    retained = agent.operator_results[-1]
    expected = json.loads(json.dumps(live['evaluations'][-1]['paid_residuals']['examples'][0]))
    costs = deepcopy(agent.costs)
    monkeypatch.setattr(Program, 'predict', no_execution)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    outcome = common_archive_inquiry.record_revision_comparison_failures(
        agent, owner, retained, source_ref=reference)
    assert outcome['new_refutation_count'] == 1 and not outcome['unknown']
    failure = agent.theory_consolidation['transfers'][-1]['ordinary_comparison_failure']
    witness = load_source_values([failure['witness_ref']], agent.workspace)[0]['value']
    assert witness == expected and agent.costs == costs


@pytest.mark.parametrize('change', ['missing_declaration', 'wrong_canonical_model', 'changed_context'])
def test_literal_alias_failure_needs_matching_execution_contract(tmp_path, monkeypatch, change):
    from hegel_machine.cognition import common_archive_inquiry
    from test_revision_comparison_failures import saved_unconsumed_comparison

    agent, lab, owner, identifier, live, reference = saved_unconsumed_comparison(tmp_path, monkeypatch)
    evaluation = live['evaluations'][-1]
    assert evaluation['paid_score_reuse']['match_basis'] == 'identical_closed_literal_execution'
    if change == 'missing_declaration':
        evaluation['paid_score_reuse'].pop('match_basis')
    elif change == 'wrong_canonical_model':
        evaluation['paid_score_reuse']['matched_model'] = 'different.py'
    else:
        (agent.workspace / evaluation['path']).with_name('model_observations.json').write_text('[]')
    costs, transfers = deepcopy(agent.costs), deepcopy(agent.theory_consolidation['transfers'])
    outcome = common_archive_inquiry.record_revision_comparison_failures(
        agent, owner, live, source_ref=reference)
    assert outcome['new_refutation_count'] == 0
    assert [item['reason'] for item in outcome['unknown']] == ['exact_current_paid_assessment_scope_required']
    assert agent.costs == costs and agent.theory_consolidation['transfers'] == transfers


def test_equal_parent_metadata_is_owned_shared_and_snapshot_independent(tmp_path):
    agent, lab = scene(tmp_path)
    choose(agent, lab)
    live = choose(agent, lab)
    saved = agent.operator_results[-1]
    first, second = saved['evaluations']
    assert first['evaluated_model'] == second['evaluated_model']
    assert first['evaluated_model'] is second['evaluated_model']
    assert first['retention_decision'] is not second['retention_decision']
    assert first['retention_decision']['incumbent'] is second['retention_decision']['incumbent']
    for original, retained in zip(live['evaluations'], saved['evaluations']):
        assert retained['evaluated_model'] is not original['evaluated_model']
        assert retained['retention_decision']['incumbent'] is not original['retention_decision']['incumbent']
        assert retained['score_result_ref'] is retained['paid_score_reuse']['assessment_ref']
    expected = deepcopy(saved)
    assert json.dumps(saved) == json.dumps(expected)
    live['evaluations'][0]['evaluated_model']['mse'] = 123.
    assert saved == expected
    agent.events.append({'step': 0, 'feedback': saved})
    snapshot = agent.result('researching')
    snapshot['events'][0]['feedback']['evaluations'][0]['evaluated_model']['mse'] = 456.
    snapshot['events'][0]['feedback']['evaluations'][0]['retention_decision']['incumbent']['mse'] = 789.
    assert saved == expected


def test_old_event_restore_compacts_before_copying_and_owns_both_arms(tmp_path, monkeypatch):
    from hegel_machine.cognition import paid_score_history
    from hegel_machine.cognition.paid_score_history import retain_event_history

    agent, lab = scene(tmp_path)
    choose(agent, lab)
    live = choose(agent, lab)
    event = {'step': 0, 'action': 'run', 'arguments': {'models': MODELS.copy()},
             'feedback': live, 'usage': {'input_tokens': 12}}
    previous = [event]
    expected = deepcopy(previous)
    original_copy = deepcopy
    forbidden = [previous, event, live, live['evaluations']]
    forbidden.extend(e['paid_residuals']['examples'] for e in live['evaluations'])

    def bounded_copy(value):
        assert all(value is not old for old in forbidden), 'Old repeated payload copied before compaction'
        return original_copy(value)

    monkeypatch.setattr(paid_score_history, 'deepcopy', bounded_copy)
    first, second = retain_event_history(previous), retain_event_history(previous)
    assert previous == expected
    assert first == second and first is not second and first[0] is not second[0]
    assert first[0]['arguments'] is not second[0]['arguments']
    assert first[0]['feedback'] is not second[0]['feedback']
    assert 'examples' not in first[0]['feedback']['evaluations'][0]['paid_residuals']
    assert first[0]['feedback']['evaluations'][0]['evaluated_model'] is not second[0]['feedback']['evaluations'][0]['evaluated_model']
    first[0]['arguments']['models'].append('other.py')
    first[0]['feedback']['evaluations'][0]['retention_decision']['candidate']['mse'] = 456.
    assert second[0]['arguments']['models'] == MODELS and previous == expected
    assert second[0]['feedback']['evaluations'][0]['retention_decision']['candidate']['mse'] != 456.


def test_cold_common_restore_rebuilds_sharing_once_after_json(tmp_path, monkeypatch):
    from hegel_machine.cognition import paid_score_history
    from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint
    import shutil

    agent, lab = scene(tmp_path / 'source')
    choose(agent, lab)
    choose(agent, lab)
    agent._prepare_program_state()
    previous = agent.result('researching')
    decoded = read_json_checkpoint(agent.workspace / 'research_state.json')
    first, second = decoded['operator_results'][-1]['evaluations']
    assert first['evaluated_model'] == second['evaluated_model']
    assert first['evaluated_model'] is not second['evaluated_model']
    assert first['score_result_ref'] is not first['paid_score_reuse']['assessment_ref']
    cold, _ = scene(tmp_path / 'cold')
    shutil.rmtree(cold.workspace)
    cold.workspace.mkdir()
    cold_lab = ChoiceLab()
    cold_lab.costs = {'observation_cost': 0}
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    cold.restore_practice_research(previous, agent.workspace, cold_lab, confirmation_seed=0)
    restored = cold.operator_results[-1]['evaluations']
    assert restored[0]['evaluated_model'] is restored[1]['evaluated_model']
    assert restored[0]['retention_decision']['incumbent'] is restored[1]['retention_decision']['incumbent']
    assert restored[0]['score_result_ref'] is restored[0]['paid_score_reuse']['assessment_ref']
    assert restored[0]['evaluated_model'] is not first['evaluated_model']
    assert cold.costs == previous['costs']
    # JSON still contains complete ordinary values at the same original paths.
    reread = read_json_checkpoint(cold.workspace / 'research_state.json')
    assert reread['operator_results'][-1] == decoded['operator_results'][-1]
    monkeypatch.setattr(paid_score_history, 'share_owned_comparison_metadata', no_execution)
    cold._prepare_program_state()
    cold.result('researching')
