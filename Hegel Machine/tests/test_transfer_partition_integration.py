"""The generated partition must change the ordinary replay and paid comparison."""

from copy import deepcopy

from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.research_programs import (
    evaluate_research_step_revisions, quote_research_step_revisions,
    replay_research_program, research_program_replay_inputs)
from test_transfer_partitions import observations, plan, proposal


def test_ordinary_quote_replay_and_paid_pair_consume_the_generated_partition(monkeypatch):
    import hegel_machine.cognition.transfer_partitions as module

    prepared, data = proposal(), observations()
    quoted = quote_research_step_revisions(plan(dormant=True), data, data,
        prepared_proposal=prepared)
    assert quoted['status'] == 'quoted' and quoted['method_count'] == 2
    baseline = replay_research_program(prepared['baseline_research_program'], data, SearchBudget(10000))
    assert baseline['training_loss'] > .01

    def no_regeneration(*args, **kwargs):
        raise AssertionError('Already generated ordinary proposal repeated')

    monkeypatch.setattr(module, 'propose_transfer_partitions', no_regeneration)
    refs = [{'file': 'paid/source.json'}]
    paid = {'execution_complete': True, 'program': baseline['candidate'].program.to_dict(),
        'research_program': baseline['research_program'],
        'training_loss': baseline['training_loss'], 'validation_loss': baseline['training_loss'],
        'paid_compute_cost': baseline['research_program']['search_compute'],
        'basis': {'replay_inputs': research_program_replay_inputs(prepared['baseline_research_program']),
            'source_refs': refs, 'training_rows': len(data), 'validation_rows': len(data), 'risk_quantile': .5}}
    result = evaluate_research_step_revisions(plan(dormant=True), data, data,
        SearchBudget(quoted['per_method_compute_upper']),
        per_method_compute=quoted['per_method_compute_upper'], source_refs=refs,
        prepared_proposal=prepared, paid_baseline=paid)
    assert result['status'] == 'development_step_revision_selected'
    assert result['methods'][0]['reused'] and result['methods'][0]['new_compute_cost'] == 0
    assert result['methods'][1]['validation_loss'] < 1e-12
    assert not result['independent_support']


def test_reuse_inputs_include_actual_parameter_bindings_but_not_provenance():
    original = proposal()['candidates'][0]['research_program']
    renamed = deepcopy(original)
    renamed['transfer_partition']['source_revision']['after_ref'] = 'relocated-equal-source'
    assert research_program_replay_inputs(original) == research_program_replay_inputs(renamed)
    changed = deepcopy(original)
    changed['selection_steps'][0]['parameters'][0]['radius_intervals'] = [[.1, .9]]
    assert research_program_replay_inputs(original) != research_program_replay_inputs(changed)


def test_paired_method_passes_development_return_and_retains_incomplete_stage(monkeypatch):
    import hegel_machine.cognition.research_programs as module
    from test_transfer_partitions import second_stage_fixture

    entry, training, feedback, measured = second_stage_fixture()
    body = entry['research_program']
    prepared = {'status': 'proposed', 'baseline_research_program': deepcopy(body),
        'candidates': [{'research_program': deepcopy(body), 'edit': {'operation': 'resolve_relation_intervals'}}]}
    calls = []

    def replay(payload, rows, budget, **kwargs):
        calls.append(kwargs)
        result = deepcopy(entry)
        result['transfer_partition_followup'] = {'feedback_loss': measured,
            'execution_complete': False, 'budget_incomplete': True}
        return result

    monkeypatch.setattr(module, 'replay_research_program', replay)
    result = evaluate_research_step_revisions(body, training, feedback, SearchBudget(2000),
        per_method_compute=1000, prepared_proposal=prepared, tolerance=1e-9)
    assert len(calls) == 2 and all(call['feedback_rows'] == feedback and call['tolerance'] == 1e-9 for call in calls)
    assert all(not method['execution_complete'] for method in result['methods'])
    assert result['budget_incomplete']
    assert all(method['validation_loss'] == measured for method in result['methods'])
    assert all(method['new_compute_cost'] == 0 for method in result['methods'])
    assert all(method['assessment_role'] == 'adaptive_development' for method in result['methods'])
    quote = quote_research_step_revisions(body, training, feedback, prepared_proposal=prepared)
    assert all(method['transfer_partition_quote']['followup_compute_upper'] == 15000 for method in quote['methods'])
