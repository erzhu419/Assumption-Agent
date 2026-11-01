"""The next method phase must not run on an active unit or use extra funds."""

from scripts import finish_native_method_composition as runner
from scripts.run_partial_role_parent import load, save


def test_active_source_is_left_untouched(tmp_path, monkeypatch):
    output, unit = tmp_path / 'queued', tmp_path / 'active'
    save(output / 'registration.json', {'completed_unit': str(unit)})
    monkeypatch.setattr(runner, 'merge_pair_returns', lambda *args: (_ for _ in ()).throw(
        AssertionError('The running source cannot start the next method phase')))
    assert runner.run(output) == {'status': 'waiting_for_frozen_unit', 'new_compute_cost': 0}
    assert not (output / 'started.json').exists() and not unit.exists()


def test_insufficient_closed_balance_does_not_increase_allowance(tmp_path, monkeypatch):
    output, unit = tmp_path / 'queued', tmp_path / 'closed'
    save(output / 'registration.json', {'completed_unit': str(unit),
        'maximum_compute': 10000, 'path_reserve': 1000})
    save(unit / 'costs.json', {'cumulative_protocol_compute_interval': [500, 600]})
    save(unit / 'receipt.json', {'closed_source_costs_ref': {'file': str(unit / 'costs.json')},
        'remaining_compute': 900})
    monkeypatch.setattr(runner, 'merge_pair_returns', lambda *args: (_ for _ in ()).throw(
        AssertionError('Unfunded formation cannot start')))
    assert runner.run(output) == {'status': 'method_formation_unfunded', 'new_compute_cost': 0}
    assert not (output / 'started.json').exists()


def test_repaired_occurrence_uses_latest_paid_balance(tmp_path):
    output, unit, prior = tmp_path / 'repair', tmp_path / 'closed', tmp_path / 'prior'
    save(unit / 'costs.json', {'cumulative_protocol_compute_interval': [500, 600]})
    save(unit / 'receipt.json', {'closed_source_costs_ref': {'file': str(unit / 'costs.json')},
        'remaining_compute': 466487, 'latest_method_review': str(unit / 'review')})
    save(prior / 'receipt.json', {'closed_unit_ref': {'file': str(unit / 'receipt.json')},
        'remaining_compute': 900, 'source_costs_ref': {'file': str(unit / 'costs.json')}})
    save(output / 'registration.json', {'completed_unit': str(unit),
        'maximum_compute': 10000, 'path_reserve': 1000,
        'previous_occurrence_ref': {'file': str(prior / 'receipt.json')}})
    assert runner.run(output) == {'status': 'method_formation_unfunded', 'new_compute_cost': 0}
    assert not (output / 'started.json').exists()


def test_completed_formation_is_referenced_once_and_practice_returns_to_same_topic(tmp_path, monkeypatch):
    from types import SimpleNamespace
    output, unit, previous = tmp_path / 'path_resume', tmp_path / 'closed', tmp_path / 'failed'
    formation = previous / 'method_return'
    source_ref = {'file': str(tmp_path / 'science/receipt.json')}
    proposal_ref = {'file': str(formation / 'proposal.json')}
    method_ref = {**proposal_ref, 'json_path': ['proposal', 'candidates', 0, 'research_program']}
    save(formation / 'proposal.json', {'basis': {'operation': 'revise_then_reidentify',
        'actual_result_ref': source_ref}})
    formed = {'compute_cost': 2114, 'return': {'research_result': {'method_action_research': {
        'staged_method_proposals': [{'proposal_ref': proposal_ref, 'research_program_refs': [method_ref]}]}}}}
    save(formation / 'receipt.json', formed)
    record_ref = {'file': str(previous / 'record.json')}
    save(previous / 'record.json', {'method_pair_ref': {'file': 'existing_pair.json'}})
    owner = {'id': 'task:method_choice', 'method_use_plan': {'native_return_refs': [record_ref]}}
    save(formation / 'common/questions.json', [owner])
    save(unit / 'costs.json', {'cumulative_protocol_compute_interval': [500, 600]})
    save(unit / 'receipt.json', {'status': 'confirmed',
        'closed_source_costs_ref': {'file': str(unit / 'costs.json')}, 'remaining_compute': 5000})
    save(previous / 'costs.json', {'cumulative_protocol_compute_interval': [3000, 4000]})
    previous_ref = {'file': str(previous / 'resume_state.json')}
    save(previous / 'resume_state.json', {'closed_unit_ref': {'file': str(unit / 'receipt.json')},
        'remaining_compute': 500, 'source_costs_ref': {'file': str(previous / 'costs.json')}})
    save(output / 'registration.json', {'completed_unit': str(unit), 'maximum_compute': 400,
        'path_reserve': 100, 'previous_occurrence_ref': previous_ref,
        'source_execution_ref': source_ref,
        'completed_formation_ref': {'file': str(formation / 'receipt.json')}})
    monkeypatch.setattr(runner, 'merge_pair_returns', lambda *args: (_ for _ in ()).throw(
        AssertionError('Already paid formation must not run again')))
    monkeypatch.setattr(runner, 'prepare_paid', lambda *args, **kwargs: (output / 'paid_path/runtime').mkdir(parents=True))
    def run_paid(*args, **kwargs):
        save(output / 'paid_path/receipt.json', {'status': 'paid_stage_paths_completed', 'compute_cost': 1})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', run_paid)
    common = SimpleNamespace(questions=[owner], costs={'compute_cost': 0})
    monkeypatch.setattr(runner, 'CommonObjectDependencyMethods', lambda *args, **kwargs: SimpleNamespace(common=common))
    monkeypatch.setattr(runner, 'ensure_python_method_topic', lambda *args, **kwargs: None)
    def record(agent, question, **kwargs):
        assert question['id'] == 'task:method_choice' and kwargs['proposal_ref'] == proposal_ref
        agent.costs['compute_cost'] += 3
        return {'status': 'staged_method_practice_completed', 'execution_complete': True}
    monkeypatch.setattr(runner, 'record_staged_method_practice', record)
    def route(*args):
        common.costs['compute_cost'] += 5
        return {'method_action_research': {'status': 'method_path_verified_pending_independent_validation'}}
    common._execute = route
    result = runner.run(output)
    assert result['compute_cost'] == 9 and result['remaining_compute'] == 491
    assert result['status'] == 'composition_ready_for_successor_registration'
    assert load(formation / 'receipt.json') == formed
    assert load(output / 'closed_source_costs.json')['cumulative_protocol_compute_interval'] == [3009, 4009]
    assert load(output / 'practice_return/receipt.json')['compute_cost'] == 8
