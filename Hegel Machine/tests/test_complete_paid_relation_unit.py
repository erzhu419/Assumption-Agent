"""Check the complete-unit ledger and retained state without buying observations."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import continue_paid_relation_unit as unit
from scripts.run_native_decision_pair import ref
from scripts.run_partial_role_parent import load, save
from test_paid_object_pool import canonical_confirmation_scene


def prepared(tmp_path, monkeypatch, *, admitted=False, maximum_steps=2, compute=100000):
    source, output = tmp_path / 'paid_source', tmp_path / 'complete_unit'
    latest = tmp_path / 'latest_method_return'
    save(latest / 'common/questions.json', [{'id': 'task:method_choice', 'version': 61}])
    save(source / 'registration.json', {'prior_method_review': str(tmp_path / 'stale_topic'),
        'experiment_methods_ref': ref(source / 'methods.json'), 'active_wall_seconds': 1800})
    save(source / 'methods.json', [{'id': f'method_{i}'} for i in range(7)])
    save(source / 'receipt.json', {'status': 'paid_interval_updated', 'execution_complete': True,
        'full_comparison': True, 'selected_ref': ref(source / 'selected.json'),
        'remaining_compute': 221, 'confirmation_gate': {'admitted': admitted}})
    save(source / 'paid_pool/receipt.json', {'unique_programs': 186, 'candidate_occurrences': 277})
    costs = {'cumulative_protocol_compute_interval': [2868125, 4308715],
        'cumulative_protocol_observations': 4749, 'parent_separate_suballocation_remaining': 20281,
        'diagnostic_suballocation_remaining': 5334,
        'previous_accounts_unchanged': {'older': 341120},
        'joint_continuation_remaining': 221}
    save(tmp_path / 'costs.json', costs)
    save(tmp_path / 'protocol.json', {'authorized_limits': {'compute_cost': 6500000,
        'observation_cost': 8000, 'model_calls': 0}})
    request = tmp_path / 'request.json'
    save(request, {'source': str(source), 'source_costs_ref': ref(tmp_path / 'costs.json'),
        'budget_protocol_ref': ref(tmp_path / 'protocol.json'), 'maximum_compute': compute,
        'maximum_experiments': maximum_steps, 'maximum_observations': 5 * maximum_steps + 48,
        'active_wall_seconds': 1800, 'prior_method_review': str(latest),
        'confirmation_protocol': {'length': 48, 'trajectories': 1,
            'maximum_attempts': 1, 'random_seed_offset': 1017017}})
    monkeypatch.setattr(unit, 'freeze_entry', lambda *args: None)
    monkeypatch.setattr(unit, 'write_text_incrementally', lambda *args, **kwargs: None)
    return source, output, request, latest


def test_registration_reserves_old_accounts_and_never_borrows_old_221(tmp_path, monkeypatch):
    source, output, request, _ = prepared(tmp_path, monkeypatch)
    protected = {p: p.read_bytes() for p in [source / 'receipt.json', tmp_path / 'costs.json']}
    cfg = unit.register(request, output)
    assert cfg['unallocated_before'] == 1824550
    assert cfg['preserved_balances'] == 366735
    assert cfg['original_science_remaining'] == 221
    assert cfg['maximum_compute'] == 100000
    assert all(p.read_bytes() == text for p, text in protected.items())
    changed = load(request)
    changed['maximum_compute'] = cfg['unallocated_before'] + 1
    save(request, changed)
    with pytest.raises(ValueError, match='registered balance'):
        unit.register(request, tmp_path / 'overspend')
    assert not (tmp_path / 'overspend').exists()


def test_previous_complete_unit_balance_cannot_be_allocated_twice(tmp_path, monkeypatch):
    _, output, request, _ = prepared(tmp_path, monkeypatch, compute=1750000)
    costs = load(tmp_path / 'costs.json')
    costs['complete_relation_unit_remaining'] = 100000
    save(tmp_path / 'costs.json', costs)
    with pytest.raises(ValueError, match='registered balance'):
        unit.register(request, output)
    assert not output.exists()


@pytest.mark.parametrize(('admitted', 'steps', 'remaining', 'expected'), [
    (True, 64, 6096, 'confirm'),
    (False, 64, 500000, 'experiment_limit'),
    (False, 0, 221, 'complete_segment_unfunded'),
    (False, 0, 500000, 'research'),
])
def test_native_gate_and_whole_segment_limits(admitted, steps, remaining, expected):
    assert unit.next_action({'confirmation_gate': {'admitted': admitted}},
        steps, 64, remaining, 186) == expected


def staged_native_entries(monkeypatch, output, *, admitted_after=True):
    """Stages store distinct global and local ledgers, as the real entries do."""
    seen = {'plans': [], 'returns': [], 'confirmations': []}

    def register_plan(source, target, *, request):
        supplied, prior = load(request), load(source / 'registration.json')
        pool = load(source / 'paid_pool/receipt.json')
        assert pool['candidate_occurrences'] >= 277
        assert len(unit.review.read_method_reference(prior['experiment_methods_ref'], source)) == 7
        seen['plans'].append((source, supplied, pool))
        save(target / 'registration.json', {**prior, **supplied, 'source': str(source),
            'available_compute': unit.review.read_method_reference(supplied['ledger_ref'], source)})

    def register_probe(request, target):
        supplied = load(request)
        prior = load(Path(supplied['source']) / 'registration.json')
        assert unit.review.read_method_reference(supplied['ledger_ref'], target) == prior['available_compute'] - 10
        save(target / 'registration.json', {**prior, **supplied,
            'available_compute': unit.review.read_method_reference(supplied['ledger_ref'], target)})

    def register_interval(source, target, *, request):
        prior, actual = load(source / 'registration.json'), load(source / 'receipt.json')
        assert load(request)['partition_source'] == prior['source_before_plan']
        save(target / 'registration.json', {**prior,
            'available_compute': actual['remaining_compute']})

    def register_confirm(source, target, *, request):
        supplied = load(request)
        seen['confirmations'].append(supplied)
        save(target / 'registration.json', {**load(source / 'registration.json'), **supplied,
            'active_wall_seconds': 1800,
            'available_compute': unit.review.read_method_reference(supplied['ledger_ref'], source)})

    def execute(argv, **kwargs):
        mode, target = argv[-3], Path(argv[-1])
        cfg = load(target / 'registration.json')
        if mode == 'run-response-plan':
            cfg['source_before_plan'] = cfg['source']
            save(target / 'registration.json', cfg)
            save(target / 'plan.json', {'probe': {'sequence': [{'x': 1}] * 3}})
            status, cost, obs = 'response_distinguishing_experiment', 10, 0
        elif mode == 'run-instrument-probe':
            status, cost, obs = 'response_distinguished', 20, 3
        elif mode == 'run-interval-probe-return':
            status, cost, obs = 'paid_interval_updated', 30, 0
            previous = load(Path(cfg['source_before_plan']) / 'paid_pool/receipt.json')
            save(target / 'paid_pool/receipt.json', {'unique_programs': previous['unique_programs'] + 1,
                'candidate_occurrences': previous['candidate_occurrences'] + 1})
        else:
            assert mode == 'run-confirm'
            status, cost, obs = 'confirmed', 96, 48
        save(target / 'receipt.json', {'status': status, 'compute_cost': cost,
            'observation_cost': obs, 'measurement_cost': 12 * obs,
            'remaining_compute': cfg['available_compute'] - cost,
            'confirmation_gate': {'admitted': admitted_after},
            'forecast_status': 'predicted_distinction', 'selected_ref': ref(target / 'selected.json')})
        return SimpleNamespace(returncode=0)

    def ordinary_return(source, target, *, allocation_ref):
        allocation = unit.review.read_method_reference(allocation_ref, source)
        prior = load(source / 'registration.json')['prior_method_review']
        seen['returns'].append((source, prior, allocation))
        cost = 15 if len(seen['returns']) == 1 else 17
        receipt = {'compute_cost': cost, 'remaining_compute': allocation['maximum_compute'] - cost}
        save(target / 'receipt.json', receipt)
        return receipt

    for name, fn in [('register_response_plan', register_plan), ('register_instrument_probe', register_probe),
                     ('register_interval_probe_return', register_interval), ('register_confirmation', register_confirm),
                     ('return_parent', ordinary_return)]:
        monkeypatch.setattr(unit.review, name, fn)
    monkeypatch.setattr(unit.subprocess, 'run', execute)
    (output / 'runtime').mkdir()
    monkeypatch.setattr(unit, 'ROOT', output / 'runtime')
    return seen


def test_complete_chain_preserves_pool_and_latest_topic_with_global_confirmation_balance(tmp_path, monkeypatch):
    source, output, request, latest = prepared(tmp_path, monkeypatch)
    initial = load(source / 'receipt.json')
    unit.register(request, output)
    seen = staged_native_entries(monkeypatch, output)
    result = unit.run(output)
    assert result['status'] == 'confirmed'
    assert result['compute_cost'] == 188 and result['remaining_compute'] == 99812
    assert result['observation_cost'] == 51
    assert seen['returns'][0][1] == str(latest)
    assert seen['returns'][1][1] == str(output / 'method_return')
    supplied = seen['confirmations'][0]
    assert unit.review.read_method_reference(supplied['ledger_ref'], output) == 99925
    assert supplied['incumbent_ref'] == initial['selected_ref']
    assert load(source / 'receipt.json') == initial
    assert result['latest_method_review'] == str(output / 'confirmation_return')
    costs = unit.review.read_method_reference(result['closed_source_costs_ref'], output)
    assert costs['cumulative_protocol_compute_interval'] == [2868313, 4308903]
    assert costs['joint_continuation_remaining'] == 221
    assert costs['cumulative_protocol_observations'] == 4800
    assert unit.run(output) == result
    assert len(seen['confirmations']) == 1


def test_step_cap_keeps_growing_pool_and_returns_without_confirmation(tmp_path, monkeypatch):
    source, output, request, latest = prepared(tmp_path, monkeypatch)
    unit.register(request, output)
    seen = staged_native_entries(monkeypatch, output, admitted_after=False)
    result = unit.run(output)
    assert result['status'] == 'experiment_limit' and result['steps'] == 2
    assert result['compute_cost'] == 135 and result['observation_cost'] == 6
    assert [x[2]['candidate_occurrences'] for x in seen['plans']] == [277, 278]
    assert not seen['confirmations']
    assert seen['returns'][0][1] == str(latest)


def test_preregistered_starting_parent_is_the_actual_frozen_comparator(tmp_path, monkeypatch):
    source, request, _, _ = canonical_confirmation_scene(tmp_path, monkeypatch)
    previous = load(source / 'registration.json')
    original = unit.review.retained_entry(previous['parent_ref'], source, 0)
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Candidate
    from hegel_machine.cognition.object_continuation import encode_entry
    original['candidate'] = Candidate('unit_start', Program('const', value=23.), 'retained unit start')
    initial = tmp_path / 'preregistered_parent.json'
    save(initial, encode_entry(original))
    save(request, {**load(request), 'incumbent_ref': ref(initial)})
    output = tmp_path / 'confirmation'
    cfg = unit.review.register_confirmation(source, output, request=request)
    frozen = load(output / 'frozen_entries.json')
    assert cfg['incumbent_ref'] == ref(initial)
    assert frozen['incumbent']['source_ref'] == ref(initial)
    assert frozen['incumbent']['candidate']['program'] == Program('const', value=23.).to_dict()
    assert frozen['selected']['source_ref'] == load(source / 'receipt.json')['selected_ref']
    assert not load(output / 'confirmation_plan.json')['responses_obtained']
