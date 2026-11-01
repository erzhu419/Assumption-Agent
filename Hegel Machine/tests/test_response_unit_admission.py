"""Unfunded new measurements stop before paying for a stranded response plan."""

from dataclasses import replace
from pathlib import Path

import pytest

from scripts import review_paid_object_pool as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry
from test_paid_instrument_planning import paid_scene, forbid_science_restart


def updated_scene(tmp_path, monkeypatch, *, balance=221, equivalent=False):
    request, source, _, _, _ = paid_scene(tmp_path, monkeypatch, balance=balance)
    inherited = load(request)
    candidates = load(source / 'new_entries.json')
    if equivalent:
        a, b = Program('feature', name='m_a'), Program('feature', name='m_b')
        for index, terms in enumerate(((a, b), (b, a))):
            entry = decode_entry(candidates[index])
            entry['candidate'] = replace(entry['candidate'], program=Program('add', children=terms))
            candidates[index] = encode_entry(entry)
        save(source / 'new_entries.json', candidates)
    # 187 occurrences, 186 actual programs: the quote must retain exact-Program
    # sharing already implemented by the full-pool suffix consumer.
    extras = [encode_entry({'candidate': Candidate(f'old{index}', Program('const', value=value), 'native'),
        'training_loss': 0., 'validation_loss': 0., 'research_program': {'kind': 'old_response', 'fit_rows': 4}})
        for index, value in enumerate([*range(1, 183), 1])]
    save(source / 'extra_pool.json', extras)
    pool = load(source / 'paid_pool/receipt.json')
    pool['scores'].extend({'candidate_refs': [runner.ref(source / 'extra_pool.json', index)], 'mse': 0.}
        for index in range(len(extras)))
    save(source / 'paid_pool/receipt.json', pool)
    frontier = load(source / 'retained_frontier.json')
    save(source / 'retained_frontier.json', dict(frontier, complete_current_pool_occurrences=187))
    binding_ref = runner.ref(tmp_path / 'bindings.json')
    save(Path(binding_ref['file']), {name: Program('feature', name=control).to_dict()
        for name, control in (('m_a', 'c_a'), ('m_b', 'c_b'))})
    evidence_ref = runner.ref(tmp_path / 'old_response.json')
    save(Path(evidence_ref['file']), {'status': 'distinguished'})
    previous = load(source / 'registration.json')
    save(source / 'registration.json', {**previous, 'binding_ref': binding_ref,
        'response_evidence_ref': evidence_ref,
        **{key: inherited[key] for key in ('object_model_ref', 'instrument_models_ref', 'experiment_methods_ref')}})
    receipt = load(source / 'receipt.json')
    save(source / 'receipt.json', {**receipt, 'status': 'paid_interval_updated',
        'candidate_refs': [runner.ref(source / 'new_entries.json', index) for index in (0, 1)]})
    save(request, {'ledger_ref': inherited['ledger_ref'], 'source_cost_refs': [inherited['source_costs_ref']],
        'reserved_compute': 96})
    return source, request


def test_221_balance_stops_before_planning_or_freezing_and_preserves_paid_source(tmp_path, monkeypatch):
    source, request = updated_scene(tmp_path, monkeypatch)
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    forbid_science_restart(monkeypatch)
    monkeypatch.setattr(Program, 'predict', lambda *a, **kw: pytest.fail('Admission must not predict'))
    monkeypatch.setattr(runner, 'freeze_entry', lambda *a: pytest.fail('Unfunded plan must not freeze'))
    output = tmp_path / 'admission'
    result = runner.register_response_plan(source, output, request=request)
    assert result['status'] == 'complete_response_unit_unfunded'
    assert result['remaining_compute'] == result['available_compute'] == 221
    quote = result['response_unit_lower_bound']
    assert quote['unique_pool_programs'] == quote['pool_consume_compute_lower_bound'] == 186
    assert quote['required_compute_lower_bound'] == 282
    assert quote['minimum_new_rows'] == 1 and not quote['algebraically_shared']
    assert not quote['sufficient_quote'] and not result['scientific_refutation']
    assert result['compute_cost'] == result['new_compute_cost'] == result['new_observations'] == 0
    assert not result['independent_support']
    assert load(output / 'admission.json') == result
    assert set(path.name for path in output.iterdir()) == {'admission.json'}
    assert all(path.read_bytes() == content for path, content in protected.items())


def test_native_algebraically_shared_route_remains_admitted_and_free(tmp_path, monkeypatch):
    source, request = updated_scene(tmp_path, monkeypatch, equivalent=True)
    forbid_science_restart(monkeypatch)
    monkeypatch.setattr(Program, 'predict', lambda *a, **kw: pytest.fail('Shared algebra needs no forecasts'))
    output = tmp_path / 'shared'
    cfg = runner.register_response_plan(source, output, request=request)
    quote = cfg['response_unit_lower_bound']
    assert quote['algebraically_shared'] and quote['minimum_new_rows'] == 0
    assert quote['required_compute_lower_bound'] == cfg['reserved_compute'] == 96
    assert cfg['compute_limit'] == 125
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_response_plan(output)
    assert result['forecast_status'] == 'algebraically_shared'
    assert result['compute_cost'] == result['new_observations'] == 0
    assert result['remaining_compute'] == 221


def test_funded_distinct_response_keeps_original_planning_contract(tmp_path, monkeypatch):
    source, request = updated_scene(tmp_path, monkeypatch, balance=2000)
    freezes = []
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args: freezes.append(args))
    forbid_science_restart(monkeypatch)
    monkeypatch.setattr(Program, 'predict', lambda *a, **kw: pytest.fail('Registration must not predict'))
    output = tmp_path / 'funded'
    cfg = runner.register_response_plan(source, output, request=request)
    assert cfg['response_unit_lower_bound']['required_compute_lower_bound'] == 282
    assert not cfg['response_unit_lower_bound']['sufficient_quote']
    assert cfg['compute_limit'] == 1904 and cfg['reserved_compute'] == 96
    assert cfg['planning_kind'] == 'updated_interval_methods'
    assert cfg['candidate_refs'] == load(source / 'receipt.json')['candidate_refs']
    assert len(freezes) == 1 and (output / 'registration.json').exists()
    assert not (output / 'admission.json').exists()
