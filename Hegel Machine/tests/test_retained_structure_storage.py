"""Paid structural knowledge borrows source rows without redoing science."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.condition_learning import prediction_matches
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, TaskSpec
from scripts import retain_confirmed_structure as storage


def forbid_science(*args, **kwargs):
    raise AssertionError('Retaining paid knowledge must not predict or fit again')


def test_paid_support_mask_matches_retained_tolerance_and_history(monkeypatch):
    program = Program('lag', name='x', lag=1)
    rows = [Observation({'x': 3.}, 0.),
            Observation({'x': 4.}, .5e-6, ({'x': 0.},)),
            Observation({'x': 5.}, 10. + 4e-6, ({'x': 10.},)),
            Observation({'x': 6.}, -10. - 4e-6, ({'x': -10.},))]
    # The paid evaluator records its fallback for an unavailable history;
    # such a row cannot become positive support even when that residual is zero.
    errors = [0. if not row.history else (program.predict(row) - row.target) ** 2 for row in rows]
    expected = [prediction_matches(program, row) for row in rows]
    assert expected == [False, True, True, True]
    monkeypatch.setattr(Program, 'predict', forbid_science)
    assert storage.paid_support_mask(program, rows, errors) == expected


def test_paid_support_mask_does_not_guess_unresolved_residuals(monkeypatch):
    program = Program('feature', name='x')
    row = Observation({'x': 2.0001}, 2.)
    assert not prediction_matches(program, row)
    error = (program.predict(row) - row.target) ** 2
    monkeypatch.setattr(Program, 'predict', forbid_science)
    with pytest.raises(ValueError, match='do not resolve'):
        storage.paid_support_mask(program, [row], [error])
    with pytest.raises(ValueError, match='same complete source rows'):
        storage.paid_support_mask(program, [row], [])


def test_retain_exports_reference_delta_and_restores_exact_knowledge(tmp_path, monkeypatch):
    source, output = tmp_path / 'source', tmp_path / 'retained'
    source.mkdir()
    spec = TaskSpec('paid_structure', 'Stored structural response', ('x',), (), ())
    term = Program('lag', name='x', lag=1)
    program = _sum_terms([term], [2.], .25)
    rows = [Observation({'x': 1.}, .25),
            Observation({'x': 2.}, 2.25, ({'x': 1.},)),
            Observation({'x': 3.}, 4.25, ({'x': 1.}, {'x': 2.}))]
    entry = {'candidate': Candidate('paid_structure', program, 'representation'),
        'kind': 'structural:retained', 'family_id': 'T12', 'training_loss': 0.,
        'validation_loss': 0., 'structure': {'compute_cost': 87, 'fits': [{
            'family': 'retained', 'fit_terms': [term.to_dict()],
            'coefficients': [.25, 2.]}]}}
    old = KnowledgeLibrary()
    old.learn(Candidate('old', Program('const', value=13.), 'observed'),
              TaskSpec('old', '', ('z',), (), ()),
              [Observation({'z': 99.}, 13.)], paid_support_matches=[True])
    before = storage.plain(old.to_dict())
    historical_ref = {'store': str(tmp_path / 'history'), 'unit': 'old', 'sequence': 7}
    config = {'root': str(tmp_path / 'scene'), 'acceptance': 'paid_acceptance',
        'previous': 'paid_previous', 'historical_knowledge_ref': historical_ref,
        'seed': 17, 'tolerance': 1e-7, 'compute_limit_including_previous': 1000}
    receipt = {'confirmation': {'selected_mse': 0., 'rows': 24}}
    batch_file = source / 'paid.json'
    storage.save(batch_file, {'probe': {'probe_id': 'paid_probe'},
                             'batch': {'channels': ['fine'], 'rows': [asdict(row) for row in rows]}})
    source_before = batch_file.read_bytes()
    scene = {'report': {}, 'paid_rows': rows,
        'registration': {'tasks': [{'seed': 17}]},
        'source_refs': {'paid_trajectories': [storage.ref(batch_file, 'batch')]}}
    storage.save(source / 'registration.json', config)
    monkeypatch.setattr(storage, 'source_inputs', lambda path: (
        config, receipt, deepcopy(entry), {'selected_paid_errors': [0., 0., 0.]}, scene))

    def historical(reference):
        assert reference == historical_ref
        return deepcopy(before), {'compute_cost': 87}, 7

    monkeypatch.setattr(storage, 'historical_knowledge', historical)
    monkeypatch.setattr(storage, 'load_paid_scene', lambda *args: scene)
    monkeypatch.setattr(storage, 'AnonymousObjectLab', lambda **settings:
        nullcontext(SimpleNamespace(spec_for=lambda channels: spec)))
    monkeypatch.setattr(Program, 'predict', forbid_science)
    monkeypatch.setattr('hegel_machine.cognition.research_programs._Fitter.fit', forbid_science)

    result = storage.retain(source, output)
    assert result['readback_equal'] and result['duplicate_confirmation_reused']
    assert result['new_predictions'] == result['new_fits'] == result['new_observations'] == 0
    assert result['support_examples_saved_by_reference'] == 1
    saved = storage.load(output / 'knowledge_delta.json')
    assert saved['support_example_refs'] == [
        {'concept_index': 1, 'scope_index': 0, 'paid_row_indices': [1, 2]}]
    assert set(saved['changes']['concepts']['updated']) == {'1'}
    assert 'support_examples' not in saved['changes']['concepts']['updated']['1']['scopes'][0]
    serialized = (output / 'knowledge_delta.json').read_text()
    assert '"history":' not in serialized and '"target":' not in serialized
    restored = storage.load_retained_knowledge(output)
    assert storage.plain(restored.concepts[0].to_dict()) == before['concepts'][0]
    assert restored.concepts[1].scopes[0].support_examples == storage.plain(
        [asdict(row) for row in rows[1:]])
    assert restored.meta_priors[-1]['probe_ids'] == ['paid_probe']
    assert batch_file.read_bytes() == source_before
    frozen_files = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert storage.retain(source, output) == result
    assert {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()} == frozen_files
