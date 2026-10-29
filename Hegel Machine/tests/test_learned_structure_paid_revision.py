"""Paid successor revision keeps its row cohorts, pool and original allowance."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget
from hegel_machine.cognition.object_continuation import encode_entry
from scripts import revise_learned_structure_successor as runner


def claim(value, loss=0.):
    return {'candidate': Candidate('same_id', Program('const', value=value), 'paid'),
            'kind': 'paid', 'family_id': 'T03', 'training_loss': loss, 'validation_loss': loss}


def score_scene():
    entries = [claim(1., 1.), claim(2., 2.), claim(3., 3.)]
    programs = [item['candidate'].program for item in entries]
    return {'pool': [*entries, deepcopy(entries[0])],
        'scene': {'training': [None] * 125, 'validation': [None] * 24},
        'exposed': [None] * 48, 'frame': {'fallback': 7.},
        'working': dict(zip(programs, (5., 6., 7.))),
        'exposed_scores': dict(zip(programs[:2], (4., 9.)))}, programs


def test_paid_scores_weight_actual_cohorts_and_only_score_missing_exposure(monkeypatch):
    scene, programs = score_scene()
    scored = []

    def missing_score(program, rows, fallback, quantile):
        assert program == programs[2] and rows is scene['exposed']
        assert fallback == 7. and quantile == .5
        scored.append(program)
        return 16.

    monkeypatch.setattr(runner, 'loss', missing_score)
    budget = SearchBudget(48)
    scores, exposed, cost = runner.paid_pool_scores(scene, budget)
    assert len(scores) == 3  # Equal candidate IDs do not collapse different programs.
    assert scores == {program: pytest.approx((125 * fit + 24 * work + 48 * test) / 197)
        for program, fit, work, test in zip(programs, (1., 2., 3.), (5., 6., 7.), (4., 9., 16.))}
    assert scored == [programs[2]] and cost == budget.spent == 48
    assert exposed == dict(zip(programs, (4., 9., 16.)))
    assert scene['exposed_scores'] == dict(zip(programs[:2], (4., 9.)))


def test_unfunded_paid_pool_does_not_return_a_complete_comparison(monkeypatch):
    scene, _ = score_scene()
    monkeypatch.setattr(runner, 'loss', lambda *args: pytest.fail('An unfunded row pass must not run'))
    budget = SearchBudget(47)
    with pytest.raises(RuntimeError, match='complete retained comparison is not funded'):
        runner.paid_pool_scores(scene, budget)
    assert budget.spent == 0


@pytest.mark.parametrize('select_shared_parent', [False, True])
def test_restore_keeps_common_pool_once_and_all_arm_candidates(tmp_path, monkeypatch, select_shared_parent):
    common, replay, structure = [claim(1., 1.), claim(2., 2.)], claim(3., 3.), claim(4., 4.)
    runner.save(tmp_path / 'prepared_pool.json', [encode_entry(item) for item in common])
    for arm, item in [('baseline', replay), ('challenger', structure)]:
        (tmp_path / arm).mkdir()
        runner.save(tmp_path / arm / 'new_claims.json', [encode_entry(item)])
    common_refs = [runner.ref(tmp_path / 'prepared_pool.json', i) for i in range(2)]
    replay_ref = runner.ref(tmp_path / 'baseline/new_claims.json', 0)
    structure_ref = runner.ref(tmp_path / 'challenger/new_claims.json', 0)
    selected_ref = common_refs[0] if select_shared_parent else structure_ref
    for arm, extra, selected in [('baseline', replay_ref, replay_ref), ('challenger', structure_ref, selected_ref)]:
        runner.save(tmp_path / arm / 'comparison.json', {'full_comparison': True,
            'scores': [{'candidate_ref': location, 'loss': float(i + 1)}
                       for i, location in enumerate([*common_refs, extra])], 'selected_ref': selected})
    runner.save(tmp_path / 'registration.json', {'operations': {'baseline': 'program_replay',
                                                              'challenger': 'structural_research'}})
    validation_file = tmp_path / 'validation.json'
    runner.save(validation_file, {'transitions': [{'before': {'x': 1.}, 'after': {'x': 2.}}]})
    row = Observation({'x': 1.}, 1.)
    base = {'training': [row], 'validation': [row], 'transitions': [],
            'references': {'validation': [runner.ref(validation_file)]}}
    monkeypatch.setattr(runner, 'load_successor_scene', lambda source: deepcopy(base))
    marker = object()
    monkeypatch.setattr(runner, 'load_prepared', lambda *args, **kwargs: (marker, {'fallback': 7.}, []))
    (tmp_path / 'confirmation').mkdir()
    selected = common[0] if select_shared_parent else structure
    runner.save(tmp_path / 'confirmation/frozen_predictors.json', {
        'baseline': {'program': replay['candidate'].program.to_dict(), 'fallback': 7.},
        'challenger': {'program': selected['candidate'].program.to_dict(), 'fallback': 7.}})
    runner.save(tmp_path / 'confirmation/receipt.json', {'arms': {'baseline': {'mse': 3.},
                                                               'challenger': {'mse': 4.}}})
    runner.save(tmp_path / 'confirmation/observations.json', {'batch': {
        'observations': [asdict(row)], 'transitions': []}})

    restored = runner.restore(tmp_path)

    assert restored['agent'] is marker and len(restored['pool']) == 4
    assert restored['refs'] == [*common_refs, replay_ref, structure_ref]
    assert {item['candidate'].program for item in restored['pool']} == {
        item['candidate'].program for item in [*common, replay, structure]}
    assert restored['parent'] is restored['pool'][0 if select_shared_parent else 3]
    assert restored['parent']['fallback'] == 7.
    assert len(restored['paid']) == 3 and len(restored['transitions']) == 1


def test_registration_uses_only_original_unspent_allowance(tmp_path, monkeypatch):
    source, output = tmp_path / 'source', tmp_path / 'revision'
    source.mkdir()
    old = {'common_compute_limit': 90096, 'per_arm_total_compute_limit': 89956,
        'operations': {'baseline': 'program_replay', 'challenger': 'structural_research'},
        'source_task_ref': {'file': 'original_task.json'}, 'quality_threshold': 1e-7,
        'confirmation_rows': 48, 'confirmation_seed': 1111}
    previous = {'compute_cost': 127991, 'observation_cost': 197, 'measurement_cost': 2364}
    runner.save(source / 'registration.json', old)
    runner.save(source / 'receipt.json', {'cumulative_costs': previous,
                                         'source_cost_refs': [{'file': 'older_costs.json'}]})
    frozen = []
    monkeypatch.setattr(runner, 'freeze_runtime', lambda target, original: frozen.append((target, original)))
    monkeypatch.setattr(runner, 'write_text_incrementally', lambda *args, **kwargs: None)
    registered = runner.register(source, output)
    assert registered['total_compute_limit'] == 270008 and registered['remaining_compute'] == 142017
    assert registered['previous_costs'] == previous
    assert registered['source_task_ref'] == old['source_task_ref']
    assert registered['confirmation_rows'] == 48
    assert frozen == [(output, source)]


@pytest.mark.parametrize('paid_passes', [False, True])
def test_fresh_readout_is_only_acquired_after_paid_gate(tmp_path, monkeypatch, paid_passes):
    original, selected = claim(0., 1.), claim(1., 0.)
    rows = [Observation({'x': 1.}, 1.)] * 3
    selected_values = [1. if paid_passes else 0.] * len(rows)
    result = {'status': 'improved', 'selected': selected, 'selected_predictions': selected_values,
        'full_comparison': True, 'new_claims': [selected], 'claims': [original, selected],
        'compared_claims': [original, selected]}
    agent = SimpleNamespace(config=SimpleNamespace(max_candidates=4),
        _object_scope_claims=lambda *args: [], _assess=lambda *args: [],
        _revise_paid_structure=lambda *args, **kwargs: result)
    scene = {'agent': agent, 'scene': {'spec': SimpleNamespace(risk_quantile=.5)},
        'paid': rows, 'parent': original, 'frame': {'fallback': 0.}, 'pool': [original],
        'refs': [{'file': 'old_pool.json', 'json_path': [0]}], 'transitions': [],
        'exposed': rows[-1:], 'exposed_scores': {original['candidate'].program: 1.},
        'registration': {'task': {}}}
    cfg = {'source': 'old_source', 'remaining_compute': 500, 'scope_fit_compute_limit': 0,
        'active_wall_seconds': 30, 'threshold': 1e-7, 'confirmation_rows': 2,
        'confirmation_seed': 42, 'previous_costs': {'compute_cost': 100}, 'source_cost_refs': []}
    runner.save(tmp_path / 'registration.json', cfg)
    monkeypatch.setattr(runner, 'ROOT', tmp_path / 'runtime')
    monkeypatch.setattr(runner, 'restore', lambda source: scene)
    monkeypatch.setattr(runner, 'paid_pool_scores', lambda scene, budget: (
        {original['candidate'].program: 1.}, {}, 0))
    monkeypatch.setattr(runner, 'predict_rows', lambda program, rows: [program.value] * len(rows))
    acquired = []

    @dataclass
    class Batch:
        observations: list

    class Laboratory:
        observation_cost = measurement_cost = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def random_probe(self, seed, *, length, name):
            return Probe(name, tuple({'x': 1.} for _ in range(length)))

        def acquire(self, probe, channels):
            assert paid_passes
            assert runner.load(tmp_path / 'selection.json')['confirmation_admitted']
            assert (tmp_path / 'frozen_predictors.json').exists()
            acquired.append(probe.cost)
            self.observation_cost, self.measurement_cost = probe.cost, 12 * probe.cost
            return Batch(rows[:probe.cost])

    monkeypatch.setattr(runner, 'AnonymousObjectLab', Laboratory)
    receipt = runner.run(tmp_path)
    assert receipt['confirmation_admitted'] is paid_passes
    assert acquired == ([2] if paid_passes else [])
    assert receipt['observation_cost'] == (2 if paid_passes else 0)
    assert receipt['confirmation'] is None if not paid_passes else receipt['confirmation']['selected']['passed']
