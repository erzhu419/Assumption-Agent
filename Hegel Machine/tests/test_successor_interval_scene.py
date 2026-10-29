"""Interval continuation preserves source references and trajectory resets."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
from hegel_machine.cognition.object_continuation import encode_entry
from scripts import successor_interval_scene as scene_io
from scripts.run_partial_role_parent import save


def entry(value):
    return {'candidate': Candidate('reused_id', Program('const', value=value), 'paid'),
            'kind': 'paid', 'family_id': 'T03', 'training_loss': value, 'validation_loss': value}


def test_pair_rows_starts_new_command_history_for_each_probe():
    first = [Observation({'m': 1.}, 2.), Observation({'m': 3.}, 4., ({'m': 1., '__target__': 2.},))]
    paired = scene_io.pair_rows([{'c': 5.}, {'c': 6.}], first)
    assert paired[1][0].history == ({'c': 5., '__target__': 2.},)
    second = Observation({'m': 1.}, 7.)
    restarted = scene_io.pair_rows([{'c': 5.}], [second])
    assert restarted[0][0].history == () and restarted[0][1] is second
    assert paired[1][1] is first[1]


def test_restore_keeps_complete_pool_and_each_command_history(tmp_path, monkeypatch):
    source, previous = tmp_path / 'revision', tmp_path / 'previous'
    source.mkdir()
    previous.mkdir()
    training, validation, old_exposed, exposed = [], [], [], []
    batches = [training, validation, old_exposed, exposed]
    files = [previous / 'train.json', previous / 'validate.json',
             previous / 'confirmation/observations.json', source / 'confirmation_observations.json']
    for serial, (rows, path, count) in enumerate(zip(batches, files, (2, 1, 2, 2))):
        sequence, measured_history = [], []
        for step in range(count):
            point, features = {'control': serial * 10. + step}, {'measured': serial + step * .1}
            row = Observation(features, serial + step + .5, tuple(measured_history))
            rows.append(row)
            sequence.append(point)
            measured_history.append({**features, '__target__': row.target})
        batch = {'observations': [asdict(row) for row in rows], 'transitions': [{'batch': serial}]}
        document = {'probe': {'sequence': sequence, 'probe_id': str(serial)}}
        document.update({'batch': batch} if serial >= 2 else batch)
        path.parent.mkdir(exist_ok=True)
        save(path, document)
    candidates = [entry(float(i)) for i in range(4)]
    paths = [previous / 'pool.json', source / 'scope_claims.json', source / 'new_claims.json']
    save(paths[0], [encode_entry(item) for item in candidates[:2]])
    save(paths[1], [encode_entry(candidates[2])])
    save(paths[2], [encode_entry(candidates[3])])
    refs = [scene_io.ref(paths[0], 0), scene_io.ref(paths[0], 1),
            scene_io.ref(paths[1], 0), scene_io.ref(paths[2], 0)]
    save(source / 'selected_entry.json', encode_entry(candidates[3]))
    save(source / 'selection.json', {'full_comparison': True,
        'scores': [{'candidate_ref': location, 'loss': float(i + 1)} for i, location in enumerate(refs)],
        'selected_ref': scene_io.ref(source / 'selected_entry.json')})
    save(source / 'frozen_predictors.json', {'selected': candidates[3]['candidate'].program.to_dict(),
        'incumbent': candidates[1]['candidate'].program.to_dict(), 'fallback': 7.})
    save(source / 'registration.json', {'source': str(previous), 'total_compute_limit': 270008,
        'previous_costs': {'compute_cost': 127991, 'observation_cost': 5, 'measurement_cost': 60}})
    save(source / 'receipt.json', {'cumulative_compute': 177215, 'observation_cost': 2,
        'measurement_cost': 24, 'source_cost_refs': [scene_io.ref(previous / 'receipt.json')],
        'confirmation': {'selected': {'mse': .1, 'predictions': [3., 3.]},
                         'incumbent': {'mse': .2, 'predictions': [1., 1.]}}})
    agent = object()
    prior = {'agent': agent, 'frame': {'fallback': 7.}, 'paid': [*training, *validation, *old_exposed],
        'transitions': [{'batch': i} for i in range(3)],
        'registration': {'task': {'seed': 123}, 'observation_limit': 512},
        'scene': {'spec': SimpleNamespace(risk_quantile=.5), 'references': {
            'training': [scene_io.ref(files[0])], 'validation': [scene_io.ref(files[1])]}}}
    monkeypatch.setattr(scene_io, 'restore_prior', lambda path: prior)
    monkeypatch.setattr(Program, 'predict', lambda *args, **kwargs: pytest.fail('Restore must not predict'))
    before = {str(path): path.read_bytes() for path in tmp_path.rglob('*.json')}

    recovered = scene_io.restore_interval_scene(source)

    assert recovered['agent'] is agent and len(recovered['pool']) == 4
    assert recovered['refs'] == refs and recovered['parent'] is recovered['pool'][3]
    assert recovered['parent']['fallback'] == 7.
    assert recovered['old_scores'] == {item['candidate'].program: float(i + 1)
                                      for i, item in enumerate(candidates)}
    assert recovered['exposed_scores'] == {candidates[3]['candidate'].program: .1,
                                          candidates[1]['candidate'].program: .2}
    assert recovered['paid_before'] == prior['paid'] and recovered['paid'] == [*prior['paid'], *exposed]
    assert recovered['original_registration'] == prior['registration']
    assert recovered['remaining_compute'] == 92793
    assert recovered['previous_costs'] == {'compute_cost': 177215, 'observation_cost': 7,
                                         'measurement_cost': 84}
    assert recovered['transitions'] == [{'batch': i} for i in range(4)]
    for serial, pairs in enumerate(recovered['paired_batches']):
        assert not pairs[0][0].history and not pairs[0][1].history
        for step, (command, measured) in enumerate(pairs):
            assert len(command.history) == step and measured == batches[serial][step]
            assert command.features == {'control': serial * 10. + step}
            if step:
                assert command.history == ({'control': serial * 10., '__target__': batches[serial][0].target},)
    assert {str(path): path.read_bytes() for path in tmp_path.rglob('*.json')} == before


def scoring_scene():
    pool = [entry(float(i)) for i in range(33)]
    programs = [item['candidate'].program for item in pool]
    return {'pool': pool, 'paid_before': [None] * 197, 'exposed': [None] * 48,
        'old_scores': {program: float(i + 1) for i, program in enumerate(programs)},
        'exposed_scores': {programs[0]: .1, programs[1]: .2},
        'parent': {'fallback': 7.}, 'spec': SimpleNamespace(risk_quantile=.5)}, programs


def test_scores_reuse_all197_and_two_exposed_programs(monkeypatch):
    scene, programs = scoring_scene()
    original = deepcopy(scene['old_scores'])
    called = []

    def score(program, rows, fallback, quantile):
        assert program not in programs[:2] and rows is scene['exposed']
        assert fallback == 7. and quantile == .5
        called.append(program)
        return 2.

    monkeypatch.setattr(scene_io, 'loss', score)
    budget = SearchBudget(1500, spent=12)
    scores, added = scene_io.score_retained(scene, budget)
    assert len(scores) == 33 and called == programs[2:]
    assert added == 31 * 48 == 1488 and budget.spent == 1500
    for i, program in enumerate(programs):
        assert scores[program] == pytest.approx((197 * (i + 1) + 48 * (.1 if i == 0 else .2 if i == 1 else 2.)) / 245)
    assert scene['old_scores'] == original and len(scene['exposed_scores']) == 2


def test_unfunded_full_pool_never_starts_an_unfinishable_score_pass(monkeypatch):
    scene, _ = scoring_scene()
    monkeypatch.setattr(scene_io, 'loss', lambda *args: pytest.fail('Missing full-pool allowance'))
    budget = SearchBudget(1499, spent=12)
    with pytest.raises(RuntimeError, match='complete retained interval comparison is not funded'):
        scene_io.score_retained(scene, budget)
    assert budget.spent == 12


def test_unconfirmed_answer_never_enters_retained_knowledge(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_io, 'restore_interval_scene',
        lambda *args: pytest.fail('Failed confirmation must not restore or mutate knowledge'))
    result = scene_io.save_confirmed_knowledge(None, None, None, [], [],
        {'selected': {'passed': False}}, tmp_path / 'new', source=tmp_path / 'old',
        new_observation_refs=[])
    assert result == {'stored': False, 'reason': 'confirmation_not_passed'}
    assert not list(tmp_path.iterdir())


def test_completed_hold_restores_full_pool_scores_and_methods_without_publishing(tmp_path, monkeypatch):
    from hegel_machine.cognition.memory import KnowledgeLibrary

    source, output = tmp_path / 'source', tmp_path / 'hold'
    source.mkdir()
    output.mkdir()
    library = KnowledgeLibrary(instrument_models=[{'instrument_id': 'old'}])
    concepts, priors = deepcopy(library.concepts), deepcopy(library.meta_priors)
    old_rows = [Observation({'m': 1.}, 2.), Observation({'m': 2.}, 3., ({'m': 1., '__target__': 2.},))]
    new_rows = [Observation({'m': 5.}, 6.), Observation({'m': 7.}, 8., ({'m': 5., '__target__': 6.},))]
    old_pairs = scene_io.pair_rows([{'c': 1.}, {'c': 2.}], old_rows)
    candidates = [entry(float(i)) for i in range(3)]
    save(source / 'pool.json', [encode_entry(item) for item in candidates[:2]])
    save(output / 'new_pool.json', [encode_entry(candidates[2])])
    refs = [scene_io.ref(source / 'pool.json', 0), scene_io.ref(source / 'pool.json', 1),
            scene_io.ref(output / 'new_pool.json', 0)]
    selected = {**candidates[2], 'validation_loss': .03, 'fallback': 7.}
    save(output / 'selected_entry.json', encode_entry(selected))
    observation_ref = scene_io.ref(output / 'observations.json')
    save(output / 'observations.json', {'probe': {'sequence': [{'c': 5.}, {'c': 7.}]},
        'batch': {'observations': [asdict(row) for row in new_rows], 'transitions': [{'new': True}]}})
    save(output / 'selection.json', {'full_comparison': True, 'scores': [
        {'candidate_ref': location, 'loss': score} for location, score in zip(refs, (.1, .2, .03))],
        'selected_ref': scene_io.ref(output / 'selected_entry.json'),
        'selected_paid_errors': [.03] * 4, 'new_observation_refs': [observation_ref],
        'actions': [{'action': 'plan_relation_experiment', 'plan_ref': scene_io.ref(output / 'old_plan.json')},
                    {'action': 'test_relation_experiment', 'observations_ref': observation_ref},
                    {'action': 'plan_relation_experiment', 'plan_ref': scene_io.ref(output / 'plan_13.json')}]})
    plan = {'probe': {'probe_id': 'paid_plan_13', 'sequence': [{'c': 9.}]},
        'hypotheses': [{'hypothesis_id': 'one', 'program': candidates[2]['candidate'].program.to_dict(),
                        'predictions': [2.]}], 'forecast_status': 'predicted_distinction',
        'predicted_fields': {'m': [9.]}, 'compute_cost': 42}
    save(output / 'plan_13.json', plan)
    save(output / 'registration.json', {'source': str(source), 'total_compute_limit': 270008,
        'previous_observations': 245, 'previous_measurement_cost': 2940})
    save(output / 'receipt.json', {'stop_reason': 'preserve_update_and_confirmation_allowance',
        'confirmation': None, 'cumulative_compute': 266778, 'observation_cost': 39,
        'measurement_cost': 468, 'radius_intervals': [{'radius_interval': [1., 1.000002]}],
        'source_cost_refs': [scene_io.ref(source / 'receipt.json')]})
    instruments = [{'instrument_id': 'new', 'status': 'supported'}]
    methods = [{'method_id': 'interval', 'status': 'supported'}]
    save(output / 'instrument_models.json', instruments)
    save(output / 'program_experiment_methods.json', methods)
    base = {'agent': SimpleNamespace(library=library), 'spec': object(), 'frame': {'fallback': 7.},
        'parent': {'fallback': 7.}, 'original_registration': {'task': {'seed': 3}},
        'paid': old_rows, 'exposed': old_rows[-1:], 'paired_batches': [old_pairs], 'paired_batch_refs': [],
        'transitions': [{'old': True}]}
    monkeypatch.setattr(scene_io, 'restore_interval_scene', lambda path: base)
    monkeypatch.setattr(Program, 'predict', lambda *a, **kw: pytest.fail('Restore predicted'))
    monkeypatch.setattr('hegel_machine.cognition.research_programs._Fitter.fit',
        lambda *a, **kw: pytest.fail('Restore fitted'))
    before = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}

    restored = scene_io.restore_interval_result(output)

    assert len(restored['pool']) == len(restored['scores']) == 3
    assert restored['refs'] == refs and restored['parent'] is restored['pool'][2]
    assert restored['parent']['validation_loss'] == .03
    assert restored['scores'] == {candidate['candidate'].program: score
        for candidate, score in zip(candidates, (.1, .2, .03))}
    assert restored['paid'] == old_rows + new_rows and restored['paid_errors'] == [.03] * 4
    assert restored['paired_batches'][1][0][0].history == ()
    assert restored['paired_batches'][1][1][0].history == ({'c': 5., '__target__': 6.},)
    assert restored['transitions'] == [{'old': True}, {'new': True}]
    assert restored['experiments'] == [tuple(old_rows[-1:]), tuple(new_rows)]
    assert restored['pending_plan']['probe'].sequence == ({'c': 9.},)
    assert restored['pending_plan']['hypotheses'] == plan['hypotheses']
    assert restored['pending_plan']['predicted_fields'] == plan['predicted_fields']
    assert restored['pending_plan']['compute_cost'] == 42
    assert restored['pending_plan_ref'] == scene_io.ref(output / 'plan_13.json')
    assert restored['calibration'] == instruments[-1]
    assert restored['knowledge_source'] == str(source)
    assert restored['knowledge_observation_refs'] == [observation_ref]
    assert restored['agent'].library.instrument_models == instruments
    assert restored['agent'].library.program_experiment_methods.to_dict() == methods
    assert library.concepts == concepts and library.meta_priors == priors
    assert restored['confirmation'] is None and restored['remaining_compute'] == 3230
    assert restored['previous_costs'] == {'compute_cost': 266778,
        'observation_cost': 284, 'measurement_cost': 3408}
    assert restored['source_scene'] is base
    assert {path: path.read_bytes() for path in tmp_path.rglob('*.json')} == before


@pytest.mark.parametrize('confirmation_filename', ['receipt.json', 'readout.json'])
def test_confirmation_retains_research_and_support_by_reference_without_science(
        tmp_path, monkeypatch, confirmation_filename):
    from tests.test_confirmed_native_knowledge import selection
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    selected, spec, rows = selection()
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    output.mkdir()
    hold = tmp_path / 'hold'
    hold.mkdir()
    old_path, hold_path, new_path = (source / 'paid.json', hold / 'paid.json',
                                    output / 'new_observations.json')
    for path, subset, name in ((old_path, rows[:3], 'old'),
                              (hold_path, rows[3:4], 'hold'), (new_path, rows[4:], 'new')):
        save(path, {'probe': {'probe_id': name}, 'batch': {
            'channels': ['fine'], 'observations': [asdict(row) for row in subset]}})
    baseline = KnowledgeLibrary(instrument_models=[{'instrument_id': 'old'}]).to_dict()

    def restore(_):
        return {'agent': SimpleNamespace(library=KnowledgeLibrary.from_dict(deepcopy(baseline))),
                'paid': rows[:3], 'paired_batch_refs': [scene_io.ref(old_path)]}

    monkeypatch.setattr(scene_io, 'restore_interval_scene', restore)
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7),
                                  library=KnowledgeLibrary.from_dict(deepcopy(baseline)))
    agent.library.instrument_models.append({'instrument_id': 'current', 'scale': 2.})
    agent.library.program_experiment_methods.records.append(
        {'method_id': 'measured_interval', 'status': 'supported', 'solver_strategy': 'interval'})
    monkeypatch.setattr(Program, 'predict', lambda *a, **kw: pytest.fail('Saved support predicted again'))
    monkeypatch.setattr('hegel_machine.cognition.research_programs._Fitter.fit',
        lambda *a, **kw: pytest.fail('Saving knowledge fitted again'))
    confirmation = {'rows': 24, 'selected': {'passed': True, 'mse': 1e-21}}
    confirmation_ref = scene_io.ref(output / confirmation_filename, 'confirmation')
    confirmation_options = ({'confirmation_ref': confirmation_ref}
                            if confirmation_filename != 'receipt.json' else {})
    save(output / confirmation_filename, {'confirmation': confirmation})
    support_refs = [scene_io.ref(hold_path), scene_io.ref(new_path)]

    result = scene_io.save_confirmed_knowledge(agent, selected, spec, rows, [0.] * len(rows),
        confirmation, output, source=source, new_observation_refs=support_refs,
        **confirmation_options)

    assert result['stored'] and result['readback_equal']
    assert result['support_rows'] == 5 and result['confirmation_rows'] == 24
    assert result['new_fits'] == result['new_predictions'] == result['new_observations'] == 0
    loaded = scene_io.load_confirmed_knowledge(output)
    assert scene_io.plain(loaded.to_dict()) == scene_io.plain(agent.library.to_dict())
    assert loaded.instrument_models == agent.library.instrument_models
    assert loaded.program_experiment_methods.to_dict() == agent.library.program_experiment_methods.to_dict()
    assert loaded.concepts[0].scopes[0].support_examples == scene_io.plain([asdict(row) for row in rows])
    assert loaded.meta_priors[0]['probe_ids'] == ['old', 'hold', 'new']
    assert loaded.meta_priors[0]['research_program']['confirmation_ref'] == confirmation_ref
    delta = scene_io.load(output / 'knowledge_delta.json')
    assert delta['confirmation_ref'] == confirmation_ref
    assert delta['source'] == str(source) and delta['new_observation_refs'] == support_refs
    assert {'instrument_models', 'program_experiment_methods'} <= set(delta['changes'])
    assert delta['support_example_refs'][0]['paid_row_indices'] == list(range(5))
    assert '"history"' not in (output / 'knowledge_delta.json').read_text()
    assert '"target"' not in (output / 'knowledge_delta.json').read_text()
    frozen = deepcopy(agent.library.to_dict())
    repeated = scene_io.save_confirmed_knowledge(agent, selected, spec, rows, [0.] * len(rows),
        confirmation, output, source=source, new_observation_refs=support_refs,
        **confirmation_options)
    assert repeated['reused'] and agent.library.to_dict() == frozen
