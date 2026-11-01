"""Restore paid instrument evidence without changing supported response models."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts import review_paid_object_pool as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry


def paid_scene(tmp_path, monkeypatch, *, balance=12000):
    source = tmp_path / 'supported_interval'
    common = source / 'paid_pool'
    measured = TaskSpec('anonymous', 'Paid sensors', ('m_a', 'm_b'), (), (),
        feature_bounds={'c_a': (-2., 2.), 'c_b': (-2., 2.)}, max_probe_length=8)
    save(common / 'public_spec.json', asdict(measured))
    refs = []
    for batch_index, values in enumerate(((-1., 0.), (0., 1.), (1., -1.))):
        history, rows, sequence = [], [], []
        for value in values:
            point = {'c_a': value, 'c_b': -value}
            fields = {'m_a': value, 'm_b': -value}
            rows.append(Observation(fields, value, tuple(history)))
            history.append({**fields, '__target__': value})
            sequence.append(point)
        location = runner.ref(common / f'probe{batch_index}.json')
        save(Path(location['file']), {'probe': {'probe_id': f'paid{batch_index}', 'sequence': sequence},
            'batch': {'observations': [asdict(row) for row in rows]}})
        refs.append(location)
    partial = {'complete': False, 'schema': None, 'model_id': 'measured_partial',
        'groups': [{'object_id': 'some_object', 'q': 'm_a', 'u': 'm_b',
                    'controls': {'q': 'c_a', 'u': 'c_b'}}],
        'control_effects': {'c_a': {'before_fields': ['m_a']}, 'c_b': {'before_fields': ['m_b']}},
        'focus_control': None, 'unresolved': ['some_object:x', 'some_object:f']}
    save(tmp_path / 'partial.json', partial)
    old_models = [{'model_id': 'earlier', 'status': 'supported_on_measured_trajectory',
                   'source_schema': None, 'fields': {}}]
    old_methods = [{'method_id': 'old_experiment', 'status': 'supported'}]
    save(tmp_path / 'instruments.json', old_models)
    save(tmp_path / 'methods.json', old_methods)
    entries = []
    feature = Program('feature', name='m_a')
    for index, cutoff in enumerate((1.25, 1.5, 1.75)):
        program = Program('if', children=(Program('lt', children=(feature, Program('const', value=cutoff))),
                                         feature, Program('feature', name='m_b')))
        entries.append(encode_entry({'candidate': Candidate(f'interval{index}', program, 'native'),
            'training_loss': 0., 'validation_loss': 0.,
            'parameter_index': 0, 'parameter_values': {'relation0': cutoff},
            'research_program': {'kind': 'saved_response', 'fit_rows': 4,
                'saved_coefficients': [2., 3., 5.], 'term_nodes': ['first', 'second']}}))
    save(source / 'new_entries.json', entries)
    old_ref = runner.ref(source / 'old_entry.json')
    save(Path(old_ref['file']), encode_entry({'candidate': Candidate('old', Program('const', value=0.), 'native'),
        'training_loss': 2 / 3, 'validation_loss': 2 / 3,
        'research_program': {'kind': 'old_response', 'fit_rows': 4}}))
    selected_ref = runner.ref(source / 'new_entries.json', 0)
    scores = [{'candidate_refs': [old_ref], 'mse': 2 / 3}]
    scores.extend({'candidate_refs': [runner.ref(source / 'new_entries.json', index)], 'mse': 0.}
                  for index in range(3))
    save(common / 'registration.json', {'data_refs': refs[:2], 'appended_observation_refs': refs[2:],
        'public_spec_ref': runner.ref(common / 'public_spec.json'),
        'training_rows': 4, 'feedback_rows': 2, 'comparison_rows': 6, 'quality_threshold': 1e-7})
    save(common / 'receipt.json', {'full_comparison': True, 'rows': 6, 'scores': scores})
    save(source / 'retained_frontier.json', {'full_comparison': True,
        'original_pool_ref': runner.ref(common / 'receipt.json'), 'appended_comparison_refs': [],
        'complete_current_pool_occurrences': 4, 'selected_ref': selected_ref, 'selected_mse': 0.})
    save(source / 'registration.json', {'source_task_ref': runner.ref(tmp_path / 'task.json'),
        'active_wall_seconds': 30, 'quality_threshold': 1e-7})
    save(source / 'receipt.json', {'status': 'paid_interval_supported', 'execution_complete': True,
        'full_comparison': True, 'selected_ref': selected_ref, 'selected_loss': 0.})
    save(tmp_path / 'ledger.json', {'remaining_compute': balance})
    request = tmp_path / 'request.json'
    save(request, {'source': str(source), 'object_model_ref': runner.ref(tmp_path / 'partial.json'),
        'instrument_models_ref': runner.ref(tmp_path / 'instruments.json'),
        'experiment_methods_ref': runner.ref(tmp_path / 'methods.json'),
        'ledger_ref': runner.ref(tmp_path / 'ledger.json', 'remaining_compute'),
        'source_costs_ref': runner.ref(tmp_path / 'ledger.json')})
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, 'write_text_incrementally', lambda *args, **kwargs: None)
    return request, source, partial, old_models, old_methods


def forbid_science_restart(monkeypatch):
    from hegel_machine.cognition import anonymous_object_lab, research_programs, transfer_partitions
    monkeypatch.setattr(anonymous_object_lab.AnonymousObjectLab, 'acquire',
        lambda *args, **kwargs: pytest.fail('Paid instrument planning must not acquire observations'))
    monkeypatch.setattr(research_programs._Fitter, 'fit',
        lambda *args, **kwargs: pytest.fail('Do not refit the supported response'))
    monkeypatch.setattr(transfer_partitions, 'continue_transfer_partition',
        lambda *args, **kwargs: pytest.fail('Do not repeat the completed interval research'))


def checked_fit(monkeypatch, partial, old_models, old_methods, calls, *, missing=False, fail=False):
    from hegel_machine.cognition import unified_research

    def fit(agent, command, measured, batches, budget, actions, *, required_fields=None, **kwargs):
        assert set(command.feature_names) == {'c_a', 'c_b'}
        assert set(measured.feature_names) == {'m_a', 'm_b'}
        assert measured.context['object_model'] == partial
        assert not measured.context.get('object_schema')
        assert agent.library.instrument_models == old_models
        assert agent.library.program_experiment_methods.records == old_methods
        assert [len(batch) for batch in batches] == [2, 2, 2]
        for batch in batches:
            assert batch[0][0].history == batch[0][1].history == ()
            assert batch[1][0].history == ({**batch[0][0].features, '__target__': batch[0][0].target},)
            assert batch[1][1].history == ({**batch[0][1].features, '__target__': batch[0][1].target},)
            for control, measured_row in batch:
                assert measured_row.features['m_a'] == control.features['c_a']
                assert measured_row.features['m_b'] == control.features['c_b']
        assert [pair[0].features['c_a'] for pair in batches[-1]] == [1., -1.]
        assert set(required_fields) == {'m_a', 'm_b'}
        calls.append(deepcopy(batches))
        assert budget.spend(7)
        if fail:
            raise RuntimeError('instrument inference interrupted')
        fields = {'m_a': {'program': Program('feature', name='c_a').to_dict()}}
        if not missing:
            fields['m_b'] = {'program': Program('feature', name='c_b').to_dict()}
        model = {'model_id': 'restored_instrument', 'status': 'supported_on_measured_trajectory',
                 'source_schema': None, 'fields': fields, 'compute_cost': 7}
        agent.library.instrument_models.append(model)
        return model

    monkeypatch.setattr(unified_research.UnifiedResearchAgent, '_fit_instruments', fit)


def test_paid_partial_bindings_feed_complete_response_plan_and_leave_world_pool_unchanged(tmp_path, monkeypatch):
    from hegel_machine.cognition import competing_object_experiments, relation_uncertainty

    request, source, partial, old_models, old_methods = paid_scene(tmp_path, monkeypatch)
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    output = tmp_path / 'instrument_plan'
    cfg = runner.register_instrument_plan(request, output)
    assert cfg['pool_occurrences'] == 4 and len(cfg['candidate_refs']) == 3
    assert cfg['instrument_training_rows'] == 4 and cfg['instrument_validation_rows'] == 2
    assert cfg['paired_trajectory_rows'] == [2, 2, 2]
    assert cfg['available_compute'] == 12000 and cfg['compute_limit'] == 12000 - 3096
    assert cfg['reserved_compute'] == 3096
    forbid_science_restart(monkeypatch)
    calls = []
    checked_fit(monkeypatch, partial, old_models, old_methods, calls)
    originals = load(source / 'new_entries.json')

    class Questions:
        def __init__(self, program, priors, budget, *, research_program):
            assert program.to_dict() == originals[0]['candidate']['program']
            assert research_program == originals[0]['research_program']
            assert budget.spend(2)

        def questions(self, entries):
            assert [entry['research_program'] for entry in entries] == [entry['research_program'] for entry in originals]
            return [{'question_id': 'remaining_relation'}]

    probe = Probe('predicted_distinction', ({'c_a': 1.6, 'c_b': 0.},))

    def response_plan(command, entries, budget, *, binding, serial, length, tolerance, questions, methods, task_id):
        assert set(command.feature_names) == {'c_a', 'c_b'}
        assert len(entries) == 3 and questions == [{'question_id': 'remaining_relation'}]
        assert binding == {'m_a': Program('feature', name='c_a'), 'm_b': Program('feature', name='c_b')}
        assert methods.records == old_methods and tolerance == 1e-7
        assert budget.spend(3)
        return {'probe': probe, 'forecast_status': 'predicted_distinction', 'predicted_separation': 1.6}

    def shorten(plan, budget, *, tolerance):
        assert plan['probe'] == probe and tolerance == 1e-7 and budget.spend(1)
        return plan

    monkeypatch.setattr(relation_uncertainty, 'RelationIntervalResearch', Questions)
    monkeypatch.setattr(competing_object_experiments, 'plan_competing_object_probe', response_plan)
    monkeypatch.setattr(competing_object_experiments, 'shorten_discriminating_probe', shorten)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_instrument_plan(output)
    assert result['status'] == 'response_distinguishing_experiment'
    assert result['required_fields'] == result['bound_fields'] == ['m_a', 'm_b']
    assert result['missing_fields'] == [] and result['world_predictors_unchanged']
    assert (result['calibration_compute'], result['planning_compute'], result['compute_cost']) == (7, 6, 13)
    assert result['remaining_compute'] == 11987
    assert result['new_world_fits'] == result['new_observations'] == result['model_calls'] == 0
    assert result['pool_occurrences'] == 4 and result['selected_ref'] == cfg['selected_ref']
    plan = load(output / 'plan.json')
    assert not plan['acquired'] and plan['probe']['sequence'] == [{'c_a': 1.6, 'c_b': 0.}]
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert runner.run_instrument_plan(output)['new_compute_cost'] == 0 and len(calls) == 1


@pytest.mark.parametrize('has_probe', [False, True])
def test_missing_instrument_binding_does_not_claim_complete_response_disagreement(tmp_path, monkeypatch, has_probe):
    from hegel_machine.cognition import competing_object_experiments, instrument_hypotheses

    request, source, partial, old_models, old_methods = paid_scene(tmp_path, monkeypatch)
    output = tmp_path / 'partial_plan'
    runner.register_instrument_plan(request, output)
    forbid_science_restart(monkeypatch)
    calls = []
    checked_fit(monkeypatch, partial, old_models, old_methods, calls, missing=True)
    monkeypatch.setattr(competing_object_experiments, 'plan_competing_object_probe',
        lambda *args, **kwargs: pytest.fail('Missing sensors cannot forecast the complete response'))

    def instrument_plan(model, command, measured, budget, *, max_length, serial, tolerance, methods):
        assert set(model['fields']) == {'m_a'}
        assert measured.context['object_model'] == partial and not measured.context.get('object_schema')
        assert max_length == 5 and methods.records == old_methods and budget.spend(5)
        return {'probe': Probe('instrument_boundary', ({'c_a': 0., 'c_b': 1.},)) if has_probe else None,
            'reason': 'predicted_field_disagreement' if has_probe else 'no_distinguishable_hypotheses'}

    monkeypatch.setattr(instrument_hypotheses, 'plan_instrument_disagreement', instrument_plan)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_instrument_plan(output)
    assert result['status'] == ('instrument_distinguishing_experiment' if has_probe else 'instrument_mapping_unresolved')
    assert result['planning_kind'] == 'instrument' and result['missing_fields'] == ['m_b']
    assert result['compute_cost'] == 12 and result['remaining_compute'] == 11988
    assert result['new_world_fits'] == result['new_observations'] == 0
    assert not load(output / 'plan.json')['acquired']


def test_instrument_inference_failure_keeps_paid_models_costs_and_original_world(tmp_path, monkeypatch):
    request, source, partial, old_models, old_methods = paid_scene(tmp_path, monkeypatch)
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    output = tmp_path / 'failed_plan'
    runner.register_instrument_plan(request, output)
    forbid_science_restart(monkeypatch)
    calls = []
    checked_fit(monkeypatch, partial, old_models, old_methods, calls, fail=True)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    with pytest.raises(RuntimeError, match='inference interrupted'):
        runner.run_instrument_plan(output)
    failure = load(output / 'failure.json')
    assert failure['compute_cost'] == 7 and failure['remaining_compute'] == 11993
    assert failure['ledger_ref'] == load(request)['ledger_ref']
    assert load(output / 'instrument_models.json') == old_models
    assert not (output / 'receipt.json').exists()
    assert all(path.read_bytes() == value for path, value in protected.items())
    with pytest.raises(ValueError, match='once'):
        runner.run_instrument_plan(output)
    assert len(calls) == 1


def test_planning_does_not_spend_reserved_return_or_confirmation_balance(tmp_path, monkeypatch):
    request, _, _, _, _ = paid_scene(tmp_path, monkeypatch, balance=3096)
    output = tmp_path / 'unfunded'
    with pytest.raises(ValueError, match='reserves'):
        runner.register_instrument_plan(request, output)
    assert not output.exists()


def test_paid_control_measurement_alignment_must_cover_each_complete_probe(tmp_path, monkeypatch):
    request, source, _, _, _ = paid_scene(tmp_path, monkeypatch)
    location = source / 'paid_pool/probe2.json'
    batch = load(location)
    batch['probe']['sequence'].pop()
    save(location, batch)
    output = tmp_path / 'bad_pairs'
    with pytest.raises(ValueError, match='commands and measurements'):
        runner.register_instrument_plan(request, output)
    assert not output.exists()


def test_nested_instrument_planning_failure_charges_completed_calibration_and_plan(tmp_path, monkeypatch):
    from hegel_machine.cognition import instrument_hypotheses

    request, source, partial, old_models, old_methods = paid_scene(tmp_path, monkeypatch)
    output = tmp_path / 'failed_nested_plan'
    runner.register_instrument_plan(request, output)
    forbid_science_restart(monkeypatch)
    calls = []
    checked_fit(monkeypatch, partial, old_models, old_methods, calls, missing=True)

    def fail_after_planning(model, command, measured, budget, **kwargs):
        assert budget.spend(5)
        raise RuntimeError('planning interrupted after paid forecasts')

    monkeypatch.setattr(instrument_hypotheses, 'plan_instrument_disagreement', fail_after_planning)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    with pytest.raises(RuntimeError, match='paid forecasts'):
        runner.run_instrument_plan(output)
    failure = load(output / 'failure.json')
    assert failure['compute_cost'] == 12 and failure['remaining_compute'] == 11988
    saved = load(output / 'instrument_models.json')
    assert saved[:-1] == old_models and saved[-1]['model_id'] == 'restored_instrument'
    assert not (output / 'receipt.json').exists()


def frozen_instrument_probe_scene(tmp_path, monkeypatch):
    request, source, partial, _, _ = paid_scene(tmp_path, monkeypatch)
    save(tmp_path / 'task.json', {'seed': 73})
    planned = tmp_path / 'frozen_instrument_plan'
    cfg = runner.register_instrument_plan(request, planned)
    cfg['prior_method_review'] = str(tmp_path / 'retained_method_review')
    save(planned / 'registration.json', cfg)
    a = Program('feature', name='c_a')
    b = Program('feature', name='c_b')
    alternative = Program('if', children=(
        Program('lt', children=(b, Program('const', value=1.5))), b,
        Program('add', children=(b, Program('const', value=1.)))))
    model = {'model_id': 'retained_ambiguous_calibration', 'source_schema': None,
        'status': 'ambiguous_on_measured_trajectory',
        'fields': {'m_a': {'program': a.to_dict()},
                   'm_b': {'program': b.to_dict(), 'hypotheses': [
                       {'hypothesis_id': 'b_current', 'program': b.to_dict(), 'status': 'supported'},
                       {'hypothesis_id': 'b_alternative', 'program': alternative.to_dict(), 'status': 'supported'}]}}}
    probe = Probe('frozen_boundary', tuple({'c_a': .5, 'c_b': value} for value in (1.5, 1.4, 1.6)))
    plan = {'probe': asdict(probe), 'predictions': {'m_b': {'b_current': [1.5, 1.4, 1.6],
        'b_alternative': [2.5, 1.4, 2.6]}}, 'field_names': ['m_b'], 'planning_kind': 'instrument',
        'reason': 'learned_instrument_experiment_method', 'sequence_mode': 'learned_boundary',
        'method_id': 'old_boundary_method', 'compute_cost': 5, 'acquired': False}
    save(planned / 'plan.json', plan)
    save(planned / 'instrument_models.json', [model])
    save(planned / 'program_experiment_methods.json', [{'method_id': 'old_boundary_method',
        'source_task_id': 'old_apparatus', 'solver_strategy': 'instrument_sequence',
        'status': 'supported', 'support': []}])
    save(planned / 'receipt.json', {'status': 'instrument_distinguishing_experiment',
        'planning_kind': 'instrument', 'remaining_compute': 11988, 'compute_cost': 12,
        'plan_ref': runner.ref(planned / 'plan.json'),
        'instrument_models_ref': runner.ref(planned / 'instrument_models.json'),
        'frontier_ref': cfg['frontier_ref'], 'selected_ref': cfg['selected_ref'],
        'pool_occurrences': cfg['pool_occurrences'], 'candidate_refs': cfg['candidate_refs'],
        'new_world_fits': 0, 'new_observations': 0, 'model_calls': 0})
    costs = tmp_path / 'after_plan_costs.json'
    save(costs, {'cumulative_protocol_observations': 6})
    limit = tmp_path / 'observation_limit.json'
    save(limit, {'observation_limit': 8000})
    probe_request = tmp_path / 'probe_request.json'
    save(probe_request, {'source': str(planned),
        'ledger_ref': runner.ref(planned / 'receipt.json', 'remaining_compute'),
        'source_costs_ref': runner.ref(costs),
        'observation_limit_ref': runner.ref(limit, 'observation_limit')})
    return probe_request, source, planned, probe, partial


def frozen_probe_laboratory(monkeypatch, expected, calls):
    from hegel_machine.cognition import anonymous_object_lab
    from hegel_machine.cognition.measurement_lab import MeasurementChannel
    from hegel_machine.cognition.unified_object_lab import ObjectMeasurementBatch

    class Lab:
        channels = (MeasurementChannel('fine', 12, 'saved sensors'),)

        def __init__(self, **task):
            assert task == {'seed': 73}
            self.observation_cost = self.measurement_cost = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def acquire(self, probe, channels):
            assert probe == expected and channels == ('fine',)
            calls.append(probe)
            history, rows = [], []
            for point in probe.sequence:
                fields = {'m_a': point['c_a'], 'm_b': point['c_b']}
                row = Observation(fields, point['c_a'], tuple(history))
                rows.append(row)
                history.append({**fields, '__target__': row.target})
            self.observation_cost += len(rows)
            self.measurement_cost += 12 * len(rows)
            return ObjectMeasurementBatch(rows, len(rows), 12 * len(rows), channels, [])

    monkeypatch.setattr(anonymous_object_lab, 'AnonymousObjectLab', Lab)
    return Lab


def forbid_refits_and_replanning(monkeypatch):
    from hegel_machine.cognition import research_programs, transfer_partitions, unified_research, instrument_hypotheses
    monkeypatch.setattr(research_programs._Fitter, 'fit',
        lambda *args, **kwargs: pytest.fail('The delivered response coefficients must not change'))
    monkeypatch.setattr(transfer_partitions, 'continue_transfer_partition',
        lambda *args, **kwargs: pytest.fail('This unit does not repeat completed interval research'))
    monkeypatch.setattr(unified_research.UnifiedResearchAgent, '_fit_instruments',
        lambda *args, **kwargs: pytest.fail('New readings assess saved instrument hypotheses without refitting'))
    monkeypatch.setattr(instrument_hypotheses, 'plan_instrument_disagreement',
        lambda *args, **kwargs: pytest.fail('Execute the frozen instrument probe instead of replanning it'))


def test_frozen_instrument_probe_assesses_only_new_pairs_then_updates_the_full_pool(tmp_path, monkeypatch):
    from scripts import continue_successor_relation_intervals as historical
    from hegel_machine.cognition import competing_object_experiments, instrument_hypotheses, relation_uncertainty

    request, world, planned, probe, partial = frozen_instrument_probe_scene(tmp_path, monkeypatch)
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    acquisitions = []
    frozen_probe_laboratory(monkeypatch, probe, acquisitions)
    output = tmp_path / 'actual_instrument_probe'
    cfg = runner.register_instrument_probe(request, output)
    assert cfg['observation_rows'] == 3 and cfg['previous_observations'] == 6
    assert cfg['update_compute_upper'] == 3 * (4 + 3)
    forbid_refits_and_replanning(monkeypatch)
    instrument_calls, suffix_calls = [], []
    assess = instrument_hypotheses.assess_instrument_hypotheses

    def assess_new_only(model, command, measured, pairs, budget, *, tolerance):
        assert len(pairs) == 3 and [pair[0].features for pair in pairs] == list(probe.sequence)
        assert pairs[0][0].history == pairs[0][1].history == ()
        assert measured.context['object_model'] == partial
        instrument_calls.append(len(pairs))
        return assess(model, command, measured, pairs, budget, tolerance)

    monkeypatch.setattr(instrument_hypotheses, 'assess_instrument_hypotheses', assess_new_only)
    original_loss = historical.loss

    def suffix_only(program, rows, *args):
        assert len(rows) == 3 and [row.features['m_b'] for row in rows] == [1.5, 1.4, 1.6]
        suffix_calls.append(program)
        return original_loss(program, rows, *args)

    monkeypatch.setattr(historical, 'loss', suffix_only)
    originals = load(world / 'new_entries.json')

    class Questions:
        def __init__(self, program, priors, budget, *, research_program):
            assert research_program == originals[0]['research_program']
            assert budget.spend(2)

        def questions(self, entries):
            assert len(entries) == 3
            assert [entry['research_program'] for entry in entries] == [entry['research_program'] for entry in originals]
            return [{'question_id': 'remaining_response_relation'}]

    def response_plan(command, entries, budget, *, binding, questions, **kwargs):
        assert set(binding) == {'m_a', 'm_b'}
        assert binding['m_b'] == Program('feature', name='c_b')
        assert questions == [{'question_id': 'remaining_response_relation'}]
        assert budget.spend(3)
        return {'probe': Probe('next_response_probe', ({'c_a': 1.6, 'c_b': -.5},)),
                'forecast_status': 'predicted_distinction', 'predicted_separation': 2.1}

    monkeypatch.setattr(relation_uncertainty, 'RelationIntervalResearch', Questions)
    monkeypatch.setattr(competing_object_experiments, 'plan_competing_object_probe', response_plan)
    monkeypatch.setattr(competing_object_experiments, 'shorten_discriminating_probe',
        lambda plan, budget, **kwargs: plan)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_instrument_probe(output)
    assert result['status'] == 'response_distinguishing_experiment'
    assert result['bound_fields'] == ['m_a', 'm_b'] and result['missing_fields'] == []
    assert acquisitions == [probe] and instrument_calls == [3] and len(suffix_calls) == 4
    assert result['pool_occurrences'] == 4 and result['comparison_rows'] == 9
    assert result['pool_compute'] == 12 and result['instrument_assessment_compute'] == 9
    assert result['planning_compute'] == 5 and result['compute_cost'] == 26
    assert result['remaining_compute'] == 11988 - 26
    assert result['observation_cost'] == 3 and result['measurement_cost'] == 36
    assert result['cumulative_protocol_observations'] == 9
    assert result['new_world_fits'] == result['new_instrument_fits'] == result['model_calls'] == 0
    assessment = load(output / 'instrument_assessment.json')
    assert assessment['assessment']['supported_fields'] == ['m_a', 'm_b']
    assert assessment['method_feedback']['action'] == 'use_instrument_experiment_method'
    assert load(output / 'program_experiment_methods.json')[0]['support'][-1]['status'] == 'distinguished'
    assert load(output / 'observations.json')['probe']['sequence'] == list(probe.sequence)
    assert not load(output / 'plan.json')['acquired']
    restored = runner.parent_inputs(output)
    assert len(restored[2]) == 9 and len(restored[5]) == 4
    assert restored[1]['comparison_rows'] == 9 and restored[1]['feedback_rows'] == 5
    assert restored[1]['training_rows'] == 4
    assert restored[6]['research_program'] == originals[0]['research_program']
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert runner.run_instrument_probe(output)['new_compute_cost'] == 0
    assert acquisitions == [probe]


def test_probe_retains_current_parent_evidence_reference_when_program_is_unchanged(tmp_path, monkeypatch):
    from hegel_machine.cognition import competing_object_experiments, relation_uncertainty

    request, world, planned, probe, _ = frozen_instrument_probe_scene(tmp_path, monkeypatch)
    frontier = load(world / 'retained_frontier.json')
    old_ref = frontier['selected_ref']
    latest = deepcopy(runner.read_method_reference(old_ref, world))
    latest['research_program']['evaluation_rows'] = 6
    latest_ref = runner.ref(world / 'latest_supported_entry.json')
    save(Path(latest_ref['file']), latest)
    scores = load(world / 'paid_pool/receipt.json')
    shared = next(item for item in scores['scores'] if old_ref in item['candidate_refs'])
    shared['candidate_refs'].append(latest_ref)
    save(world / 'paid_pool/receipt.json', scores)
    save(world / 'retained_frontier.json', {**frontier,
        'selected_ref': latest_ref, 'complete_current_pool_occurrences': 5})
    planned_receipt = load(planned / 'receipt.json')
    save(planned / 'receipt.json', {**planned_receipt, 'selected_ref': latest_ref, 'pool_occurrences': 5})
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    output = tmp_path / 'keep_latest_parent_evidence'
    cfg = runner.register_instrument_probe(request, output)
    assert cfg['selected_ref'] == latest_ref and cfg['update_compute_upper'] == 21
    acquisitions = []
    frozen_probe_laboratory(monkeypatch, probe, acquisitions)
    forbid_refits_and_replanning(monkeypatch)

    class Questions:
        def __init__(self, *args, research_program, **kwargs):
            assert research_program == latest['research_program']

        def questions(self, entries):
            return []

    monkeypatch.setattr(relation_uncertainty, 'RelationIntervalResearch', Questions)
    monkeypatch.setattr(competing_object_experiments, 'plan_competing_object_probe',
        lambda *args, **kwargs: {'probe': None, 'forecast_status': 'no_predicted_distinction',
            'predicted_separation': 0.})
    monkeypatch.setattr(competing_object_experiments, 'shorten_discriminating_probe',
        lambda plan, *args, **kwargs: plan)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_instrument_probe(output)
    assert result['selected_ref'] == latest_ref and result['selected_ref'] != old_ref
    assert result['selected_loss'] == 0. and result['pool_occurrences'] == 5
    assert result['pool_compute'] == 12  # Four unique programs, three new rows.
    current = load(output / 'paid_pool/receipt.json')
    retained = next(item for item in current['scores'] if old_ref in item['candidate_refs'])
    assert retained['candidate_refs'] == [old_ref, latest_ref] and retained['mse'] == 0.
    assert load(output / 'retained_frontier.json')['selected_ref'] == latest_ref
    restored = runner.parent_inputs(output)[6]
    assert restored['research_program']['evaluation_rows'] == 6
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert runner.run_instrument_probe(output)['new_compute_cost'] == 0
    assert acquisitions == [probe]


def test_probe_failure_preserves_purchased_batch_and_paid_instrument_assessment(tmp_path, monkeypatch):
    from scripts import continue_successor_relation_intervals as historical

    request, world, planned, probe, partial = frozen_instrument_probe_scene(tmp_path, monkeypatch)
    acquisitions = []
    frozen_probe_laboratory(monkeypatch, probe, acquisitions)
    output = tmp_path / 'interrupted_probe'
    runner.register_instrument_probe(request, output)
    forbid_refits_and_replanning(monkeypatch)

    def fail_after_new_score(program, rows, *args):
        assert len(rows) == 3
        raise RuntimeError('suffix readout interrupted')

    monkeypatch.setattr(historical, 'loss', fail_after_new_score)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    with pytest.raises(RuntimeError, match='suffix readout interrupted'):
        runner.run_instrument_probe(output)
    failure = load(output / 'failure.json')
    assert failure['compute_cost'] == 12 and failure['remaining_compute'] == 11988 - 12
    assert failure['observation_cost'] == 3 and failure['measurement_cost'] == 36
    assert len(load(output / 'observations.json')['batch']['observations']) == 3
    assert load(output / 'instrument_assessment.json')['assessment']['supported_fields'] == ['m_a', 'm_b']
    hypotheses = load(output / 'instrument_models.json')[-1]['fields']['m_b']['hypotheses']
    assert [item['status'] for item in hypotheses] == ['supported', 'refuted']
    assert not (output / 'receipt.json').exists()
    with pytest.raises(ValueError, match='once'):
        runner.run_instrument_probe(output)
    assert acquisitions == [probe]


def test_frozen_probe_respects_remaining_observation_and_update_reserves(tmp_path, monkeypatch):
    request, _, planned, _, _ = frozen_instrument_probe_scene(tmp_path, monkeypatch)
    supplied = load(request)
    save(Path(supplied['observation_limit_ref']['file']), {'observation_limit': 8})
    output = tmp_path / 'unfunded_probe'
    with pytest.raises(ValueError, match='allowances'):
        runner.register_instrument_probe(request, output)
    assert not output.exists()
    save(Path(supplied['observation_limit_ref']['file']), {'observation_limit': 8000})
    receipt = load(planned / 'receipt.json')
    save(planned / 'receipt.json', {**receipt, 'remaining_compute': 3096 + 21})
    with pytest.raises(ValueError, match='allowances'):
        runner.register_instrument_probe(request, output)
    assert not output.exists()


def test_declared_small_reserve_still_requires_complete_paid_pool_suffix_before_purchase(tmp_path, monkeypatch):
    from hegel_machine.cognition import anonymous_object_lab

    request, _, planned, _, _ = frozen_instrument_probe_scene(tmp_path, monkeypatch)
    supplied = load(request)
    save(request, {**supplied, 'reserved_compute': 273})
    receipt = load(planned / 'receipt.json')
    # Three instrument hypotheses cost 9, but every retained program also
    # needs the new three-row suffix: total update 3 * (3 + 4) = 21.
    # This balance could fund the local readout and reserve, not the full pool.
    save(planned / 'receipt.json', {**receipt, 'remaining_compute': 273 + 20})
    monkeypatch.setattr(anonymous_object_lab, 'AnonymousObjectLab',
        lambda **kwargs: pytest.fail('An incomplete update quote cannot begin a purchase'))
    output = tmp_path / 'insufficient_full_pool_update'
    with pytest.raises(ValueError, match='balance=293, update=21, reserve=273'):
        runner.register_instrument_probe(request, output)
    assert not output.exists()
    assert load(planned / 'receipt.json')['remaining_compute'] == 293


def completed_negative_response_scene(tmp_path, monkeypatch):
    from hegel_machine.cognition import competing_object_experiments, relation_uncertainty

    request, world, planned, probe, partial = frozen_instrument_probe_scene(tmp_path, monkeypatch)
    frozen_probe_laboratory(monkeypatch, probe, [])
    source = tmp_path / 'completed_field_probe'
    runner.register_instrument_probe(request, source)
    forbid_refits_and_replanning(monkeypatch)

    class Questions:
        def __init__(self, program, priors, budget, **kwargs):
            assert budget.spend(2)

        def questions(self, entries):
            assert len(entries) == 3
            return [{'question_id': 'paid_relation_predicate'}]

    def negative_plan(command, entries, budget, **kwargs):
        assert kwargs['questions'] and budget.spend(3)
        return {'probe': None, 'forecast_status': 'no_predicted_distinction',
                'questions_attempted': 1, 'directed_candidates': 4,
                'generic_candidates': 0, 'predicted_separation': 0.}

    monkeypatch.setattr(relation_uncertainty, 'RelationIntervalResearch', Questions)
    monkeypatch.setattr(competing_object_experiments, 'plan_competing_object_probe', negative_plan)
    monkeypatch.setattr(competing_object_experiments, 'shorten_discriminating_probe',
        lambda plan, budget, **kwargs: plan)
    monkeypatch.setattr(runner, 'ROOT', source / 'runtime')
    result = runner.run_instrument_probe(source)
    assert result['status'] == 'response_disagreement_unresolved' and not result['missing_fields']
    monkeypatch.setattr(runner, 'ROOT', Path(__file__).resolve().parents[1])
    return source, world, result


def test_complete_program_planning_follows_negative_condition_plan_once_without_reacquisition(tmp_path, monkeypatch):
    from hegel_machine.cognition import competing_object_experiments, anonymous_object_lab

    source, world, actual = completed_negative_response_scene(tmp_path, monkeypatch)
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    output = tmp_path / 'complete_response_plan'
    cfg = runner.register_response_plan(source, output)
    assert cfg['compute_limit'] == min(6000, actual['remaining_compute'] - 3096)
    monkeypatch.setattr(anonymous_object_lab.AnonymousObjectLab, 'acquire',
        lambda *args, **kwargs: pytest.fail('This complementary search only forecasts control consequences'))
    calls = []
    original_entries = load(world / 'new_entries.json')
    probe = Probe('complete_response_distinction', ({'c_a': 1.6, 'c_b': -.5},))

    def complete_plan(command, entries, budget, *, binding, serial, length, tolerance, questions, methods, task_id):
        assert questions == () and serial == 1 and length == 5
        assert set(binding) == {'m_a', 'm_b'}
        assert [entry['research_program'] for entry in entries] == [entry['research_program'] for entry in original_entries]
        assert [entry['candidate'].program.to_dict() for entry in entries] == [entry['candidate']['program'] for entry in original_entries]
        assert set(command.feature_names) == {'c_a', 'c_b'} and tolerance == 1e-7
        assert budget.spend(11)
        calls.append(questions)
        return {'probe': probe, 'forecast_status': 'predicted_distinction',
            'predicted_separation': 2.1, 'generic_candidates': 2, 'directed_candidates': 1}

    def shorten(plan, budget, *, tolerance):
        assert plan['probe'] == probe and budget.spend(3)
        return plan

    monkeypatch.setattr(competing_object_experiments, 'plan_competing_object_probe', complete_plan)
    monkeypatch.setattr(competing_object_experiments, 'shorten_discriminating_probe', shorten)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_response_plan(output)
    assert result['status'] == 'response_distinguishing_experiment'
    assert result['compute_cost'] == 14 and result['remaining_compute'] == actual['remaining_compute'] - 14
    assert result['new_world_fits'] == result['new_observations'] == result['model_calls'] == 0
    assert not load(output / 'plan.json')['acquired']
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert runner.run_response_plan(output)['new_compute_cost'] == 0 and calls == [()]


def test_complete_program_followup_requires_bound_negative_condition_result(tmp_path, monkeypatch):
    source, _, actual = completed_negative_response_scene(tmp_path, monkeypatch)
    output = tmp_path / 'ineligible_complete_plan'
    for replacement in ({'missing_fields': ['m_b']}, {'status': 'response_distinguishing_experiment'}):
        save(source / 'receipt.json', {**actual, **replacement})
        with pytest.raises(ValueError):
            runner.register_response_plan(source, output)
        assert not output.exists()


@pytest.mark.parametrize('recall_succeeds', [False, True])
def test_full_program_synthesis_receives_its_own_budget_after_paid_recall(monkeypatch, recall_succeeds):
    from hegel_machine.cognition import competing_object_experiments, program_disagreement_probes
    from hegel_machine.cognition.models import SearchBudget

    spec = TaskSpec('budget_handoff', 'Compare executable responses', ('x',), (), (),
        feature_bounds={'x': (-2., 2.)}, max_probe_length=5)
    x = Program('feature', name='x')
    programs = [x, Program('mul', children=(x, x))]
    entries = [{'candidate': Candidate(str(index), program, 'retained')}
        for index, program in enumerate(programs)]
    proposal = {'probe': Probe('supplied_distinction', ({'x': 2.},)),
        'predictions': [[2.], [4.]], 'predicted_separation': 4.}
    parent, fresh_calls = SearchBudget(6000), []

    class Methods:
        last_proposal = {}

        def propose(self, command, retained, allowance, **kwargs):
            assert retained == programs and allowance.limit == 750
            assert allowance.spend(747)
            return [proposal] if recall_succeeds else []

    def synthesize(command, retained, allowance, **kwargs):
        # The real missed pass could not even extract its 112-node program.
        # Its allowance must not be the three credits left by failed recall.
        assert not recall_succeeds and retained == programs
        assert parent.spent == 747
        assert allowance.limit == min(6000, parent.remaining // 8) == 656
        assert allowance.spend(112) and allowance.spend(18)
        fresh_calls.append(allowance.spent)
        return [proposal]

    monkeypatch.setattr(program_disagreement_probes, 'propose_program_disagreement_probes', synthesize)
    monkeypatch.setattr(competing_object_experiments, '_probe_pool',
        lambda *args: iter([({'x': 0.},)]))
    result = competing_object_experiments.plan_competing_object_probe(
        spec, entries, parent, methods=Methods(), questions=(), length=1)

    generation_cost = 747 + (0 if recall_succeeds else 130)
    assert result['forecast_status'] == 'predicted_distinction'
    assert result['probe'].sequence == proposal['probe'].sequence
    assert result['proposal_source'] == ('learned_method' if recall_succeeds else 'program_disagreement')
    assert result['method_search_cost'] == 747
    assert result['generation_cost'] == generation_cost
    assert result['compute_cost'] == parent.spent == generation_cost + 2 <= parent.limit
    assert result['fresh_synthesis_skipped'] is recall_succeeds
    assert fresh_calls == ([] if recall_succeeds else [130])


def test_directed_response_continuation_uses_latest_balance_and_paid_pool_without_replaying_prior_search(tmp_path, monkeypatch):
    from hegel_machine.cognition import competing_object_experiments, program_experiment_methods

    source, world, first = completed_negative_response_scene(tmp_path, monkeypatch)
    previous = tmp_path / 'negative_complete_plan'
    old_cfg = runner.register_response_plan(source, previous)

    def completed_generic_search(command, entries, budget, **kwargs):
        assert budget.spend(13)
        return {'probe': None, 'forecast_status': 'no_predicted_distinction',
            'predicted_separation': 0., 'generic_candidates': 17, 'directed_candidates': 0,
            'method_candidates': 0, 'generation_cost': 7, 'method_search_cost': 7,
            'hypotheses': [{'program': entry['candidate'].program.to_dict()}
                for entry in entries]}

    monkeypatch.setattr(competing_object_experiments, 'plan_competing_object_probe', completed_generic_search)
    monkeypatch.setattr(runner, 'ROOT', previous / 'runtime')
    previous_result = runner.run_response_plan(previous)
    monkeypatch.setattr(runner, 'ROOT', Path(__file__).resolve().parents[1])
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    output = tmp_path / 'paid_directed_followup'
    cfg = runner.register_response_plan(previous, output)
    assert cfg['source'] == str(source)
    assert cfg['planning_kind'] == 'paid_directed_response'
    assert cfg['available_compute'] == previous_result['remaining_compute'] == first['remaining_compute'] - 13
    assert cfg['ledger_ref'] == runner.ref(previous / 'receipt.json', 'remaining_compute')
    assert cfg['compute_limit'] == cfg['available_compute'] - 3096
    for key in ('candidate_refs', 'binding_ref', 'frontier_ref', 'selected_ref', 'serial', 'length'):
        assert cfg[key] == old_cfg[key]

    forbid_science_restart(monkeypatch)
    monkeypatch.setattr(runner, 'score_paid_rows',
        lambda *args, **kwargs: pytest.fail('All current pool scores are already paid'))
    monkeypatch.setattr(competing_object_experiments, 'plan_competing_object_probe',
        lambda *args, **kwargs: pytest.fail('Do not restart old recall or generic response probes'))
    monkeypatch.setattr(program_experiment_methods.ProgramExperimentMethods, 'propose',
        lambda *args, **kwargs: pytest.fail('The current failed recall was already paid'))
    calls = []
    originals = load(world / 'new_entries.json')
    probe = Probe('new_directed_response', ({'c_a': 1.6, 'c_b': -.5},))

    def review(command, hypotheses, records, budget, *, score_scope_ref, score_rows,
               control_levels, tolerance, serial, length, binding):
        assert score_rows == 9 and serial == 1 and length == 5 and tolerance == 1e-7
        assert control_levels is None and set(binding) == {'m_a', 'm_b'}
        assert sum(len(record['rows']) for record in records) == 9
        assert all(set(row['features']) == {'c_a', 'c_b'}
            for record in records for row in record['rows'])
        assert [item['program'] for item in hypotheses] == [entry['candidate']['program'] for entry in originals]
        assert all(item['score']['mse'] == 0. for item in hypotheses)
        assert all(item['score_ref']['file'] == score_scope_ref['file'] for item in hypotheses)
        assert budget.spend(11)
        calls.append(score_scope_ref)
        return {'status': 'distinguishing_experiment_available', 'compute_cost': 11,
            'new_scores': 0, 'new_fits': 0, 'new_observations': 0,
            'plan': {'probe': probe, 'forecast_status': 'predicted_distinction',
                'predicted_separation': 2.1, 'hypotheses': hypotheses,
                'predicted_fields': {'m_a': [1.6], 'm_b': [-.5]}}}

    monkeypatch.setattr(competing_object_experiments, 'review_paid_program_competition', review)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_response_plan(output)
    assert result['status'] == 'response_distinguishing_experiment'
    assert result['compute_cost'] == 11 and result['remaining_compute'] == cfg['available_compute'] - 11
    assert result['new_world_fits'] == result['new_observations'] == result['model_calls'] == 0
    assert result['pool_occurrences'] == 4 and result['selected_ref'] == old_cfg['selected_ref']
    saved_probe = load(output / 'plan.json')['probe']
    assert saved_probe['probe_id'] == probe.probe_id and saved_probe['sequence'] == list(probe.sequence)
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert runner.run_response_plan(output)['new_compute_cost'] == 0 and len(calls) == 1


def test_bound_response_probe_consumes_frozen_predictions_and_only_new_pool_suffix_once(tmp_path, monkeypatch):
    from dataclasses import replace
    from scripts import continue_successor_relation_intervals as historical
    from hegel_machine.cognition import competing_object_experiments, instrument_hypotheses
    from hegel_machine.cognition.models import SearchBudget

    source, world, paid_result = completed_negative_response_scene(tmp_path, monkeypatch)
    planned = tmp_path / 'bound_response_plan'
    prior = runner.register_response_plan(source, planned)
    frontier, original, rows, _, _, _, _, _ = runner.parent_inputs(source)
    public = TaskSpec(**runner.read_method_reference(original['public_spec_ref'], source))
    command = replace(public, feature_names=tuple(public.feature_bounds), context={})
    binding = {name: Program.from_dict(program) for name, program in load(source / 'instrument_binding.json').items()}
    hypotheses = [{'candidate_id': entry['candidate']['candidate_id'],
        'program': entry['candidate']['program'],
        'score': {'mse': 0., 'returncode': 0, 'rows': len(rows)},
        'score_ref': runner.ref(source / 'paid_pool/receipt.json', 'scores', index + 1)}
        for index, entry in enumerate(load(world / 'new_entries.json'))]
    records = [{'rows': [{'features': point} for point in
        runner.read_method_reference(location, source)['probe']['sequence']]}
        for location in [*original['data_refs'], *original.get('appended_observation_refs', [])]]
    reviewed = competing_object_experiments.review_paid_program_competition(
        command, hypotheses, records, SearchBudget(6000), score_scope_ref=frontier['original_pool_ref'],
        score_rows=len(rows), binding=binding, control_levels=None, serial=1, length=5)
    assert reviewed['status'] == 'distinguishing_experiment_available'
    plan = reviewed['plan']
    probe = plan['probe']
    save(planned / 'plan.json', {**plan, 'probe': asdict(probe), 'acquired': False})
    available = paid_result['remaining_compute'] - reviewed['compute_cost']
    save(planned / 'receipt.json', {'status': 'response_distinguishing_experiment',
        'selected_ref': frontier['selected_ref'], 'remaining_compute': available,
        'compute_cost': reviewed['compute_cost']})
    save(tmp_path / 'after_response_costs.json', {'cumulative_protocol_observations': len(rows)})
    request = tmp_path / 'response_request.json'
    save(request, {'source': str(planned),
        'ledger_ref': runner.ref(planned / 'receipt.json', 'remaining_compute'),
        'source_costs_ref': runner.ref(tmp_path / 'after_response_costs.json'),
        'observation_limit_ref': runner.ref(tmp_path / 'observation_limit.json', 'observation_limit')})
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    output = tmp_path / 'actual_response_probe'
    cfg = runner.register_instrument_probe(request, output)
    assert cfg['planning_kind'] == 'response' and cfg['world_source'] == str(source)
    assert cfg['candidate_refs'] == prior['candidate_refs'] and cfg['binding_ref'] == prior['binding_ref']
    acquisitions, suffix_calls = [], []
    frozen_probe_laboratory(monkeypatch, probe, acquisitions)
    forbid_refits_and_replanning(monkeypatch)
    monkeypatch.setattr(competing_object_experiments, 'plan_competing_object_probe',
        lambda *args, **kwargs: pytest.fail('Stop after this frozen response experiment'))
    monkeypatch.setattr(instrument_hypotheses, 'assess_instrument_hypotheses',
        lambda *args, **kwargs: pytest.fail('Valid forecast instruments do not need another calibration assessment'))
    loss = historical.loss

    def new_suffix_only(program, observations, *args):
        assert len(observations) == probe.cost and observations[0].history == ()
        suffix_calls.append(program)
        return loss(program, observations, *args)

    monkeypatch.setattr(historical, 'loss', new_suffix_only)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_instrument_probe(output)
    assessment = load(output / 'response_assessment.json')
    assert result['status'] == 'response_distinguished'
    assert assessment['assessment']['calibration_status'] == 'supported_on_probe'
    assert {item['status'] for item in assessment['assessment']['hypotheses']} == {'supported_on_probe', 'counterexample'}
    assert assessment['method_feedback']['action'] == 'learn_program_experiment_method'
    assert len(load(output / 'program_experiment_methods.json')) == 2
    assert acquisitions == [probe] and len(suffix_calls) == 4
    assert result['pool_occurrences'] == 4 and result['comparison_rows'] == 9 + probe.cost
    assert result['selected_loss'] == 0. and result['selected_ref'] != frontier['selected_ref']
    assert result['pool_compute'] == 4 * probe.cost
    assert result['instrument_assessment_compute'] == 0
    assert result['response_assessment_compute'] == (3 + 2) * probe.cost
    assert result['planning_compute'] == 0 and result['compute_cost'] == 9 * probe.cost
    assert result['remaining_compute'] == available - result['compute_cost']
    assert result['new_world_fits'] == result['new_instrument_fits'] == result['model_calls'] == 0
    assert load(output / 'plan.json')['probe'] is None
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert runner.run_instrument_probe(output)['new_compute_cost'] == 0 and acquisitions == [probe]
