"""The saved distinguishing plan consumes the original balance and confirmation gate."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import continue_successor_relation_intervals as study
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, TaskSpec
from hegel_machine.cognition.unified_object_lab import ObjectMeasurementBatch
from tests.test_successor_interval_continuation import entry, selector


def test_resume_registration_preserves_seed_limits_gates_and_paid_history(tmp_path, monkeypatch):
    source, output = tmp_path / 'paid', tmp_path / 'resume'
    old = {'source': 'older-245-row-source', 'total_compute_limit': 270008,
        'previous_compute': 177215, 'remaining_compute': 92793,
        'previous_observations': 245, 'previous_measurement_cost': 2940,
        'maximum_experiments': 12, 'threshold': 1e-7,
        'confirmation_seed': 107063141, 'confirmation_rows': 48,
        'source_cost_refs': [{'file': 'upstream.json'}]}
    result = {'full_comparison': True, 'confirmation': None, 'actual_experiments': 11,
        'cumulative_compute': 266778, 'observation_cost': 39, 'measurement_cost': 468}
    study.save(source / 'registration.json', old)
    study.save(source / 'receipt.json', result)
    monkeypatch.setattr(study, 'freeze_runtime', lambda *args: None)
    monkeypatch.setattr(study, 'write_text_incrementally', lambda *args, **kwargs: None)

    config = study.register_resume(source, output)

    assert config['remaining_compute'] == 3230
    assert config['previous_compute'] + config['remaining_compute'] == old['total_compute_limit']
    assert config['confirmation_seed'] == old['confirmation_seed']
    assert config['threshold'] == 1e-7 and config['relation_precision'] == 1e-6
    assert config['confirmation_rows'] == 48 and config['maximum_experiments'] == 12
    assert config['previous_experiments'] == 11
    assert config['previous_observations'] == 284 and config['previous_measurement_cost'] == 3408
    assert study.load(source / 'registration.json') == old
    assert study.load(source / 'receipt.json') == result
    # A resumed segment may contain one local experiment while consuming the
    # twelfth experiment of the original contract.
    study.save(source / 'receipt.json', {**result, 'actual_experiments': 1,
        'cumulative_experiments': 12})
    blocked_output = tmp_path / 'beyond-original-count'
    with pytest.raises(ValueError, match='experiment count has been consumed'):
        study.register_resume(source, blocked_output)
    assert not blocked_output.exists()


def resume_scene(tmp_path, monkeypatch, *, remaining=3230, width=5e-7, fresh_target=0.):
    output = tmp_path / 'resume'
    cfg = {'resume_result': str(tmp_path / 'paid'), 'remaining_compute': remaining,
        'previous_compute': 270008-remaining, 'total_compute_limit': 270008,
        'confirmation_rows': 48, 'threshold': 1e-7, 'relation_precision': 1e-6,
        'active_wall_seconds': 30, 'maximum_experiments': 12, 'previous_experiments': 11,
        'previous_observations': 284, 'observation_limit': 512,
        'previous_measurement_cost': 3408, 'measurement_limit': 6144,
        'confirmation_seed': 107063141, 'source_cost_refs': [{'file': 'paid-cost.json'}]}
    study.save(output / 'registration.json', cfg)
    paid = [Observation({'x': 0.}, 0.) for _ in range(4)]
    acquired = [Observation({'x': 1.}, 0.) for _ in range(3)]
    parent = entry(Program('feature', name='x'), 'parent', parameter_index=0,
                   parameter_values={'r': .5}, research_program={'selection_steps': []})
    duplicate = {**deepcopy(parent), 'candidate': entry(parent['candidate'].program, 'same-law')['candidate']}
    bad = entry(Program('const', value=2.), 'older-strong-pool-member')
    candidate = entry(Program('const', value=0.), 'new', parameter_index=0,
                      parameter_values={'r': .5000001}, research_program={'selection_steps': []})
    pool = [parent, duplicate, bad]
    refs = [study.ref(tmp_path / f'old-{i}.json') for i in range(len(pool))]
    old_knowledge_ref = study.ref(tmp_path / 'paid' / 'observations' / 'old.json')
    plan = {'probe': Probe('saved-plan-13', ({'x': 1.},)*3),
        'forecast_status': 'predicted_distinction', 'hypotheses': [{}, {}],
        'predicted_fields': {'x': [1., 1., 1.]}, 'proposal_source': 'learned_method'}
    spec = TaskSpec('public', 'Controls', ('x',), (), (), feature_bounds={'x': (-2., 2.)})
    agent = selector()
    agent.config.seed = 104
    feedback = []
    methods = SimpleNamespace(
        observe=lambda actual, evidence, **kwargs: feedback.append((actual, evidence)) or {'kept': True},
        to_dict=lambda: [{'method_id': 'paid-method', 'status': 'supported'}])
    agent.library = SimpleNamespace(meta_priors=[], instrument_models=[{'saved': True}],
                                    program_experiment_methods=methods)
    agent._instrument_binding = lambda *args: {}
    scene = {'agent': agent, 'spec': spec, 'paid': paid, 'pool': pool, 'refs': refs,
        'scores': {parent['candidate'].program: 0., bad['candidate'].program: 4.},
        'parent': parent, 'paid_errors': [0.]*len(paid), 'source_scene': {'parent': bad},
        'pending_plan': plan, 'pending_plan_ref': study.ref(tmp_path/'paid'/'plan_13.json'),
        'experiments': [tuple(paid[-2:])], 'calibration': {'saved': True},
        'original_registration': {'task': {'seed': 104}},
        'knowledge_source': str(tmp_path/'original-245'),
        'knowledge_observation_refs': [old_knowledge_ref],
        'source_refs': {'selection': study.ref(tmp_path/'paid'/'selection.json')}}
    monkeypatch.setattr(study, 'ROOT', output/'runtime')
    monkeypatch.setattr(study, 'restore_interval_result', lambda source: scene)
    monkeypatch.setattr(study, 'rebind_instrument_model', lambda *args: {})
    calls = {'purchases': [], 'loss': [], 'updates': [], 'knowledge': [], 'readouts': []}

    class Research:
        def __init__(self, program, priors, budget, **kwargs):
            assert program == parent['candidate'].program and budget.spend(2)
            self.status = 'unresolved'
            self.reuse_statistics = {'paid_partition_rows': len(paid)}
            self.intervals = [{'radius_interval': [.5, .5+2e-6]}]

        def reuse_paid_partition(self, actual, selected):
            assert actual == paid and selected is parent
            return True

        def alternatives(self, actual, budget, tolerance=1e-7, *, experiments):
            assert actual == paid+acquired and experiments == [tuple(paid[-2:]), tuple(acquired)]
            assert tolerance == 1e-7 and budget.spend(7)
            calls['updates'].append(actual)
            self.intervals = [{'radius_interval': [.5, .5+width]}]
            return [candidate]

        def adopt(self, selected):
            assert selected['candidate'] == candidate['candidate']

    monkeypatch.setattr(study, 'RelationIntervalResearch', Research)
    def prohibited(*args, **kwargs):
        pytest.fail('Resume must not refit, regenerate the paid plan or re-score old rows')
    monkeypatch.setattr(study, 'score_retained', prohibited)
    monkeypatch.setattr(study, 'plan_competing_object_probe', prohibited)
    monkeypatch.setattr('hegel_machine.cognition.research_programs._Fitter.fit', prohibited)

    def new_rows_only(program, actual, fallback, quantile):
        assert actual is acquired
        calls['loss'].append(program)
        return sum((program.predict(row)-row.target)**2 for row in actual)/len(actual)
    monkeypatch.setattr(study, 'loss', new_rows_only)

    def assess(actual, pairs, budget, **kwargs):
        assert actual is plan and [measured for _, measured in pairs] == acquired
        assert budget.spend(3)
        return {'status': 'distinguished', 'calibration_status': 'supported_on_probe'}
    monkeypatch.setattr(study, 'assess_competing_object_probe', assess)

    class PublicLab:
        def __init__(self, **kwargs):
            self.observation_cost = self.measurement_cost = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def spec_for(self, channels): return spec
        def random_probe(self, seed, *, length, name):
            assert seed == cfg['confirmation_seed'] and length == 48
            frozen = study.load(output/'frozen_predictors.json')
            assert Program.from_dict(frozen['selected']) == candidate['candidate'].program
            assert Program.from_dict(frozen['incumbent']) == bad['candidate'].program
            return Probe('fresh-confirmation', ({'x': 2.},)*length)
        def acquire(self, probe, channels):
            assert channels == ('fine',)
            calls['purchases'].append(probe.probe_id)
            actual = acquired if probe.probe_id == 'saved-plan-13' else [
                Observation({'x': 2.}, fresh_target) for _ in range(48)]
            self.observation_cost += len(actual)
            self.measurement_cost += 12*len(actual)
            return ObjectMeasurementBatch(actual, len(actual), 12*len(actual), channels, [])
    monkeypatch.setattr(study, 'AnonymousObjectLab', PublicLab)
    def predict(program, actual):
        calls['readouts'].append((program, len(actual)))
        return [program.predict(row) for row in actual]
    monkeypatch.setattr(study, 'predict_rows', predict)
    def retain(agent, selected, actual_spec, actual_rows, errors, confirmation, destination, **kwargs):
        assert actual_rows == paid+acquired and len(errors) == 7
        assert kwargs['source'] == scene['knowledge_source']
        assert kwargs['new_observation_refs'] == [old_knowledge_ref, study.ref(output/'observations'/'00.json')]
        calls['knowledge'].append(confirmation)
        return {'stored': True}
    monkeypatch.setattr(study, 'save_confirmed_knowledge', retain)
    return output, cfg, scene, calls


def test_saved_plan_executes_once_old_scores_extend_then_original_confirmation_runs(tmp_path, monkeypatch):
    output, cfg, scene, calls = resume_scene(tmp_path, monkeypatch)
    result = study.resume(output)
    assert calls['purchases'] == ['saved-plan-13', 'fresh-confirmation']
    assert calls['loss'] == list(scene['scores']) and len(calls['updates']) == 1
    assert result['retained_pool'] == 3 and result['complete_pool'] == 4
    assert result['cumulative_experiments'] == 12
    assert result['confirmation_admitted'] and result['status'] == 'confirmed'
    assert result['confirmation']['selected']['passed'] and not result['confirmation']['incumbent']['passed']
    assert result['cumulative_compute'] <= 270008
    saved = study.load(output/'selection.json')
    assert [record['candidate_ref'] for record in saved['scores'][:3]] == scene['refs']
    assert saved['new_observation_refs'] == [*scene['knowledge_observation_refs'],
        study.ref(output/'observations'/'00.json')]
    assert len(calls['knowledge']) == 1
    assert [count for _, count in calls['readouts']] == [7, 48, 48]


@pytest.mark.parametrize('remaining,width,purchases', [
    (100, 5e-7, []), (3230, 2e-6, ['saved-plan-13'])])
def test_insufficient_allowance_or_unresolved_precision_never_buys_confirmation(
        tmp_path, monkeypatch, remaining, width, purchases):
    output, cfg, scene, calls = resume_scene(tmp_path, monkeypatch, remaining=remaining, width=width)
    result = study.resume(output)
    assert calls['purchases'] == purchases
    assert result['confirmation'] is None and not result['confirmation_admitted']
    assert result['status'] == 'unresolved' and not calls['knowledge']
    assert result['cumulative_compute'] <= 270008
    if not purchases:
        assert not calls['updates'] and not calls['loss'] and not calls['readouts']
        assert result['complete_pool'] == len(scene['pool'])


def test_failed_fresh_confirmation_keeps_predictor_and_feedback_without_publishing(tmp_path, monkeypatch):
    output, cfg, scene, calls = resume_scene(tmp_path, monkeypatch, fresh_target=1.)
    result = study.resume(output)
    assert calls['purchases'] == ['saved-plan-13', 'fresh-confirmation']
    assert result['confirmation_admitted'] and not result['confirmation']['selected']['passed']
    assert result['status'] == 'unresolved' and not calls['knowledge']
    assert study.load(output/'selection.json')['fresh_confirmation_admitted']
    assert study.load(output/'program_experiment_methods.json') == [
        {'method_id': 'paid-method', 'status': 'supported'}]
