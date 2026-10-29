"""A successor shares one real prefix, preserves its pool, and freezes before readout."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_learned_structure_successor as study
from hegel_machine.cognition.experiment_runtime import ExecutionTiming
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec


class PublicLab:
    channels = (SimpleNamespace(name='fine', cost_per_row=2),)
    purchases = []

    def __init__(self, **settings):
        self.observation_cost = self.measurement_cost = 0
        self.settings = settings

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def costs(self):
        return {'observation_cost': self.observation_cost, 'measurement_cost': self.measurement_cost}

    def spec_for(self, channels=()):
        return TaskSpec('public', 'Test public bank', ('x',), (), (),
            feature_bounds={'x': (-1., 1.)})

    def acquire(self, probe, channels):
        self.purchases.append(probe.probe_id)
        self.observation_cost += probe.cost
        self.measurement_cost += probe.cost * 2
        rows = [Observation(dict(point), 1., tuple(probe.sequence[:i]))
                for i, point in enumerate(probe.sequence)]
        from hegel_machine.cognition.unified_object_lab import ObjectMeasurementBatch
        return ObjectMeasurementBatch(rows, probe.cost, probe.cost * 2, channels,
            [{'before': dict(point), 'after': dict(point)} for point in probe.sequence])

    def random_probe(self, seed, *, length, name):
        return Probe(name, tuple({'x': i / max(1, length)} for i in range(length)))


def registration(tmp_path):
    return {'task': {'seed': study.SEED}, 'research_config': asdict(ResearchConfig()),
        'source_task_ref': study.ref(tmp_path / 'registration.json', 'task'),
        'historical_knowledge_ref': {'store': 'read-only-history', 'sequence': 1},
        'knowledge_contract_ref': {'file': 'paid-knowledge-delta.json'},
        'knowledge_root': '/paid/k1',
        'source_cost_refs': [{'file': 'paid-source-costs.json'}],
        'raw_source_ref': {'file': 'paid-raw-source.json'},
        'role_method_ref': {'file': str(tmp_path / 'method.json')},
        'observation_limit': 100, 'measurement_limit': 200,
        'confirmation_rows': 4, 'confirmation_seed': 100,
        'preparation_active_wall_seconds': 1800, 'quality_threshold': 1e-7,
        'operations': {'baseline': 'program_replay', 'challenger': 'structural_research'}}


def test_acquisition_reserves_confirmation_and_keeps_one_copy(tmp_path):
    PublicLab.purchases = []
    lab = PublicLab()
    probe = Probe('first', tuple({'x': .5} for _ in range(6)))
    agent = SimpleNamespace(_spec=lambda lab, channels: lab.spec_for(channels),
        _probe=lambda *args: probe)
    config = {**registration(tmp_path), 'observation_limit': 12, 'measurement_limit': 24}
    batches, actions = [], []
    acquire = study.acquire_common(agent, lab, tmp_path, config, SearchBudget(10), actions, batches)
    assert len(acquire(('fine',), 'observe_anonymous_dependencies', rows_count=6)) == 6
    assert acquire(('fine',), 'compare_on_new_trajectory', rows_count=3) is None
    assert PublicLab.purchases == ['first']
    assert len(batches) == len(list((tmp_path / 'observations').glob('*.json'))) == 1
    receipt = study.load(tmp_path / 'acquisition_receipt.json')
    assert receipt['costs'] == {'observation_cost': 6, 'measurement_cost': 12}
    assert receipt['batches'] == [study.ref(tmp_path / 'observations/000.json')]
    assert 'observations' not in receipt


def test_common_prefix_restores_complete_pool_and_none_parent(tmp_path, monkeypatch):
    PublicLab.purchases = []
    study.save(tmp_path / 'method.json', {'model_id': 'old', 'source_ref': {'file': 'old.json'}})
    study.save(tmp_path / 'registration.json', registration(tmp_path))
    monkeypatch.setattr(study, 'AnonymousObjectLab', PublicLab)
    monkeypatch.setattr(study, 'load_retained_knowledge', lambda path: KnowledgeLibrary())

    def research_roles(agent, laboratory, channels, acquire, batches, budget, actions, **kwargs):
        assert 'object_schema' not in laboratory.spec_for(channels).context
        acquire(channels, 'observe_anonymous_dependencies', rows_count=6)
        budget.spend(3)
        model = {'groups': [{'q': 'x'}], 'schema': {'objects': [], 'fields': {}},
            'model_id': 'measured', 'complete': True}
        agent._object_models[channels] = dict(model)
        agent._object_schemas[channels] = model['schema']
        model['membership_witnesses'] = [{'x': 1.}]
        agent.library.object_models.append(model)
        return model

    def prepare(agent, spec, train, validation, budget, actions, **kwargs):
        assert kwargs['parent'] is None and len(train) == 6 and len(validation) == 24
        assert len(agent.library.object_models) == 1
        budget.spend(5)
        entries = [{'candidate': Candidate('old', Program('const', value=1.), 'retained'),
                    'training_loss': 0., 'validation_loss': 0., 'kind': 'retained'}]
        return dict(spec=spec, training=train, validation=validation, comparison=validation,
            parent=None, transitions=kwargs['transitions'], assessed=entries, uncompared=[],
            fallback=1., prepare_compute_cost=5, remaining_compute=budget.remaining,
            comparison_complete=True, terminal=True, independent_validation=True,
            observation_remaining=kwargs['observation_remaining'],
            measurement_remaining=kwargs['measurement_remaining'])

    monkeypatch.setattr(study.CommonObjectResearchAgent, '_research_objects', research_roles)
    monkeypatch.setattr(study.CommonObjectResearchAgent, 'prepare_research_decision', prepare)
    budget = SearchBudget(100)
    timing = ExecutionTiming(('preparation',))
    agent, frame, scene = study.prepare_successor(tmp_path, registration(tmp_path),
        KnowledgeLibrary(), budget, timing)
    assert frame['terminal'] and frame['parent'] is None
    assert frame['assessed'][0]['candidate'].candidate_id == 'old'
    assert len(agent.library.object_models) == 1 and scene['knowledge']['object_models'] == []
    assert budget.spent == 8
    saved = study.load(tmp_path / 'prepared_state.json')
    assert 'knowledge' not in saved and 'training' not in saved
    assert saved['parent_ref'] is None
    assert saved['knowledge_base_ref'] == registration(tmp_path)['knowledge_contract_ref']
    assert len(study.load(tmp_path / 'common_receipt.json')['batches']) == 2
    assert len(list((tmp_path / 'observations').glob('*.json'))) == 2
    assert not (tmp_path / 'common_knowledge_delta.json').exists()
    assert 'membership_witnesses' not in scene['spec'].context['object_model']
    assert scene['object_model']['membership_witnesses'] == [{'x': 1.}]
    before = list(PublicLab.purchases)
    recovered = study.load_successor_scene(tmp_path)
    again, restored, _ = study.load_prepared(tmp_path, scene=recovered)
    assert PublicLab.purchases == before
    assert restored['training'] == frame['training'] and restored['assessed'] == frame['assessed']
    assert again.library.to_dict() == agent.library.to_dict()


def frozen_outcomes(tmp_path, *, complete=True):
    study.save(tmp_path / 'prepared_state.json', {'fallback': 0.})
    for index, arm in enumerate(('baseline', 'challenger')):
        candidate = {'candidate': {'program': Program('const', value=float(index)).to_dict()}}
        study.save(tmp_path / arm / 'claim.json', candidate)
        study.save(tmp_path / arm / 'outcome.json', {'execution_complete': complete,
            'full_comparison': True, 'selected_ref': study.ref(tmp_path / arm / 'claim.json')})


def test_confirmation_freezes_both_before_one_shared_purchase(tmp_path, monkeypatch):
    frozen_outcomes(tmp_path)
    PublicLab.purchases = []

    class ConfirmLab(PublicLab):
        def acquire(self, probe, channels):
            frozen = study.load(tmp_path / 'confirmation/frozen_predictors.json')
            assert set(frozen) == {'baseline', 'challenger'}
            return super().acquire(probe, channels)

    monkeypatch.setattr(study, 'AnonymousObjectLab', ConfirmLab)
    result = study.confirm_pair(tmp_path, registration(tmp_path))
    assert result['arms']['baseline']['mse'] == 1.
    assert result['arms']['challenger']['mse'] == 0.
    assert result['shared_actual_costs'] == {'compute_cost': 8, 'observation_cost': 4, 'measurement_cost': 8}
    assert PublicLab.purchases == ['shared_frozen_confirmation']
    assert not result['used_for_selection_or_condition_learning']
    with pytest.raises(FileExistsError):
        study.confirm_pair(tmp_path, registration(tmp_path))


def test_incomplete_pair_does_not_buy_confirmation(tmp_path, monkeypatch):
    frozen_outcomes(tmp_path, complete=False)

    def no_lab(**kwargs):
        raise AssertionError('No fresh observations for an incomplete pair')

    monkeypatch.setattr(study, 'AnonymousObjectLab', no_lab)
    result = study.confirm_pair(tmp_path, registration(tmp_path))
    assert result['status'] == 'not_run_incomplete_pair' and result['new_observations'] == 0


def test_solved_common_pool_stops_without_dispatch_or_restart(tmp_path, monkeypatch):
    config = {**registration(tmp_path), 'knowledge_root': '/paid/k1', 'preparation_compute_limit': 100}
    study.save(tmp_path / 'registration.json', config)
    monkeypatch.setattr(study, 'ROOT', tmp_path / 'runtime')
    monkeypatch.setattr(study, 'load_retained_knowledge', lambda path: KnowledgeLibrary())
    monkeypatch.setattr(study, 'prepare_successor', lambda *args:
        (None, {'comparison_complete': True, 'terminal': True}, None))

    def forbidden(*args, **kwargs):
        raise AssertionError('An already solved common pool must stop')

    monkeypatch.setattr(study, 'finish_prepared_pair', forbidden)
    monkeypatch.setattr(study, 'confirm_pair', forbidden)
    assert study.run(tmp_path)['status'] == 'common_pool_already_explained'
    with pytest.raises(ValueError, match='no implicit restart'):
        study.run(tmp_path)
