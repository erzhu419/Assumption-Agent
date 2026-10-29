"""The paid preparation survives metadata errors and resumes without refitting."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.action_policy import ResearchActionPolicy
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from scripts.run_native_decision_pair import load_prepared, save_prepared, preparation_matches


def test_preparation_match_keeps_refit_and_retained_same_kind_separate():
    retained = {'kind': 'partial_object:window_3_totals', 'validation_loss': .047}
    refit = {'kind': retained['kind'], 'validation_loss': .041}
    frame = {'assessed': [{'kind': 'current_affine', 'validation_loss': 4.}, refit, retained],
             'retained_on_current': retained}
    scene = {'expected_preparation': {'partial_claims': [dict(refit)],
        'affine_validation_loss': 4., 'retained_response_validation_loss': .047}}
    assert preparation_matches(frame, scene)['matched']
    retained['validation_loss'] = .041
    wrong = preparation_matches(frame, scene)
    assert not wrong['matched'] and set(wrong['changed']) == {'retained_response'}


def prepared_scene():
    x = Program('feature', name='x')
    square = Program('mul', children=(x, x))
    payload = _plan({'fit_terms': (square,), 'coefficients': [0., 1.],
        'program': square, 'loss': 0.}, [], 7, 20)
    template = AbstractTemplate('source:abstract:0', 'source', square,
        ('input',), ('x',), ())
    cfg = ResearchConfig(minimum_improvement=1e-7)
    agent = UnifiedResearchAgent(cfg)
    agent.library.meta_priors.append({'source_concept_id': 'source', 'research_program': payload})
    initial_knowledge = deepcopy(agent.library.to_dict())
    rows = [Observation({'x': value}, .25 + 1.8 * value * value) for value in (-3., -2., -1., 0., 1., 2., 3.)]
    validation = [Observation({'x': value}, .25 + 1.8 * value * value) for value in (-2.5, .5, 2.5)]
    model = {'model_id': 'objects_test', 'schema': {'objects': [], 'fields': {}}}
    spec = TaskSpec('checkpoint', 'Paid measured scene', ('x',), (), (),
        context={'object_model': deepcopy(model)})
    parent = {'candidate': Candidate('kept', x, 'retained_response'),
        'kind': 'retained_response', 'training_loss': 20., 'validation_loss': 20.}
    retained = {**parent, 'validation_loss': 19.}
    affine = {'candidate': Candidate('affine', Program('const', value=4.), 'current_affine'),
        'kind': 'current_affine', 'training_loss': 7., 'validation_loss': 8.,
        'risk_action_origin': {'context': 'paid_risk'}}
    agent._learn_instruments = agent._revise_inputs = True
    agent._object_models[('fine',)] = deepcopy(model)
    agent._object_schemas[('fine',)] = deepcopy(model['schema'])
    policy = ResearchActionPolicy(rows, spec, agent.library.action_experience)
    policy.set_condition('unified:structured:calls:parent')
    policy.set_action_knowledge('program_replay', [{'terms': [square.to_dict()]}])
    frame = dict(spec=spec, training=rows, validation=validation, comparison=validation,
        transitions=[{'before': {'x': 1.}, 'after': {'x': 2.}}], parent=parent,
        assessed=[affine, retained], baseline=affine, retained_on_current=retained,
        uncompared=[], fallback=4., prepare_compute_cost=1234, remaining_compute=82868,
        terminal=False, independent_validation=True, observation_remaining=389,
        measurement_remaining=5181, directions=['program_replay', 'structural_research'],
        comparison_complete=True, structured=True, usable_parent=True,
        parent_fitted_comparison=False, state_identification=0, experiment_compute=0,
        learn_policy=True, policy=policy, calls=[x], call_origins={x: {'source_id': 'source'}},
        relation_kernels={square: [{'source_concept_id': 'source', 'radius_intervals': [[1., 2.]]}]},
        replay_plans=[('source', payload, template, [{'input': x}])])
    scene = dict(spec=spec, training=rows, validation=validation, transitions=frame['transitions'],
        parent=parent, object_model=model, knowledge=initial_knowledge,
        research_config=asdict(cfg), knowledge_additions=[{'file': 'roles.json'}],
        historical_knowledge_ref={'store': 'original.sqlite', 'sequence': 117195},
        references={'report': {'file': 'source.json'}, 'object_model': {'file': 'roles.json'},
                    'parent_original': {'file': 'source.json', 'json_path': ['parent']}})
    return agent, frame, scene


def test_restore_typed_program_keys_and_policy_library_without_scientific_work(tmp_path, monkeypatch):
    agent, frame, scene = prepared_scene()
    save_prepared(tmp_path, agent, frame, scene, [{'action': 'paid_preparation'}])

    def no_research(*args, **kwargs):
        raise AssertionError('Restoring paid preparation must not fit, score or select mappings')

    monkeypatch.setattr(Program, 'predict', no_research)
    monkeypatch.setattr(UnifiedResearchAgent, 'prepare_research_decision', no_research)
    monkeypatch.setattr(UnifiedResearchAgent, '_replay_program_queue', no_research)
    restored, actual, actions = load_prepared(tmp_path, scene)
    assert actions == [{'action': 'paid_preparation'}]
    assert {k: v for k, v in actual.items() if k != 'policy'} == {
        k: v for k, v in frame.items() if k != 'policy'}
    assert actual['baseline'] is actual['assessed'][0]
    assert actual['retained_on_current'] is actual['assessed'][1]
    assert actual['policy'].__dict__ == frame['policy'].__dict__
    assert actual['policy']._store is restored.library.action_experience
    assert actual['replay_plans'][0][1] is restored.library.meta_priors[0]['research_program']
    assert restored.library.to_dict() == agent.library.to_dict()
    assert restored._object_models == agent._object_models
    assert restored._object_schemas == agent._object_schemas
    assert restored._learn_instruments and restored._revise_inputs
    state = json.loads((tmp_path / 'prepared_state.json').read_text())
    assert state['assessed'][0]['file'] == str(tmp_path / 'prepared_pool.json')
    assert 'training' not in state and 'knowledge' not in state
    assert 'graph' not in state['replay_plans'][0]
    assert json.loads((tmp_path / 'reconstruction_ledger.json').read_text())['compute_cost'] == 1234


def test_saved_pair_restores_its_registered_source_instead_of_default_task(tmp_path, monkeypatch):
    agent, frame, scene = prepared_scene()
    save_prepared(tmp_path, agent, frame, scene, [])
    (tmp_path / 'registration.json').write_text(json.dumps({
        'source': '/retained/second-apparatus', 'source_seed': 106029102,
        'source_arm': 'pr3', 'decision_index': 79}))

    def restore(source, *, seed, arm, decision_index):
        assert (source, seed, arm, decision_index) == (
            '/retained/second-apparatus', 106029102, 'pr3', 79)
        return scene

    monkeypatch.setattr('scripts.run_native_decision_pair.load_native_decision_scene', restore)
    restored, actual, _ = load_prepared(tmp_path)
    assert actual['assessed'] == frame['assessed']
    assert restored.library.to_dict() == agent.library.to_dict()


def test_partial_object_recovery_does_not_install_a_null_complete_schema(tmp_path):
    agent, frame, scene = prepared_scene()
    scene['object_model']['schema'] = None
    agent._object_schemas.clear()
    scene['knowledge_base_ref'] = {'file': 'confirmed_method/knowledge_delta.json'}
    save_prepared(tmp_path, agent, frame, scene, [])
    restored, _, _ = load_prepared(tmp_path, scene)
    assert not restored._object_schemas
    state = json.loads((tmp_path / 'prepared_state.json').read_text())
    assert state['knowledge_base_ref'] == scene['knowledge_base_ref']


def test_frozen_run_uses_registered_task_before_any_research(tmp_path, monkeypatch):
    import scripts.run_native_decision_pair as runner

    (tmp_path / 'registration.json').write_text(json.dumps({
        'source': '/retained/second-apparatus', 'source_seed': 106029102,
        'source_arm': 'pr3', 'decision_index': 79}))
    monkeypatch.setattr(runner, 'ROOT', (tmp_path / 'runtime').resolve())

    class LoadedRegisteredScene(Exception):
        pass

    def restore(source, *, seed, arm, decision_index):
        assert (source, seed, arm, decision_index) == (
            '/retained/second-apparatus', 106029102, 'pr3', 79)
        raise LoadedRegisteredScene

    monkeypatch.setattr(runner, 'load_native_decision_scene', restore)
    with pytest.raises(LoadedRegisteredScene):
        runner.run(tmp_path)
    assert not (tmp_path / 'reconstruction_ledger.json').exists()


def test_restored_replay_executes_same_program_mapping_and_retains_whole_pool(tmp_path):
    agent, frame, scene = prepared_scene()
    save_prepared(tmp_path, agent, frame, scene, [])
    restored, actual, _ = load_prepared(tmp_path, scene)
    first, second = SearchBudget(2000), SearchBudget(2000)
    expected = agent.execute_research_decision(frame, 'program_replay', first, [])
    result = restored.execute_research_decision(actual, 'program_replay', second, [])
    assert expected == result and first.spent == second.spent
    assert result['execution_complete'] and result['full_comparison']
    assert result['new_claims'] and result['selected']['validation_loss'] < 1e-7
    assert result['claims'][:2] == frame['assessed']
    assert result['quote']['old_comparison_compute'] == 0


def test_metadata_failure_preserves_paid_pool_and_charge(tmp_path, monkeypatch):
    agent, frame, scene = prepared_scene()

    def failed_metadata(self):
        raise ValueError('interrupted metadata')

    monkeypatch.setattr(AbstractTemplate, 'to_dict', failed_metadata)
    with pytest.raises(ValueError, match='interrupted metadata'):
        save_prepared(tmp_path, agent, frame, scene, [])
    assert json.loads((tmp_path / 'reconstruction_ledger.json').read_text())['compute_cost'] == 1234
    assert len(json.loads((tmp_path / 'prepared_pool.json').read_text())) == 2
    assert not (tmp_path / 'prepared_state.json').exists()
