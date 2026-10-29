"""The column repair changes the executed method, not the common paid scene."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_learned_structure_successor as study
from scripts import run_native_decision_pair as pair
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, SearchBudget
from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.research_programs import _plan
from tests.test_native_decision_checkpoint import prepared_scene


@pytest.mark.parametrize('field', ['research_program', 'input_research_program'])
def test_repaired_prior_rebinds_the_saved_queue_without_replacing_paid_candidates(monkeypatch, field):
    agent, frame, _ = prepared_scene()
    original = agent.library.meta_priors[0]
    if field == 'input_research_program':
        original[field] = deepcopy(original['research_program'])
        source, _, template, mappings = frame['replay_plans'][0]
        frame['replay_plans'][0] = (source, original[field], template, mappings)
    baseline_agent, baseline_frame = deepcopy((agent, frame))
    replacement = deepcopy(original)
    square = Program('mul', children=(Program('feature', name='x'),)*2)
    z = Program('feature', name='z')
    replacement[field] = _plan({'fit_terms': (square, z), 'coefficients': [0., 1., 0.],
        'program': square, 'loss': 0.}, [], 7, 20)
    replacement[field]['contract_marker'] = 'independent_fitted_columns'
    frame['spec'] = replace(frame['spec'], feature_names=('x', 'z'))
    baseline_frame['spec'] = frame['spec']
    expanded_template = AbstractTemplate('source:abstract:0', 'source', square,
        ('input', 'extra'), ('x', 'z'), ())
    mapping = {'input': Program('feature', name='x'), 'extra': z}
    queue_calls = []

    def queue(spec, calls, observations=None):
        assert spec is frame['spec'] and calls == frame['calls']
        assert agent.library.meta_priors[0] == replacement
        queue_calls.append(True)
        return [('source', agent.library.meta_priors[0][field], expanded_template, [mapping])]

    monkeypatch.setattr(agent, '_replay_program_queue', queue)
    patch = {'prior_index': 0, 'source_concept_id': original['source_concept_id'],
             'prior': replacement, 'repaired_entry_ref': {'file': 'paid/repaired_entry.json'}}
    retained_pool, training = frame['assessed'], frame['training']
    actions = []
    pair.apply_method_program_revision(agent, frame, patch, actions)

    assert agent.library.meta_priors[0] == replacement
    assert queue_calls == [True]
    assert frame['replay_plans'][0][1] is agent.library.meta_priors[0][field]
    assert frame['replay_plans'][0][2] is expanded_template
    assert frame['replay_plans'][0][3] == [mapping]
    assert frame['assessed'] is retained_pool and frame['training'] is training
    assert baseline_agent.library.meta_priors[0] == original
    assert baseline_frame['replay_plans'][0][1] == original[field]
    assert 'contract_marker' not in baseline_frame['replay_plans'][0][1]
    assert frame['calls'] == baseline_frame['calls']
    assert frame['spec'] == baseline_frame['spec']

    # Exercise the ordinary four-export caller: changing the prior alone is
    # insufficient if this previously saved queue still points at the old body.
    source, payload, template, mappings = frame['replay_plans'][0]
    frame['replay_plans'][0] = (source, payload, template, mappings * 8)
    delivered = []

    def replay(body, rows, budget, **options):
        assert body is payload and body['contract_marker'] == 'independent_fitted_columns'
        assert rows is training and budget.spend(1)
        assert options['feature_mapping']['z'] == z
        delivered.append(options)
        return {'candidate': Candidate(f'replayed:{len(delivered)}',
                    Program('const', value=float(len(delivered))), 'replay'),
                'kind': 'replay', 'training_loss': 1., 'validation_loss': 1.}

    monkeypatch.setattr('hegel_machine.cognition.research_programs.replay_research_program', replay)
    result = agent.execute_research_decision(frame, 'program_replay', SearchBudget(2000), actions)
    assert len(delivered) == len(result['new_claims']) == 4
    assert result['claims'][:len(retained_pool)] == retained_pool
    assert result['quote']['old_comparison_compute'] == 0


def test_same_replay_action_keeps_distinct_original_and_repaired_method_bodies(tmp_path):
    patch_ref = pair.ref(tmp_path / 'method_revision.json')
    configuration = {'operations': {'baseline': 'program_replay', 'challenger': 'program_replay'},
        'knowledge_contract_ref': {'file': 'frozen_k2/knowledge_delta.json'},
        'method_versions': {'baseline': 3, 'challenger': 4},
        'method_program_revisions': {'challenger': patch_ref}}
    bound = pair.bind_native_method_artifacts(tmp_path, configuration,
        pair.ref(tmp_path / 'prepared_state.json'))
    old = pair.load(Path(bound['baseline']['method_artifact_ref']['file']))
    revised = pair.load(Path(bound['challenger']['method_artifact_ref']['file']))
    assert old['id'] == revised['id'] == 'native:program_replay'
    assert old['version'] == 3 and revised['version'] == 4
    assert old['execution']['knowledge_ref'] == revised['execution']['knowledge_ref']
    assert old['execution']['research_action'] == revised['execution']['research_action'] == 'program_replay'
    assert 'research_program_revision_ref' not in old['execution']
    assert revised['execution']['research_program_revision_ref'] == patch_ref


def test_new_decision_reuses_only_the_original_baseline_artifact(tmp_path):
    historical, successor = tmp_path / 'historical', tmp_path / 'successor'
    knowledge_ref = {'file': 'frozen_k2/knowledge_delta.json'}
    historical_decision = {'file': 'old_response.json', 'json_path': ['actions', 77]}
    original = pair.bind_native_method_artifacts(historical, {
        'operations': {'baseline': 'program_replay', 'challenger': 'structural_research'},
        'method_version': 3, 'knowledge_contract_ref': knowledge_ref}, historical_decision)
    retained_ref = original['baseline']['method_artifact_ref']
    retained = pair.load(Path(retained_ref['file']))
    current_decision = pair.ref(successor / 'prepared_state.json', 'directions')
    config = {'operations': {'baseline': 'program_replay', 'challenger': 'program_replay'},
        'knowledge_contract_ref': knowledge_ref,
        'method_versions': {'baseline': 3, 'challenger': 4},
        'method_program_revisions': {'challenger': pair.ref(successor / 'method_revision.json')},
        'retained_method_artifact_refs': {'baseline': retained_ref}}
    actual = pair.bind_native_method_artifacts(successor, config, current_decision)
    assert actual['baseline']['method_artifact_ref'] == retained_ref
    assert not (successor / 'baseline_method.json').exists()
    assert pair.load(Path(retained_ref['file'])) == retained
    assert retained['source_ref'] == historical_decision
    assert pair.load(successor / 'method_bindings.json')['decision_ref'] == current_decision
    assert pair.load(Path(actual['challenger']['method_artifact_ref']['file']))['source_ref'] == current_decision
    # Only provenance may differ. A real execution change cannot borrow @3.
    wrong = {**config, 'knowledge_contract_ref': {'file': 'different_k/knowledge_delta.json'}}
    with pytest.raises(ValueError, match='Retained native method body differs'):
        pair.bind_native_method_artifacts(tmp_path / 'wrong', wrong, current_decision)


def test_new_successor_keeps_one_common_preparation_and_equal_total_allowance(tmp_path, monkeypatch):
    from scripts import successor_interval_scene

    knowledge, source, review, output = [tmp_path / name for name in ('k2', 'source', 'review', 'output')]
    study.save(source / 'registration.json', {'historical_knowledge_ref': {
        'store': 'original.sqlite', 'sequence': 7}, 'tasks': [{'seed': 106029101}]})
    study.save(review / 'common/questions.json', [{'id': 'task:method_choice',
        'method_use_plan': {'native_return_refs': [{'file': 'first.json'}, {'file': 'second.json'}]}}])
    baseline_artifact = study.ref(review / 'baseline_method.json')
    study.save(review / 'baseline/outcome.json', {'method_artifact_ref': baseline_artifact})
    study.save(review / 'pair.json', {'baseline_ref': study.ref(review / 'baseline/outcome.json')})
    revised_prior = {'source_concept_id': 'concept_0004', 'research_program': {'columns': 8}}
    retained = {'prior_index': 0, 'source_concept_id': 'concept_0004',
        'repaired_entry_ref': {'file': str(tmp_path / 'repair/entry.json')}}
    library = SimpleNamespace(meta_priors=[revised_prior])
    reads, frozen = [], []

    def history(path, readonly):
        assert path == 'original.sqlite' and readonly
        return SimpleNamespace(connection=SimpleNamespace(execute=lambda *args: []), close=lambda: None)

    def repaired(root, entry_ref):
        reads.append((root, entry_ref))
        return library, retained

    monkeypatch.setattr(study, 'HistoryStore', history)
    monkeypatch.setattr(successor_interval_scene, 'load_repaired_knowledge', repaired)
    monkeypatch.setattr(study, 'freeze_runtime', lambda output, source: frozen.append((output, source)))
    cfg = study.register_column_successor(knowledge, retained['repaired_entry_ref'], source, review, output)
    assert reads == [(knowledge, retained['repaired_entry_ref'])]
    assert frozen == [(output, knowledge)]
    assert cfg['source_seed'] == cfg['task']['seed'] == 106029105
    assert cfg['operations'] == {'baseline': 'program_replay', 'challenger': 'program_replay'}
    assert cfg['knowledge_loader'] == 'confirmed_interval'
    assert cfg['knowledge_contract_ref'] == study.ref(knowledge / 'knowledge_delta.json')
    assert cfg['preparation_compute_limit'] == 110000
    assert cfg['per_arm_compute_limit'] + cfg['confirmation_compute_per_arm'] == 80000
    assert cfg['common_compute_limit'] + 2 * cfg['per_arm_total_compute_limit'] == 270004
    assert cfg['method_versions'] == {'baseline': 3, 'challenger': 4}
    assert cfg['retained_method_artifact_refs'] == {'baseline': baseline_artifact}
    assert cfg['quality_threshold'] == 1e-7
    assert cfg['method_program_revisions'] == {'challenger': study.ref(output / 'method_revision.json')}
    assert study.load(output / 'method_revision.json')['prior'] == revised_prior

