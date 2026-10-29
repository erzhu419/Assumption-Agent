"""Frozen K2 belongs to both native arms, their artifacts and paid restoration."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts import run_native_decision_pair as runner
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, SearchBudget
from tests.test_native_decision_checkpoint import prepared_scene
from tests.test_native_paired_method_feedback import common, deliver, route


def test_k2_registration_keeps_original_pair_limits_and_freezes_required_loaders(tmp_path, monkeypatch):
    source, prior, output, knowledge = [tmp_path/name for name in ('source', 'review', 'pair', 'k2')]
    scene = {'choice': {'eligible_actions': ['program_replay', 'structural_research']},
        'choice_index': 79, 'ordinary_operation': 'program_revision',
        'references': {'choice': {'file': 'old/choices.json', 'json_path': [79]}},
        'source_task_ref': {'file': 'old/task.json'}, 'remaining_compute': 78193,
        'historical_knowledge_ref': {'store': 'old.sqlite', 'sequence': 7},
        'training': [None]*129, 'validation': [None]*24,
        'research_config': {'minimum_improvement': 1e-7},
        'restoration_scope': 'retained historical decision',
        'object_identity_remap': {'original_model_id': 'objects_original',
                                 'current_model_id': 'objects_current'}}
    restored, frozen = [], []

    def restore(actual_source, **options):
        restored.append((actual_source, options))
        return scene

    runner.save(prior/'common/questions.json', [{'id': 'task:method_choice',
        'method_use_plan': {'native_return_refs': [{'file': 'previous-pair.json'}]}}])
    monkeypatch.setattr(runner, 'load_native_decision_scene', restore)
    monkeypatch.setattr(runner, 'write_text_incrementally',
        lambda path, text, **kwargs: frozen.append(Path(path).relative_to(output/'runtime')))
    cfg = runner.register(source, prior, output, seed=106029102, arm='pr3',
                          decision_index=79, knowledge_root=knowledge)

    assert restored == [(source, {'seed': 106029102, 'arm': 'pr3', 'decision_index': 79,
                                  'knowledge_root': knowledge})]
    assert cfg['operations'] == {'baseline': 'program_replay', 'challenger': 'structural_research'}
    assert cfg['preparation_compute_limit'] == 90000 and cfg['per_arm_compute_limit'] == 78193
    assert cfg['per_arm_active_wall_seconds'] == 1800 and cfg['maximum_new_observations'] == 0
    assert cfg['quality_threshold'] == 1e-7 and cfg['method_review_compute_limit'] == 3
    assert cfg['method_version'] == 3
    assert cfg['knowledge_scope'] == 'retrospective_development_with_frozen_later_knowledge'
    assert cfg['knowledge_contract_ref'] == runner.ref(knowledge/'knowledge_delta.json')
    assert cfg['source_cost_refs'] == [runner.ref(knowledge/'receipt.json')]
    bindings = runner.bind_native_method_artifacts(output, cfg, cfg['choice_ref'])
    for name, operation in cfg['operations'].items():
        artifact = runner.load(Path(bindings[name]['method_artifact_ref']['file']))
        assert artifact['version'] == 3
        assert artifact['execution'] == {'entry': 'UnifiedResearchAgent.execute_research_decision',
            'research_action': operation, 'knowledge_ref': cfg['knowledge_contract_ref']}
    assert {Path('scripts')/name for name in ('successor_interval_scene.py',
        'revise_learned_structure_successor.py', 'run_learned_structure_successor.py',
        'retain_confirmed_structure.py', 'partial_role_paid_scene.py')} <= set(frozen)


def test_k2_method_artifacts_form_their_own_condition_group(tmp_path):
    agent = common(tmp_path)
    _, k0 = deliver(agent, tmp_path, 'k0', decision_index=58)
    _, k1 = deliver(agent, tmp_path, 'k1', decision_index=0, version=2,
        knowledge_ref={'file': 'k1/knowledge_delta.json'})
    _, k2 = deliver(agent, tmp_path, 'k2', decision_index=79, version=3,
        knowledge_ref={'file': 'k2/knowledge_delta.json'})
    reviewed = route(agent)['method_action_research']
    assert not reviewed['paired_method_exclusions']
    assert [group['pair_refs'] for group in reviewed['paired_condition_research']] == [[k0], [k1], [k2]]
    assert all(group['condition'] is None for group in reviewed['paired_condition_research'])


def test_paid_preparation_loads_registered_k2_without_refitting_and_keeps_four_export_cap(tmp_path, monkeypatch):
    agent, frame, scene = prepared_scene()
    k2_prior = {'source_concept_id': 'confirmed-k2-marker', 'status': 'active'}
    agent.library.meta_priors.append(deepcopy(k2_prior))
    scene['knowledge'] = deepcopy(agent.library.to_dict())
    scene['knowledge_base_ref'] = runner.ref(tmp_path/'k2/knowledge_delta.json')
    source, knowledge = '/paid/historical-apparatus', '/paid/k2'
    runner.save_prepared(tmp_path, agent, frame, scene, [])
    runner.save(tmp_path/'registration.json', {'source': source, 'source_seed': 106029102,
        'source_arm': 'pr3', 'decision_index': 79, 'knowledge_root': knowledge})
    calls = []

    def restore(actual_source, **options):
        calls.append((actual_source, options))
        return scene

    monkeypatch.setattr(runner, 'load_native_decision_scene', restore)
    restored, actual, _ = runner.load_prepared(tmp_path)
    assert calls == [(source, {'seed': 106029102, 'arm': 'pr3', 'decision_index': 79,
                              'knowledge_root': knowledge})]
    assert restored.library.meta_priors[-1] == k2_prior
    assert actual['assessed'] == frame['assessed']
    assert runner.load(tmp_path/'prepared_state.json')['knowledge_base_ref'] == scene['knowledge_base_ref']
    source_id, payload, template, mappings = actual['replay_plans'][0]
    actual['replay_plans'][0] = (source_id, payload, template, mappings*6)
    replays = []

    def replay(plan, training, budget, **kwargs):
        assert plan is payload and training == actual['training']
        assert budget.spend(1)
        replays.append(kwargs)
        program = Program('const', value=float(len(replays)))
        return {'candidate': Candidate(f'controlled:{len(replays)}', program, 'replay'),
                'kind': 'replay', 'training_loss': 1., 'validation_loss': 1.}

    monkeypatch.setattr('hegel_machine.cognition.research_programs.replay_research_program', replay)
    result = restored.execute_research_decision(actual, 'program_replay', SearchBudget(2000), [])
    assert len(replays) == len(result['new_claims']) == result['quote']['maximum_new_candidates'] == 4
    assert result['quote']['old_comparison_compute'] == 0
    assert result['claims'][:len(frame['assessed'])] == frame['assessed']


def test_frozen_run_passes_k2_before_any_preparation(tmp_path, monkeypatch):
    runner.save(tmp_path/'registration.json', {'source': '/paid/original',
        'source_seed': 106029102, 'source_arm': 'pr3', 'decision_index': 79,
        'knowledge_root': '/paid/k2'})
    monkeypatch.setattr(runner, 'ROOT', (tmp_path/'runtime').resolve())

    class SceneReached(Exception):
        pass

    def restore(source, **options):
        assert source == '/paid/original'
        assert options == {'seed': 106029102, 'arm': 'pr3', 'decision_index': 79,
                           'knowledge_root': '/paid/k2'}
        raise SceneReached

    monkeypatch.setattr(runner, 'load_native_decision_scene', restore)
    with pytest.raises(SceneReached):
        runner.run(tmp_path)
    assert not (tmp_path/'reconstruction_ledger.json').exists()
