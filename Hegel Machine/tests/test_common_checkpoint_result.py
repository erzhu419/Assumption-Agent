"""Synchronous checkpoints retain values without copying the whole native bank."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import (
    common_archive_inquiry, online_inquiry_llm, temporal_paired_method,
)
from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from test_common_archive_resume import SEED, agent_at, checkpoint
from test_question_routing import theory


def native_history():
    original = theory('retained_native', Program('const', value=1.), version=1,
                      support={'status': 'refuted', 'evidence': [{'record_index': 0}]})
    return {original['id']: theory(original['id'], Program('const', value=2.),
        version=2, history=[original],
        source_snapshot={'index_ref': {'file': 'original_index.json', 'json_path': []}})}


def test_synchronous_checkpoint_keeps_full_JSON_without_whole_index_deepcopy(
        tmp_path, monkeypatch):
    agent = agent_at(tmp_path)
    agent.theory_index = native_history()
    agent._prepare_program_state()
    expected_state = json.loads((agent.workspace / 'research_state.json').read_text())
    expected_result = agent.result('researching')
    original_copy = common_archive_inquiry.deepcopy

    def no_whole_index_copy(value, *args, **kwargs):
        assert value is not agent.theory_index, 'synchronous save copied the complete native index'
        return original_copy(value, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(common_archive_inquiry, 'deepcopy', no_whole_index_copy)
        agent._prepare_program_state()
        transient = agent.checkpoint_result('researching')
        assert transient['theory_index'] is agent.theory_index
        write_json_checkpoint(agent.workspace / 'training.json', transient)

    assert json.loads((agent.workspace / 'research_state.json').read_text()) == expected_state
    saved = json.loads((agent.workspace / 'training.json').read_text())
    assert saved == json.loads(json.dumps(expected_result))
    agent.theory_index['retained_native']['program']['value'] = 9.
    agent.theory_index['retained_native']['history'][0]['support']['evidence'].append({'record_index': 3})
    assert expected_result['theory_index']['retained_native']['program']['value'] == 2.
    assert len(expected_result['theory_index']['retained_native']['history'][0]['support']['evidence']) == 1
    assert json.loads((agent.workspace / 'training.json').read_text()) == saved


def test_borrowed_checkpoint_restores_paid_scene_versions_candidates_and_predictions(
        tmp_path, monkeypatch):
    old, laboratory, _ = checkpoint(tmp_path)
    old.theory_index = native_history()
    (old.workspace / 'retained_native.py').write_text('def predict(row): return 2.\n')
    old.candidate_paths.append('retained_native.py')
    old._prepare_program_state()
    saved_state = json.loads((old.workspace / 'research_state.json').read_text())
    expected_index = deepcopy(old.theory_index)
    expected_parent = old._evaluated_summary()
    expected_predictions = {identifier: Program.from_dict(item['program']).predict(old.rows[0])
                            for identifier, item in old.theory_index.items()}
    write_json_checkpoint(old.workspace / 'training.json', old.checkpoint_result('researching'))
    previous = json.loads((old.workspace / 'training.json').read_text())
    old.theory_index['retained_native']['program']['value'] = 99.

    resumed, target = agent_at(tmp_path / 'restored'), TemporalInquiryLab(task_index=1)
    monkeypatch.setattr(target, 'execute', lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(AssertionError('restoration must not buy observations')))
    restored = resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)

    assert restored['new_observations'] == 0
    assert resumed.theory_index == expected_index
    assert resumed.candidate_paths == saved_state['exported_candidate_paths']
    assert resumed.rows == old.rows
    assert resumed.observations == json.loads(json.dumps(old.observations))
    assert resumed.costs == old.costs and target.costs == laboratory.costs
    assert resumed._evaluated_summary() == expected_parent
    assert {identifier: Program.from_dict(item['program']).predict(resumed.rows[0])
            for identifier, item in resumed.theory_index.items()} == expected_predictions
    assert (resumed.workspace / 'retained_native.py').read_text() == (
        old.workspace / 'retained_native.py').read_text()
    assert json.loads((old.workspace / 'research_state.json').read_text()) == saved_state


def test_checkpoint_does_not_copy_paid_history_and_keeps_ordinary_result_owned(tmp_path, monkeypatch):
    agent = agent_at(tmp_path)
    question = agent._open_question({'id': 'paid', 'question': 'Explain retained errors',
        'parent_id': None, 'depends_on': []})
    question['evidence'] = [{'record_index': 0, 'retained': [1, 2]}]
    agent.events.append({'action': 'run', 'arguments': {'operator': 'research_choice'},
        'feedback': {'evaluations': [{'model': 'old.py', 'mse': .3}]}})
    agent.method_results.append({'path': 'other.py', 'step': 0,
        'feedback': {'error': 'retained failure', 'paid': [1, 2]}})
    agent.program_contexts['other.py'] = {'source': 'retained source', 'start': 0}
    agent.operator_results.append({'operator': 'research_choice', 'evaluations': [
        {'model': 'old.py', 'mse': .3, 'costs': {'compute_cost': 17}}]})
    agent.research_candidates['old.py'] = {'status': 'retained', 'scores': [.3]}
    agent.relation_research['old'] = {'failures': [{'error': 'retained'}]}
    agent.theory_index = native_history()
    agent._prepare_program_state()
    state_before = json.loads((agent.workspace / 'research_state.json').read_text())
    owned = agent.result('researching')
    owned_before = json.loads(json.dumps(owned))
    names = ('events', 'questions', 'method_results', 'program_contexts',
        'operator_results', 'research_candidates', 'relation_research', 'theory_index',
        'costs', 'observations', 'revisions')
    paid_banks = [getattr(agent, name) for name in names]
    original_copy = deepcopy

    def reject_paid_bank_copy(value, *args, **kwargs):
        assert all(value is not bank for bank in paid_banks), 'copied complete paid history during save'
        return original_copy(value, *args, **kwargs)

    with monkeypatch.context() as patched:
        for module in (common_archive_inquiry, online_inquiry_llm, temporal_paired_method):
            patched.setattr(module, 'deepcopy', reject_paid_bank_copy)
        agent._prepare_program_state()
        borrowed = agent.checkpoint_result('researching')
        write_json_checkpoint(agent.workspace / 'training.json', borrowed)

    assert json.loads((agent.workspace / 'research_state.json').read_text()) == state_before
    assert json.loads((agent.workspace / 'training.json').read_text()) == owned_before
    # The ordinary public result must still outlive further research safely.
    for bank in paid_banks:
        if isinstance(bank, dict):
            bank['later'] = {'new': True}
        else:
            bank.append({'later': True})
    assert json.loads(json.dumps(owned)) == owned_before
    assert json.loads((agent.workspace / 'training.json').read_text()) == owned_before


@pytest.mark.parametrize('status', ['researching', 'token_budget_exhausted'])
def test_outgoing_references_do_not_read_discarded_index_and_library(tmp_path, monkeypatch, status):
    agent = agent_at(tmp_path)
    agent.theory_index = native_history()
    agent.theory_consolidation = {'executions': {'paid': {'status': 'retained'}}}
    paths = {'shared/theory_index.json', 'shared/theory_library.json'}
    for name in paths:
        (agent.workspace / name).write_text('{"retained": true}')
        agent.files[name] = '{"retained": true}'
    # Isolate outgoing file enumeration: the index publisher has its own writer
    # and retained-file cache; this regression concerns its second, unused read.
    monkeypatch.setattr(common_archive_inquiry, 'save_theory_index', lambda _agent: None)
    read = Path.read_text

    def no_discarded_read(path, *args, **kwargs):
        assert path not in {agent.workspace / name for name in paths}, 'read a file later discarded'
        return read(path, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(Path, 'read_text', no_discarded_read)
        result = agent.checkpoint_result(status)
    assert all(name not in result['outgoing_knowledge']['files'] for name in paths)
    for key, name in (('theory_index', 'shared/theory_index.json'),
                      ('theory_consolidation', 'shared/theory_library.json')):
        assert result['outgoing_knowledge'][key] == {'file': str(agent.workspace / name), 'json_path': []}
    # Provider context still receives the original complete shared-file view.
    assert paths <= agent._shared_files().keys()


def test_snapshot_builder_keeps_frozen_method_fields_in_existing_multiple_inheritance(tmp_path):
    from hegel_machine.cognition.anonymous_method_learning import AnonymousPreparedMethodAgent

    agent = AnonymousPreparedMethodAgent(None, tmp_path, arm='hegel', source_task=False)
    agent.bind_method('def select_actions(spec, observations, memory): return []\n')
    agent._prepare_program_state()
    checkpoint = agent.checkpoint_result('researching')
    owned = agent.result('researching')
    assert checkpoint['method_path'] == owned['method_path']
    assert checkpoint['method_source'] == owned['method_source'] == agent.frozen_method_source
    assert checkpoint['method_results'] == owned['method_results'] == agent.method_results


def test_direct_common_state_save_releases_idle_forks_before_building_snapshot(tmp_path, monkeypatch):
    from multiprocessing import get_context
    from hegel_machine.cognition import experiment_runtime, parallel_prediction

    agent = agent_at(tmp_path)
    expected = json.loads(json.dumps(agent._program_state()))
    costs = deepcopy(agent.costs)
    build = agent._program_state
    cpus = tuple(experiment_runtime.physical_worker_cpus()[:2])
    parallel_prediction.configure_prediction_workers(cpus)
    monkeypatch.setattr(experiment_runtime, '_CHECKPOINT_LOCK', get_context('fork').RLock())
    try:
        executor = parallel_prediction._executor((None,))
        assert executor.submit(parallel_prediction._predict_block,
            (Program('const', value=2.), 0, 1)).result(timeout=15) == [2.]

        def build_after_idle_workers_release():
            assert parallel_prediction._EXECUTOR is None, 'snapshot built while idle forks retain its agent'
            return build()

        monkeypatch.setattr(agent, '_program_state', build_after_idle_workers_release)
        # The direct Common action path also reaches this method without an
        # outer registered checkpoint. A surrounding save may nest it safely.
        agent._prepare_program_state()
        with experiment_runtime.checkpoint_execution():
            agent._prepare_program_state()
        assert json.loads((agent.workspace / 'research_state.json').read_text()) == expected
        assert agent.costs == costs
        assert parallel_prediction._CPUS == cpus
    finally:
        parallel_prediction.configure_prediction_workers(())
