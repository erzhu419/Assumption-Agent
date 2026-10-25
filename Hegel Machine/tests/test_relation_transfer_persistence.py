"""Persist bounded relation batches once without dropping failed completed work."""
from copy import deepcopy
import json

import pytest

import hegel_machine.cognition.theory_consolidation as consolidation
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import register_theories
from test_question_routing import CONTEXT, theory
from test_theory_consolidation import (
    agent_scene, assert_no_new_science, consolidate, context_relation_scene, transfer,
)


def context_batch(tmp_path, *, target=2.):
    agent, lab, _, _, relation = context_relation_scene(tmp_path)
    lag = Program('lag', name='u', lag=1)
    register_theories(agent, [theory('lag_component', lag),
        theory('scaled_component', Program('mul', children=(Program('const', value=3.), lag)))])
    arguments = {'operator': 'theory_transfer', 'candidate_id': relation['id'],
        'theory_id': 'guarded', 'component_theory_ids': ['lag_component', 'scaled_component'],
        'context_ref': deepcopy(CONTEXT), 'output_unit': 'V',
        'allocation': {'compute': 5000},
        'evidence_checks': [{'source_ref': {'file': 'already_paid.json', 'json_path': [0]},
            'context_ref': deepcopy(CONTEXT), 'tolerance': 0.,
            'observation': {'features': {'u': 7.}, 'target': target, 'history': [{'u': 2.}]}}]}
    return agent, lab, arguments


def observe_library_writes(agent, monkeypatch):
    writes = []
    save = agent._save
    def record(path, content):
        if path == consolidation.LIBRARY_PATH:
            writes.append(json.loads(content))
        return save(path, content)
    monkeypatch.setattr(agent, '_save', record)
    return writes


def test_completed_relation_batch_saves_once_with_same_result_cost_and_negative_result(tmp_path, monkeypatch):
    baseline, old_lab, old_arguments = context_batch(tmp_path / 'baseline')
    agent, lab, arguments = context_batch(tmp_path / 'batched')
    original = consolidation._checked_transfer
    old_writes = observe_library_writes(baseline, monkeypatch)
    writes = observe_library_writes(agent, monkeypatch)
    def immediate(*args, **kwargs):
        kwargs.pop('persist', None)
        return original(*args, **kwargs)
    monkeypatch.setattr(consolidation, '_checked_transfer', immediate)
    old_before = baseline.costs['compute_cost']
    old_result = baseline._execute('run', old_arguments, old_lab)
    monkeypatch.setattr(consolidation, '_checked_transfer', original)
    before = agent.costs['compute_cost']
    result = agent._execute('run', arguments, lab)
    assert len(old_writes) == len(result['generated_theories']) + 1 == 3
    assert len(writes) == 1
    assert result == old_result
    assert agent.costs['compute_cost'] - before == baseline.costs['compute_cost'] - old_before
    assert agent.theory_index == baseline.theory_index
    assert agent.theory_consolidation['transfers'] == baseline.theory_consolidation['transfers']
    assert agent.theory_consolidation['executions']['relation_transfer:guarded'] == (
        baseline.theory_consolidation['executions']['relation_transfer:guarded'])
    assert agent.theory_consolidation['candidates'] == baseline.theory_consolidation['candidates']
    assert {item['status'] for item in result['generated_theories']} == {'candidate_registered', 'refuted'}
    from hegel_machine.cognition.research_episodes import load_source_values
    assert load_source_values([{'file': consolidation.LIBRARY_PATH, 'json_path': []}],
        agent.workspace)[0]['value'] == agent.theory_consolidation
    assert 'check_basis_ref' in writes[0]['transfers'][1]
    assert all('binding_ref' in item and 'checks_ref' in item for item in
               writes[0]['executions']['relation_transfer:guarded']['result']['generated_theories'])
    assert_no_new_science(agent, lab)


def test_relation_batch_exception_persists_completed_counterexample_and_actual_cost(tmp_path, monkeypatch):
    agent, lab, arguments = context_batch(tmp_path, target=99.)
    arguments['mission'] = {'mission_id': 'interrupted_batch', 'purpose': 'theory_learning',
        'theory_learning_allocation': {'compute': 5000}}
    writes = observe_library_writes(agent, monkeypatch)
    original = consolidation._checked_transfer
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(args[1]['theory_id'])
        if len(calls) == 2:
            raise OSError('interrupted second candidate')
        return original(*args, **kwargs)
    monkeypatch.setattr(consolidation, '_checked_transfer', interrupted)
    before = agent.costs['compute_cost']
    with pytest.raises(OSError, match='interrupted second candidate'):
        agent._execute('run', arguments, lab)
    assert len(calls) == 2 and len(writes) == 1
    completed, = writes[0]['transfers']
    assert completed['theory_id'] == calls[0]
    assert completed['status'] == 'refuted'
    assert completed['checks'][0]['status'] == 'failed'
    assert completed['checks'][0]['actual'] == 99.
    assert completed['theory_id'] in agent.theory_index
    spent = agent.costs['compute_cost'] - before
    assert spent > 0 and agent.research_mission['resource_usage']['compute'] == spent
    # The existing checkpoint result retains these costs and the registered
    # index; no recovery-only cost field is added to the theory library.
    saved = agent.result('failed')
    assert saved['costs']['compute_cost'] == agent.costs['compute_cost']
    assert saved['theory_index'][calls[0]] == agent.theory_index[calls[0]]
    assert_no_new_science(agent, lab)


def test_single_template_transfer_still_persists_its_counterexample_immediately(tmp_path, monkeypatch):
    agent, lab = agent_scene(tmp_path)
    built = consolidate(agent, lab)
    candidate, = built['frozen_candidates']
    writes = observe_library_writes(agent, monkeypatch)
    result = transfer(agent, lab, candidate['candidate_id'], evidence_checks=[{
        'source_ref': {'file': 'already_paid.json', 'json_path': [0]},
        'context_ref': deepcopy(CONTEXT),
        'observation': {'features': {'u': 1.}, 'target': 99., 'history': []}, 'tolerance': 0.}])
    assert result['status'] == 'refuted' and len(writes) == 1
    assert writes[0]['transfers'][-1]['checks'][0]['actual'] == 99.
    assert_no_new_science(agent, lab)


def test_compact_library_restores_after_rename_without_mutable_source_or_path_changes(tmp_path):
    from hegel_machine.cognition.research_episodes import load_source_values
    agent, lab, arguments = context_batch(tmp_path / 'source')
    result = agent._execute('run', arguments, lab)
    original = deepcopy(agent.theory_consolidation)
    source = agent.workspace / consolidation.LIBRARY_PATH
    frozen = tmp_path / 'frozen' / 'renamed.json'
    frozen.parent.mkdir()
    frozen.write_bytes(source.read_bytes())
    # The frozen artifact must not depend on the writable source's later value.
    source.write_text('{"transfers": []}')
    raw = json.loads(frozen.read_text())
    assert 'check_basis' not in raw['transfers'][1]
    paths = [['transfers', 1, 'check_basis'],
        ['executions', 'relation_transfer:guarded', 'result', 'generated_theories', 1, 'binding'],
        ['executions', 'relation_transfer:guarded', 'result', 'generated_theories', 1, 'checks']]
    references = [{'file': str(frozen), 'json_path': path} for path in paths]
    loaded = [item['value'] for item in load_source_values(references, tmp_path)]
    assert loaded == [original['transfers'][1]['check_basis'],
        result['generated_theories'][1]['binding'], result['generated_theories'][1]['checks']]
    restored = consolidation._read_library({'file': str(frozen), 'json_path': [],
        'workspace': str(agent.workspace)}, tmp_path)
    # Anchoring relative source evidence is the established restore behavior.
    expected = consolidation._anchor_file_references(original, agent.workspace)
    assert restored == expected
    restored['transfers'][1]['check_basis'][0]['tolerance'] = 123.
    assert restored['transfers'][0]['check_basis'][0]['tolerance'] == 0.
    generated = restored['executions']['relation_transfer:guarded']['result']['generated_theories']
    generated[0]['checks'][0]['actual'] = 4321.
    generated[0]['binding']['context_ref']['source_id'] = 'changed'
    assert restored['transfers'][0]['checks'][0]['actual'] != 4321.
    assert restored['transfers'][0]['binding']['context_ref'].get('source_id') != 'changed'
    assert_no_new_science(agent, lab)


def test_distinct_counterexample_contract_stays_inline_and_round_trips(tmp_path):
    from hegel_machine.cognition.research_episodes import load_source_values
    agent, lab, arguments = context_batch(tmp_path)
    agent._execute('run', arguments, lab)
    altered = deepcopy(agent.theory_consolidation['transfers'][0])
    altered['check_basis'][0]['tolerance'] = 0.125
    agent.theory_consolidation['transfers'].append(altered)
    consolidation._save(agent)
    saved = json.loads((agent.workspace / consolidation.LIBRARY_PATH).read_text())
    assert saved['transfers'][-1]['check_basis'][0]['tolerance'] == 0.125
    loaded = load_source_values([{'file': consolidation.LIBRARY_PATH,
        'json_path': ['transfers', len(saved['transfers']) - 1, 'check_basis']}], agent.workspace)[0]['value']
    assert loaded == altered['check_basis']


def test_source_reader_keeps_non_address_binding_metadata_distinct(tmp_path):
    from hegel_machine.cognition.research_episodes import load_source_values
    # Executable-evidence instances also call their input metadata binding_ref;
    # it is not an address for a missing binding field.
    source = tmp_path / 'instance.json'
    source.write_text(json.dumps({'binding_ref': {'u': 'public_control'}}))
    with pytest.raises(KeyError, match='binding'):
        load_source_values([{'file': str(source), 'json_path': ['binding']}], tmp_path)
