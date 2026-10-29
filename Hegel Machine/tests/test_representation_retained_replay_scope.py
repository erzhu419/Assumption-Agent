"""The declared saved-coordinate replay scope reaches the real Common caller."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from hegel_machine.cognition.research_programs import _entry
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_native_representation_inquiry import run, agent_with_records
from test_native_representation_routing import scene, declaration


def saved_conditional(tmp_path):
    current = Program('feature', name='u')
    gate = Program('ge', children=(current, Program('const', value=.45)))
    active = Program('if', children=(gate, current, Program('const', value=0.)))
    window = Program('window_sum', lag=2, children=(active,))
    terms = [current, window]
    parent = _sum_terms(terms, [.3, .7], 2.)
    fit = {'program': parent, 'loss': .5, 'terms': terms,
        'fit_terms': terms, 'coefficients': [2., .3, .7]}
    step = {'operation': 'discover_conditional_coordinates',
        'question': 'Synthetic saved fitted branch', 'program': parent, 'loss': .5,
        'gate_coordinate': current.to_dict(), 'branch_basis': [current.to_dict()],
        'threshold_interval': [.4, .5], 'width': 2, 'placement': 'inside_window'}
    claim = _entry(fit, [step], 48, 192)
    path = tmp_path / 'saved_conditional.json'
    path.write_text(json.dumps({'candidate': {'program': parent.to_dict(),
        'research_program': claim['research_program']}}))
    return parent, {'file': str(path), 'json_path': ['candidate']}


def test_common_plan_runs_fixed_saved_inputs_and_scope_change_invalidates_cache(tmp_path, monkeypatch):
    from hegel_machine.cognition import conditional_coordinates
    from hegel_machine.cognition.research_programs import replay_research_program

    agent, lab = scene(tmp_path)
    parent, reference = saved_conditional(tmp_path)
    source = (tmp_path / 'saved_conditional.json').read_bytes()
    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', lambda *a: [])
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims',
        lambda *a, replayed, **k: replayed)
    monkeypatch.setattr(conditional_coordinates, 'replay_conditional_coordinates',
        lambda *a, **k: pytest.fail('fixed-coordinate Common plan repeated gate/window search'))
    plan = {'parent_theory_ids': ['old'], 'validation_records': [4],
        'compute_limit': 50000, 'max_candidates': 4,
        'retained_research_program_refs': [reference],
        'retained_replay_scope': 'retained_coordinates'}
    arguments = {'operator': 'research_route', 'research_question': {
        **declaration(), 'kind': 'program_revision'},
        'theories': [{'id': 'old', 'program': parent.to_dict()}],
        'native_representation_plan': plan}
    first = agent._execute('run', arguments, lab)
    native = load_native_representation_result(first['native_representation_ref'], agent.workspace)
    replay, = [a for a in native['actions']
        if a['action'] == 'replay_retained_parent_research_program']
    assert native['retained_replay_scope'] == replay['execution_scope'] == 'retained_coordinates'
    assert replay['status'] == 'refitted_requires_validation'
    assert replay['compute_cost'] == 48 * (2 + 2)
    assert native['candidates'][0]['research_program']['execution_scope'] == 'retained_coordinates'
    assert (tmp_path / 'saved_conditional.json').read_bytes() == source
    before = deepcopy(agent.costs)
    same = agent._execute('run', arguments, lab)
    assert same['cache_hit'] and same['new_compute_cost'] == 0 and agent.costs == before
    called = []

    def procedure(payload, rows, budget, **kwargs):
        called.append(len(rows))
        return replay_research_program(payload, rows, budget,
            execution_scope='retained_coordinates', **kwargs)

    monkeypatch.setattr(conditional_coordinates, 'replay_conditional_coordinates', procedure)
    changed = deepcopy(arguments)
    changed['native_representation_plan']['retained_replay_scope'] = 'procedure'
    second = agent._execute('run', changed, lab)
    assert not second['cache_hit'] and called == [48]
    native = load_native_representation_result(second['native_representation_ref'], agent.workspace)
    assert native['retained_replay_scope'] == 'procedure'
    assert agent.costs['model_calls'] == agent.costs['observation_cost'] == 0


def test_unknown_replay_scope_is_rejected_before_scientific_work(tmp_path):
    agent = agent_with_records(tmp_path)
    before = deepcopy(agent.costs)
    with pytest.raises(ValueError, match='retained_replay_scope'):
        run(agent, {'id': 'invalid_scope', 'kind': 'program_revision'},
            retained_replay_scope='relabel_as_refit')
    assert agent.costs == before


def test_complete_local_candidate_reference_never_reads_the_surrounding_state_text(tmp_path, monkeypatch):
    from hegel_machine.cognition import research_episodes
    from hegel_machine.cognition.representation_inquiry import retained_native_research_programs

    parent, reference = saved_conditional(tmp_path)
    state = Path(reference['file'])
    original = json.loads(state.read_text())['candidate']
    state.write_text(json.dumps({'older_unrelated_payload': ['retained'] * 100,
        'bank': [original]}))
    addressed = {'file': str(state), 'field': 'bank[0]'}
    read_text = Path.read_text

    def read(path, *args, **kwargs):
        assert path != state, 'candidate lookup decoded the complete surrounding state'
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(research_episodes, 'load_source_values',
        lambda *a: pytest.fail('complete candidate reference entered the whole-file loader'))
    result = retained_native_research_programs([addressed], {'actual': parent}, tmp_path)
    assert result[0]['research_program'] == original['research_program']
    assert result[0]['source_ref'] == {'file': str(state), 'json_path': ['bank', 0]}
    assert result[0]['parent_theory_ids'] == ['actual']


def test_historical_local_field_reference_keeps_its_original_restore_semantics(tmp_path, monkeypatch):
    from hegel_machine.cognition import research_episodes
    from hegel_machine.cognition.representation_inquiry import retained_native_research_programs

    parent, reference = saved_conditional(tmp_path)
    state = Path(reference['file'])
    candidate = json.loads(state.read_text())['candidate']
    state.write_text(json.dumps({'payload': candidate,
        'candidate_ref': {'json_path': ['payload']}}))
    old_loader = research_episodes.load_source_values
    calls = []

    def restore(references, workspace):
        calls.append(deepcopy(references))
        return old_loader(references, workspace)

    monkeypatch.setattr(research_episodes, 'load_source_values', restore)
    result = retained_native_research_programs([reference], {'actual': parent}, tmp_path)
    assert result[0]['research_program'] == candidate['research_program']
    assert calls == [[reference]]

