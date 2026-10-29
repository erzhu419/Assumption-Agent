"""Routing metadata preserves evidence semantics without copying Programs."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import question_routing as routing
from hegel_machine.cognition.language import Program
from test_frozen_counterexample_reuse import scene, declaration
from test_theory_consolidation import agent_scene


def old_projection(agent):
    return [{key: deepcopy(theory[key]) for key in routing._POOL_BASIS_FIELDS if key in theory}
        for theory in routing.resolved_theories(agent)]


def test_metadata_preserves_historical_scoped_failures_aliases_and_native_source_status(tmp_path, monkeypatch):
    agent = scene(); agent.workspace = tmp_path
    routing.register_theories(agent, [declaration('alias'),
        declaration('other_context', context_ref={'episode': 'different'}),
        declaration('failed', Program('const', value=7.))])
    owner = {'id': 'native', 'meaning_version': 1, 'native_condition_research': [{
        'candidates': [{'program': Program('const', value=3.).to_dict(), 'status': 'retained_candidate'}]}]}
    agent.questions.append(owner)
    for identifier, meaning in [('native_ready', 1), ('native_missing', 2)]:
        agent.theory_index[identifier] = declaration(identifier, source_ref={
            'kind': 'native_candidate', 'question_id': 'native', 'meaning_version': meaning,
            'round_index': 0, 'candidate_index': 0})
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('metadata executed a predictor'))
    before = deepcopy(agent.theory_index)
    expected = old_projection(agent)
    actual = routing._resolved_pool_basis(agent)
    assert actual == expected and [list(item) for item in actual] == [list(item) for item in expected]
    by_id = {item['id']: item for item in actual}
    assert by_id['alias']['compatible'] is False and by_id['failed']['compatible'] is True
    assert by_id['other_context']['compatible'] is True
    assert by_id['native_missing']['source_status'] == 'unavailable_retained_version'
    assert agent.theory_index == before


def test_metadata_uses_the_same_recorded_forecast_validity_and_file_versions(tmp_path, monkeypatch):
    agent = scene(); agent.workspace = tmp_path
    saved = tmp_path / 'columns.json'
    saved.write_text(json.dumps({'columns': [{'model': 'old.py', 'prediction_inputs': [
        {'features': {'x': .5}, 'history': []}], 'predictions': [1.]}]}))
    for identifier, filename in [('ready', 'columns.json'), ('unavailable', 'missing.json')]:
        entry = declaration(identifier); entry.pop('program')
        entry['forecast_ref'] = {'file': filename, 'json_path': ['columns', 0], 'model': 'old.py'}
        agent.theory_index[identifier] = entry
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('saved-column metadata predicted'))
    assert routing._resolved_pool_basis(agent) == old_projection(agent)
    saved.unlink()
    actual = routing._resolved_pool_basis(agent)
    assert actual == old_projection(agent)
    assert next(item for item in actual if item['id'] == 'ready')['source_status'] == 'unavailable_retained_forecast'


def test_basis_never_resolves_or_copies_world_program_payloads(tmp_path, monkeypatch):
    agent, laboratory = agent_scene(tmp_path)
    question = agent._open_question({'id': 'metadata', 'question': 'Reuse the retained routing identity.'})
    agent.theory_index['retained'] = declaration('retained')
    expected = old_projection(agent)

    class NeverCopyProgram(dict):
        def __deepcopy__(self, memo):
            pytest.fail('metadata copied a retained Program payload')

    agent.theory_index['retained']['program'] = NeverCopyProgram(agent.theory_index['retained']['program'])
    monkeypatch.setattr(routing, 'resolved_theories', lambda *args, **kwargs:
        pytest.fail('routing identity expanded the full scientific pool'))
    before = dict(agent.costs)
    basis = routing._basis(agent, question, {})
    assert basis['pool'] == expected
    basis['pool'][0]['compatible'] = False
    assert routing._basis(agent, question, {})['pool'] == expected
    assert agent.costs == before and laboratory.actions == []


def test_one_completion_pass_shares_metadata_but_next_pass_sees_version_and_source_changes(tmp_path, monkeypatch):
    agent, laboratory = agent_scene(tmp_path)
    saved = tmp_path / 'saved.json'
    saved.write_text(json.dumps({'model': 'old.py', 'prediction_inputs': [
        {'features': {'x': .5}, 'history': []}], 'predictions': [1.]}))
    agent.theory_index['forecast'] = {'id': 'forecast', 'version': 1,
        'forecast_ref': {'file': str(saved), 'json_path': [], 'model': 'old.py'}}
    questions = [agent._open_question({'id': name, 'question': 'Reuse retained public work.'})
        for name in ('first', 'second')]
    for question in questions:
        question['routing_execution'] = {'basis': routing._basis(agent, question, {}),
            'result': {'status': 'selected'}, 'allowance': 100}
    original = routing._resolved_pool_basis
    projections = []

    def counted(actual):
        projection = original(actual)
        projections.append(deepcopy(projection))
        return projection

    monkeypatch.setattr(routing, '_resolved_pool_basis', counted)
    before = dict(agent.costs)
    routing._refresh_completion(agent, {}, 100)
    assert len(projections) == 1 and all(question.get('completed_agenda_basis') for question in questions)
    agent.theory_index['forecast']['version'] = 2
    saved.unlink()
    routing._refresh_completion(agent, {}, 100)
    assert len(projections) == 2 and all('completed_agenda_basis' not in question for question in questions)
    assert projections[-1][0]['version'] == 2
    assert projections[-1][0]['source_status'] == 'unavailable_retained_forecast'
    assert agent.costs == before and laboratory.actions == []
