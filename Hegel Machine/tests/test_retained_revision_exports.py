"""Typed retained export metadata is not a new scientific experiment."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_episodes import extract_episodes, retained_revision_sources
from test_retained_revision_fit_state import retained_state


def exported_state(tmp_path):
    agent = retained_state(tmp_path)
    agent.revisions = []
    agent.relation_research = {}
    question = agent.questions[0]
    question['native_condition_research'] = question['native_condition_research'][:1]
    target = {'role_signature': {'u': {'role': 'public_control', 'unit': 'V', 'time': 'current'}},
              'output_unit': 'V', 'context_ref': question['context_ref']}
    question['theory_target'] = deepcopy(target)
    question['revision_candidate_exports'] = []
    for i, identifier in enumerate(('revision_one', 'revision_two', 'unadopted')):
        agent.theory_index[identifier] = {**deepcopy(target),
            'program': Program('const', value=float(i + 1)).to_dict(), 'version': 1,
            'consolidation_ref': {'candidate_id': 'old_frozen_object', 'binding': i},
            'support': {'status': 'candidate', 'evidence': []}}
        question['revision_candidate_exports'].append({'model': f'revision_{i}.py',
            'theory_ref': {'id': identifier, 'version': 1},
            'program_source_ref': {'file': 'research_state.json',
                'json_path': ['theory_index', identifier, 'program'], 'version': 1},
            'transfer_ref': {'file': 'shared/theory_library.json', 'json_path': ['transfers', i]}})
    feedback = agent.events[0]['feedback']
    first = feedback['evaluations'][1]
    first['path'] = 'revision_0.py'
    first['retention_decision']['candidate']['model'] = first['path']
    second = deepcopy(first)
    second.update(path='revision_1.py', mse=.5)
    second['retention_decision'].update(
        incumbent={'source_model': first['path'], 'evaluated_rows': 3, 'mse': 1.},
        candidate={'model': second['path'], 'evaluated_rows': 3})
    feedback['evaluations'].append(second)
    return agent


def changes(agent):
    return [change for ep in extract_episodes(agent)['episodes']
            for change in ep['actual_changes'] if change['kind'] == 'retained_program_replaced']


def test_unique_typed_exports_bridge_native_and_revision_actual_adoptions(tmp_path, monkeypatch):
    agent = exported_state(tmp_path)
    original = deepcopy((agent.questions, agent.events, agent.theory_index, agent.costs))
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('metadata only'))
    first, second = changes(agent)
    assert first['before_ref']['json_path'] == [
        'questions', 0, 'native_condition_research', 0, 'candidates', 0, 'program']
    assert first['source_ref'] == second['before_ref'] == {
        'file': 'research_state.json', 'json_path': ['theory_index', 'revision_one', 'program'], 'version': 1}
    assert second['source_ref']['json_path'] == ['theory_index', 'revision_two', 'program']
    assert first['fit_state_refs']['after']['kind'] == 'frozen_revision_export'
    selected = retained_revision_sources(agent)
    assert set(selected) == {'revision_one', 'revision_two'}
    assert {item['source_id'] for item in selected.values()} == {'toy_physical_source'}
    assert all(item['episode_ref']['question_id'] == 'measured' for item in selected.values())
    assert len(selected['revision_two']['retention_refs']) == 1
    assert not first['scope']['independent_confirmation'] and not second['scope']['independent_confirmation']
    assert agent.theory_index['revision_one']['support']['status'] == 'candidate'
    assert (agent.questions, agent.events, agent.theory_index, agent.costs) == original


@pytest.mark.parametrize('reason', ['wrong_version', 'wrong_context', 'same_program', 'not_adopted', 'cached_selection'])
def test_missing_exact_endpoint_or_actual_decision_does_not_invent_parent(tmp_path, reason):
    agent = exported_state(tmp_path)
    feedback = agent.events[0]['feedback']
    feedback['evaluations'] = feedback['evaluations'][:2]
    if reason == 'wrong_version':
        agent.questions[0]['revision_candidate_exports'][0]['theory_ref']['version'] = 2
    elif reason == 'wrong_context':
        agent.theory_index['revision_one']['context_ref'] = {'source_id': 'other_source'}
    elif reason == 'same_program':
        agent.theory_index['revision_one']['program'] = deepcopy(
            agent.questions[0]['native_condition_research'][0]['candidates'][0]['program'])
    elif reason == 'not_adopted':
        feedback['evaluations'][1]['retention_decision']['status'] = 'incumbent_retained'
    else:
        feedback['evaluated_model'] = {'selection': feedback['evaluations'][1].pop('retention_decision')}
    assert changes(agent) == []
    assert retained_revision_sources(agent) == {}
