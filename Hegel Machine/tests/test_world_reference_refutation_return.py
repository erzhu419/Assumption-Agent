"""A physically excluded proposal returns to its author, before old fallback."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import world_llm_inquiry as inquiry
from test_world_reference_extension import PATH, QUESTION, prepared


RECORDS = json.loads((Path(__file__).parent / 'fixtures' /
                     'reference_hypothesis_return_20261010.json').read_text())


def run_return(tmp_path, monkeypatch, returned):
    steps = [('run', {'operator': 'reference_hypothesis', 'path': PATH,
                      'hypothesis': deepcopy(returned['hypothesis'])}),
             ('finish', {'lesson': 'The family is excluded; no replacement has been established.'})]
    agent, provider, _ = prepared(tmp_path, steps, calls=24, readings=(3, 4))
    execute = agent._execute

    def trial(action, params, lab):
        if action != 'run' or params.get('operator') != 'reference_hypothesis':
            return execute(action, params, lab)
        # The scheduling test uses archived scientific feedback. Acquisition is
        # real and charged, but this small laboratory is not the crystal task.
        execute('query', {'actions': [{'action': 'MEASURE'}]}, lab)
        return deepcopy(returned)

    monkeypatch.setattr(agent, '_execute', trial)
    result = inquiry.execute_llm_inquiry(agent, agent.lab, {'question': {
        'id': QUESTION, 'parent_id': 'task', 'question': 'Revise the current explanation',
        'parent_program_path': PATH}, 'max_model_calls': 24})
    return agent, provider, result


@pytest.mark.parametrize('record', RECORDS, ids=['actual_interaction_exclusion', 'actual_cubic_return'])
def test_archived_verdict_selects_author_continuation_or_parent_return(tmp_path, monkeypatch, record):
    agent, provider, result = run_return(tmp_path, monkeypatch, record['feedback'])
    assert len(provider.requests) == record['expected_model_calls']
    assert all(request['llm_inquiry']['question_id'] == QUESTION for request in provider.requests)
    assert result['status'] == 'inquiry_finished'
    assert ('finish' in [step['action'] for step in result['steps']]) == (record['expected_model_calls'] == 2)
    assert 'answer' not in result['question']
    assert agent.status == 'running'
    assert agent.costs['environment_steps'] == len(agent.observations) == 2


@pytest.mark.parametrize('change,expected_calls', [
    ('unknown', 1), ('excluded_other_family', 1), ('excluded_alias', 2),
])
def test_family_identity_and_unknown_are_not_point_refutations(tmp_path, monkeypatch, change, expected_calls):
    feedback = deepcopy(RECORDS[0]['feedback'])
    family = next(f for f in feedback['response_research']['family_research']['families']
                  if feedback['hypothesis']['id'] in f['source_ids'])
    if change == 'unknown':
        family['status'] = 'UNKNOWN'
    elif change == 'excluded_other_family':
        family['id'] = 'another_family'
        family['source_ids'] = ['another_family']
    else:
        family['id'] = 'canonical_basis'
        family['source_ids'].insert(0, 'earlier_alias')
    _, provider, _ = run_return(tmp_path, monkeypatch, feedback)
    assert len(provider.requests) == expected_calls
