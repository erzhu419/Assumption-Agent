"""Local numerical evidence constrains returns without supplying scientific laws."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.inquiry_revision import analyze_paid_evidence, analyze_revision
from test_inquiry_revision import PublicLab, action, make_agent, revision, write


class EvenResponseLab(PublicLab):
    def execute(self, request):
        result = super().execute(request)
        for row in result['rows']:
            row['target'] = row['features']['u'] ** 2
        return result


def readout_source(answer):
    return f'def readout(result,spec,memory): return {answer!r}\n'


def relation(identifier, coefficients, **extra):
    return {'id': identifier, 'terms': [{'row': row, 'coefficient': coefficient}
                                      for row, coefficient in coefficients],
            'relation': 'eq', 'expected': 0., 'atol': 1e-9,
            'scope': {'protocol': 'matched reset pulses'}, **extra}


def test_conflicting_check_overrides_prose_but_keeps_independent_supported_claim(tmp_path):
    agent, lab = make_agent(tmp_path, 'hegel'), EvenResponseLab()
    agent.questions.append({'id': 'parent', 'question': 'Explain the response', 'status': 'open'})
    write(agent, 'model.py', 'def predict(row): return 0.\n', lab)
    authored = {'status': 'supported', 'answer': 'Both relations hold', 'evidence': [0, 1, 2],
        'checks': [relation('equal_magnitude', [(1, 1), (2, -1)]),
                   relation('odd_response', [(0, -2), (1, 1), (2, 1)])]}
    write(agent, 'readout.py', readout_source(authored), lab)
    result = agent._execute('query', {'actions': [action(0), action(1), action(-1)],
        'question': {'id': 'local', 'question': 'Check local relations',
                     'parent_id': 'parent', 'depends_on': []},
        'revision': revision(['model.py'], readout='readout.py')}, lab)
    record = result['revision']
    assert record['authored_answer'] == authored
    assert record['evidence_status'] == 'refuted'
    assert [check['status'] for check in record['check_results']] == ['satisfied', 'violated']
    assert record['check_results'][1]['value'] == 2.
    assert record['check_results'][1]['witnesses'][-1] == {
        'row': 2, 'coefficient': 1., 'features': {'u': -1}, 'history': (), 'target': 1}
    assert result['question']['status'] == 'refuted'
    assert result['question']['authored_status'] == 'supported'
    returned = agent.questions[0]['child_answers'][-1]
    assert returned['status'] == 'refuted'
    assert returned['check_results'][0]['status'] == 'satisfied'
    # Full persistent state retains the check independently of later free notes.
    saved = json.loads((tmp_path / 'research_state.json').read_text())
    assert saved['revisions'][0]['check_results'][1]['residual'] == 2.
    assert lab.paid == 3 and agent.costs['model_calls'] == 0


def test_paid_analysis_uses_existing_rows_and_carries_executed_parameter_answer(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    agent._execute('query', {'actions': [action(-1, 1)]}, lab)
    answer = {'status': 'supported', 'answer': 'Local contrast brackets an interval',
        'evidence': [0, 1], 'parameter_region': {'lower': -1., 'upper': 1.,
                                              'lower_closed': False, 'upper_closed': False},
        'scope': {'background': 'tested controls'},
        'checks': [relation('ordering', [(1, 1), (0, -1)], relation='gt')]}
    write(agent, 'readout.py', readout_source(answer), lab)
    question = {'id': 'parameter', 'status': 'open'}
    costs = deepcopy(agent.costs)
    record = analyze_paid_evidence(agent, 'readout.py', lab, question)
    assert record['kind'] == 'paid_evidence_analysis' and record['status'] == 'answered'
    assert record['returned_answer']['parameter_region'] == answer['parameter_region']
    assert record['returned_answer']['evidence_status'] == 'supported'
    assert question['evidence_status'] == 'supported'
    assert record['forecasts'] == [] and record['actions'] == []
    assert 'observations' not in record  # Existing archive is referenced, not copied.
    assert lab.paid == costs['observation_cost'] == agent.costs['observation_cost'] == 2
    assert agent.costs['compute_cost'] - costs['compute_cost'] == 2


@pytest.mark.parametrize('checks,expected', [
    (None, 'not_checked'),
    ([relation('uncited', [(1, 1)])], 'inconclusive'),
    ([relation('unknown_predicate', [(0, 1)], relation='approximately')], 'inconclusive'),
])
def test_unexecuted_relations_never_gain_numerical_support(tmp_path, checks, expected):
    agent, lab = make_agent(tmp_path), PublicLab()
    agent._execute('query', {'actions': [action(0)]}, lab)
    answer = {'status': 'supported', 'answer': 'A written interpretation', 'evidence': [0]}
    if checks is not None:
        answer['checks'] = checks
    write(agent, 'readout.py', readout_source(answer), lab)
    record = analyze_paid_evidence(agent, 'readout.py', lab)
    assert record['evidence_status'] == expected
    assert record['returned_answer']['status'] == ('supported' if checks is None else 'inconclusive')
    assert record['authored_answer']['status'] == 'supported'


def test_reanalysis_preserves_earlier_counterexample_and_does_not_repurchase(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    agent._execute('query', {'actions': [action(2)]}, lab)
    answer = {'status': 'supported', 'answer': 'Zero', 'evidence': [0],
              'checks': [relation('value', [(0, 1)])]}
    write(agent, 'readout.py', readout_source(answer), lab)
    first = analyze_paid_evidence(agent, 'readout.py', lab)
    assert first['evidence_status'] == 'refuted'
    answer['answer'] = 'Measured value'
    answer['checks'][0]['expected'] = 2.
    write(agent, 'readout.py', readout_source(answer), lab)
    updated = analyze_revision(agent, agent.revisions[0], lab)
    assert updated['evidence_status'] == 'supported'
    assert [attempt['evidence_status'] for attempt in updated['analysis_attempts']] == ['refuted', 'supported']
    assert updated['analysis_attempts'][0]['check_results'][0]['residual'] == 2.
    assert len(agent.revisions) == 1 and lab.paid == 1


def test_channel_check_uses_purchased_target_not_response(tmp_path):
    from test_common_archive_observable_research import MeasuredLab

    agent, lab = make_agent(tmp_path), MeasuredLab()
    agent._execute('query', {'actions': [action(0, channels=['meter'])]}, lab)
    assert agent.rows[0].target == 3.  # Public meter reading is 1, response is 3.
    answer = {'status': 'supported', 'answer': 'Measured channel value', 'evidence': [0],
              'checks': [relation('meter', [(0, 1)], expected=1.)]}
    write(agent, 'readout.py', readout_source(answer), lab)
    record = analyze_paid_evidence(agent, 'readout.py', lab, target='meter')
    assert record['evidence_status'] == 'supported'
    assert record['check_results'][0]['witnesses'][0]['target'] == 1.
    missing = analyze_paid_evidence(agent, 'readout.py', lab, target='unbought')
    assert missing['evidence_status'] == 'inconclusive'
    assert missing['returned_answer']['status'] == 'inconclusive'
