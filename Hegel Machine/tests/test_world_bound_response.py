"""Computed representations reuse paid Boolean evidence and the actual controller."""

from copy import deepcopy
import json
import math

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from test_world_reference_hypothesis import PATH, prepared
from test_world_response_controller import BoundedReferenceLab


BINDING = "shared/computed_basis.py"
BIND_CODE = '''
import math, re

def bind(public_spec, observations, memory):
    rows = {}
    for record in observations:
        action = record['action']
        if action.get('action') != 'USE' or action.get('arg1') != 91:
            continue
        value = float(re.search(r'density is approximately ([\\d.]+)',
            record['observation']['lastActionMessage'])[1])
        number = action['arg2'] - 200
        rows[number] = {'crystal': number,
            'features': {'log_density': math.log(value)},
            'evidence': [record['index']]}
    return {'rows': list(rows.values()), 'estimators': [
        {'id': 'computed_log', 'basis': [{'op': 'feature', 'name': 'log_density'}]}]}
'''


def _state(agent):
    return json.loads((agent.workspace / recipe.STATE).read_text())


def _ready(tmp_path):
    agent, _ = prepared(tmp_path)
    lab = BoundedReferenceLab()
    # Identical density does not imply identical response. Before acquiring a
    # cross-crystal negative, the shared-band hypothesis can explain both
    # successes despite its poor least-squares fit to their successful labels.
    lab.density = {1: 1., 2: 1., 3: math.exp(2), 4: math.e, 5: math.exp(1.5)}
    lab.required = {1: 100., 2: 103., 3: 110., 4: 160., 5: 180.}
    lab.frequencies.update({n: lab.required[n] for n in (1, 2, 3)})
    agent.lab = lab
    for number in range(1, 6):
        agent._execute('query', {'actions': [
            {'action': 'TALK', 'arg1': 100 + number}, {'chosen_dialog_option_int': 1},
            {'action': 'USE', 'arg1': 91, 'arg2': 200 + number},
            {'action': 'USE', 'arg1': 92, 'arg2': 200 + number}]}, lab)
    _, _, _, readings, references = recipe._scan(agent.observations,
        lab.public_observation(), recipe.ESTIMATORS)
    table = {'references': [], 'targets': []}
    for number in range(1, 6):
        row = {'crystal': number, 'measurements': {
            name: readings[name + '_' + str(number)] for name in ('density', 'gap')}}
        if number <= 3:
            row['frequency'] = references[str(number)]
            table['references'].append(row)
        else:
            row['response_settings'] = []
            table['targets'].append(row)
    state = {'completed': [], 'rejected': {}, 'reference_numbers': [1, 2, 3],
             'target_numbers': [4, 5], 'research_table': table}
    recipe.refresh_response_evidence(state, agent.observations, lab.public_observation(),
                                     recipe.ESTIMATORS)
    (agent.workspace / recipe.STATE).write_text(json.dumps(state))
    agent._execute('write', {'path': BINDING, 'code': BIND_CODE}, lab)
    # Retain a real parent's comparison so a bound inquiry cannot borrow or
    # replace its family state merely because it shares the response operator.
    agent._execute('run', {'operator': 'reference_response', 'path': PATH}, lab)
    return agent, lab


def _compare(agent, lab):
    return agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'binding_path': BINDING, 'question': {
            'id': 'computed', 'question': 'Can this computed representation explain the paid responses?',
            'parent_id': 'task'}}, lab)


def _families(result):
    return {family['id']: family['status'] for family in result['family_research']['families']}


def test_computed_feature_boolean_comparison_executes_selected_trial_and_refreshes(tmp_path):
    agent, lab = _ready(tmp_path)
    before, source = deepcopy(_state(agent)), (agent.workspace / PATH).read_text()
    paid = len(agent.observations)
    result = _compare(agent, lab)

    assert result['binding_path'] == BINDING
    values = {row['crystal']: row['features']['log_density'] for row in result['bound_rows']}
    assert values == pytest.approx({1: 0., 2: 0., 3: 2., 4: 1., 5: 1.5})
    assert any(candidate.get('fit_mse', 0) >= 1.5 for candidate in result['candidates'])
    assert _families(result) == {'computed_log': 'SAT_WITNESS'}
    question = next(q for q in agent.questions if q['id'] == 'computed')
    assert question['status'] == 'inconclusive'
    assert len(agent.observations) == paid
    assert result['new_observation_indices'] == []
    assert _state(agent)['research_table'] == before['research_table']
    assert _state(agent)['response_research'] == before['response_research']
    assert (agent.workspace / PATH).read_text() == source
    assert all(index in result['evidence'] for row in result['bound_rows'] for index in row['evidence'])

    action = result['experiment_action']
    assert action and action['action'] == 'run'
    assert 'binding_path' not in action['arguments']
    trial = agent._execute(action['action'], action['arguments'], lab)
    assert trial['requested_setting_reached']
    assert trial['prediction_status'] == 'refuted'
    assert any(not row['activated'] for row in trial['observed_responses'])
    measured = len(agent.observations)
    after_trial = _state(agent)
    revised = _compare(agent, lab)
    assert revised['response_count'] > result['response_count']
    assert set(trial['new_observation_indices']) & set(revised['evidence'])
    assert len(agent.observations) == measured
    assert _state(agent)['research_table'] == after_trial['research_table']
    assert _state(agent)['response_research'] == after_trial['response_research']
    saved = _state(agent)['bound_response_research']['computed']['result']
    assert saved['response_count'] == revised['response_count']
    assert len([q for q in agent.questions if q['id'] == 'computed']) == 1
    assert agent.costs['model_calls'] == 0


def test_actual_cross_crystal_negative_excludes_computed_shared_family(tmp_path):
    agent, lab = _ready(tmp_path)
    first = _compare(agent, lab)
    assert _families(first) == {'computed_log': 'SAT_WITNESS'}
    # 103 was publicly observed to activate crystal 2, whose computed feature
    # equals crystal 1's. The controller now buys the missing contrast.
    trial = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'experiment': {'number': 1, 'setting': 103.}}, lab)
    assert trial['prediction_status'] == 'refuted'
    assert trial['setting_recoveries'][-1]['restored']
    assert lab.active(1)
    paid = len(agent.observations)
    compute_before = agent.costs['research_compute_units']
    revised = _compare(agent, lab)
    assert _families(revised) == {'computed_log': 'UNSAT_VERIFIED'}
    checks = revised['paid_family_checks']
    assert checks['computed_log']['status'] == 'UNSAT_VERIFIED'
    assert checks['computed_log']['branch_limit'] is None
    assert checks['computed_log']['compute_limit'] == 50000
    assert revised['paid_comparison_cost'] == sum(check['compute_cost'] for check in checks.values())
    assert revised['paid_comparison_cost'] > 0
    # This fresh paid-history exclusion already answers all strengthened family
    # queries; the consequence stage must not solve or charge it a second time.
    assert revised['family_research']['query_count'] == 0
    assert revised['family_research']['compute_cost'] == 0
    assert revised['compute_cost'] == revised['paid_comparison_cost']
    assert agent.costs['research_compute_units'] - compute_before == revised['compute_cost']
    assert len(agent.observations) == paid
    assert revised['experiment_action'] is None
    question = next(q for q in agent.questions if q['id'] == 'computed')
    assert question['status'] == 'refuted'
    assert set(trial['new_observation_indices']) & set(question['evidence'])
    assert 'log_density' not in _state(agent)['research_table']['references'][0]['measurements']


def test_revised_binding_reopens_its_own_comparison_without_replacing_parent(tmp_path):
    agent, lab = _ready(tmp_path)
    first = _compare(agent, lab)
    parent = deepcopy(_state(agent)['response_research'])
    revised_code = BIND_CODE.replace('math.log(value)', 'math.sqrt(value)').replace(
        'log_density', 'root_density').replace('computed_log', 'computed_root')
    agent._execute('write', {'path': BINDING, 'code': revised_code}, lab)
    revised = _compare(agent, lab)
    assert revised['question_id'] == first['question_id']
    assert revised['family_research']['epoch'] == first['family_research']['epoch'] + 1
    assert revised['family_research']['nonlocal_progress'] == 0
    assert set(_families(revised)) == {'computed_root'}
    assert _state(agent)['response_research'] == parent
    assert len([q for q in agent.questions if q['id'] == 'computed']) == 1


@pytest.mark.parametrize('change, message', [
    ("rows.pop(5)", 'every current'),
    ("rows[1]['evidence'] = [99999]", 'paid observation'),
])
def test_incomplete_or_uncited_computed_rows_do_not_change_parent(tmp_path, change, message):
    agent, lab = _ready(tmp_path)
    malformed = BIND_CODE.replace("    return {'rows':", "    " + change + "\n    return {'rows':")
    agent._execute('write', {'path': BINDING, 'code': malformed}, lab)
    before, paid = _state(agent), len(agent.observations)
    with pytest.raises(ValueError, match=message):
        _compare(agent, lab)
    assert _state(agent) == before
    assert len(agent.observations) == paid


def test_unavailable_binding_does_not_run_parent_controller(tmp_path):
    agent, lab = _ready(tmp_path)
    (agent.workspace / BINDING).unlink()
    before, paid = _state(agent), len(agent.observations)
    with pytest.raises(ValueError, match='Program .* failed'):
        _compare(agent, lab)
    assert _state(agent) == before
    assert len(agent.observations) == paid


def test_parent_family_round_preserves_an_authored_child_with_the_next_round_name(tmp_path):
    agent, lab = _ready(tmp_path)
    trial = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'experiment': {'number': 1, 'setting': 103.}}, lab)
    assert trial['prediction_status'] == 'refuted'
    next_round = len(_state(agent)['response_research_history']) + 1
    child = agent._question({'id': f'reference_response_4_{next_round}',
        'question': 'Does crystal 1 activate at the tested 103 Hz?', 'parent_id': 'task'})
    agent._record_answer(child, {'status': 'refuted', 'answer': trial['answer'],
        'evidence': trial['new_observation_indices']})
    recorded = deepcopy(child)

    compared = agent._execute('run', {'operator': 'reference_response', 'path': PATH}, lab)

    assert compared['question_id'] != recorded['id']
    assert next(q for q in agent.questions if q['id'] == recorded['id']) == recorded
    assert _state(agent)['response_research_history'][-1]['question_id'] == compared['question_id']
