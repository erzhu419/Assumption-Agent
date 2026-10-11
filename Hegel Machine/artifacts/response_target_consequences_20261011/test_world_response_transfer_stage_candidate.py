"""Archive screening preserves useful trials and resumes existing local research."""

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition import world_response_hypotheses as research
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration
from hegel_machine.cognition.world_reference_response import execute_reference_response
from hegel_machine.cognition.world_response_transfer import KNOWLEDGE_PATH
from scripts.run_response_matched_adoption import prepare_bundle
from test_world_reference_hypothesis import PATH, prepared, square
from test_world_response_controller import BoundedReferenceLab, _state
from test_world_response_hypotheses import fixture


def _route(tmp_path, monkeypatch, *, archive=True, prior=None, namespace=None):
    monkeypatch.chdir(tmp_path)
    if archive:
        Path(KNOWLEDGE_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(KNOWLEDGE_PATH).write_text('{}')
    state, estimators = fixture()
    state['expanded_features'] = ['measurement']
    state.update(deepcopy(prior or {}))
    if namespace is None:
        monkeypatch.setattr(recipe, 'ESTIMATORS', tuple(estimators))
        extend = recipe._extend_hypotheses
    else:
        namespace['ESTIMATORS'] = tuple(estimators)
        extend = namespace['_extend_hypotheses']
    return extend(state, {'limits': {'max_model_calls': 0},
                         'costs': {'model_calls': 0}}, 5, PATH)


def test_first_archive_screening_is_distinct_from_its_scientific_context(tmp_path, monkeypatch):
    result = _route(tmp_path, monkeypatch)

    assert result['arguments']['operator'] == 'reference_response'
    assert result['arguments']['stage'] == 'frozen_recall'
    assert 'stage' not in result['arguments']['context']
    assert result['arguments']['context']['number'] == 5


@pytest.mark.parametrize('prior', [
    {'fit_bindings': {'reference_fit_quadratic_5': [{'features': {'measurement': 5.}}]}},
    {'tests': [{'number': 5, 'status': 'refuted', 'family': 'quadratic'}]},
    {'response_research': {'status': 'no_compatible_response_hypotheses',
                           'frozen_transfer': {'status': 'refuted'}}},
    {'response_research': {'status': 'no_untried_supported_setting',
                           'frozen_transfer': {'status': 'exhausted'}}},
    {'response_research': {'status': 'confirmation_trial',
                           'frozen_transfer': {'status': 'selected'}}},
])
def test_existing_local_research_keeps_full_response_entry(tmp_path, monkeypatch, prior):
    result = _route(tmp_path, monkeypatch, prior=prior)

    assert result['arguments']['operator'] == 'reference_response'
    assert 'stage' not in result['arguments']


def test_no_archive_keeps_existing_response_entry(tmp_path, monkeypatch):
    result = _route(tmp_path, monkeypatch, archive=False)

    assert result['arguments']['operator'] == 'reference_response'
    assert 'stage' not in result['arguments']


def test_exhausted_local_fits_do_not_mistake_initial_screening_for_full_research(tmp_path, monkeypatch):
    screening = _route(tmp_path, monkeypatch)
    prior = {
        'response_context': screening['arguments']['context'],
        'fit_bindings': {'reference_fit_quadratic_5': []},
        'response_research': {
            'status': 'response_evidence_needed', 'selection': None,
            'frozen_transfer': {'status': 'refuted', 'research_stage': 'initial_recall'}},
    }

    following = _route(tmp_path, monkeypatch, prior=prior)

    assert following['arguments']['operator'] == 'reference_response'
    assert following['arguments']['context'] == screening['arguments']['context']
    assert 'stage' not in following['arguments']


def _public_initial_state(tmp_path, *, intercept=100., archived_gain=10., target_densities=None):
    declaration = square('density', 'density_square')
    source = _retain_declaration(Path(recipe.__file__).read_text(), declaration)
    agent, _ = prepared(tmp_path, source=source)
    lab = BoundedReferenceLab(intercept=intercept)
    agent.lab = lab
    if target_densities:
        lab.density.update(target_densities)
        for number, density in target_densities.items():
            lab.required[number] = intercept + 10. * density ** 2
    # Acquire all inputs through the real interface, before any point fit or
    # target installation. No source observations are installed in this world.
    for number in range(1, 6):
        agent._execute('query', {'actions': [
            {'action': 'TALK', 'arg1': 100 + number}, {'chosen_dialog_option_int': 1},
            {'action': 'USE', 'arg1': 91, 'arg2': 200 + number},
            {'action': 'USE', 'arg1': 92, 'arg2': 200 + number}]}, lab)
    x = Program('feature', name='density')
    program = Program('add', children=(Program('const', value=100.),
        Program('mul', children=(Program('const', value=archived_gain),
            Program('mul', children=(x, x))))))
    knowledge = {'source_id': 'old_square', 'measurements': {'density': None},
        'models': [{'program': program.to_dict(), 'lower': -.5, 'upper': .5,
                    'family': 'density_square', 'source_question': 'old_response'}]}
    (agent.workspace / KNOWLEDGE_PATH).write_text(json.dumps(knowledge))
    return agent, lab, knowledge


def _forbid_new_response_members(*args, **kwargs):
    pytest.fail('Initial recall must not enter expensive new response-member research')


def test_reference_refutation_returns_parent_to_real_local_fit(tmp_path, monkeypatch):
    agent, lab, knowledge = _public_initial_state(tmp_path, intercept=160.)
    observations = deepcopy(agent.observations)
    monkeypatch.setattr(research, '_propose_response_members', _forbid_new_response_members)

    first = agent._run_procedure(lab, PATH, max_steps=1)

    arguments = first['steps'][0]['proposal']['arguments']
    assert arguments['stage'] == 'frozen_recall'
    state = _state(agent)
    transfer = state['response_research']['frozen_transfer']
    assert transfer['status'] == 'refuted' and transfer['retained_count'] == 0
    assert transfer['research_stage'] == 'initial_recall'
    assert not state.get('fit_bindings') and not state.get('tests')
    assert state['response_context'] == arguments['context']
    assert agent.observations == observations
    assert all(row['crystal'] <= 3 for row in state['response_settings'])
    archive_cost = agent.costs['research_compute_units']
    assert archive_cost == state['response_research']['compute_cost'] > 0

    following = agent._run_procedure(lab, PATH, max_steps=1)

    proposal = following['steps'][0]['proposal']
    assert proposal['arguments']['operator'] == 'research_update'
    assert _state(agent)['fit_bindings']
    assert len(_state(agent)['response_research_history']) == 1
    assert json.loads((agent.workspace / KNOWLEDGE_PATH).read_text()) == knowledge
    assert agent.costs['model_calls'] == 0


def test_usable_initial_recall_reaches_same_physical_trial_without_refitting(tmp_path, monkeypatch):
    agent, lab, knowledge = _public_initial_state(tmp_path)
    monkeypatch.setattr(research, '_propose_response_members', _forbid_new_response_members)

    first = agent._run_procedure(lab, PATH, max_steps=1)

    before = _state(agent)
    response = before['response_research']
    assert response['frozen_transfer']['status'] == 'selected'
    hypothesis, = response['hypotheses']
    assert hypothesis['program'] == knowledge['models'][0]['program']
    assert (hypothesis['lower'], hypothesis['upper']) == (-.5, .5)
    experiment = response['selection']['selected_experiment']
    assert (experiment['number'], experiment['prediction']) == (4, 260.)
    assert before['response_context'] == first['steps'][0]['proposal']['arguments']['context']
    initial_cost = agent.costs['research_compute_units']

    following = agent._run_procedure(lab, PATH, max_steps=3)

    assert [step['proposal']['actions'][0]['action'] for step in following['steps']] == [
        'PICKUP', 'PUT', 'TALK']
    state = _state(agent)
    assert state['test']['prediction'] == experiment['prediction']
    assert state['test']['response_question'] == response['question_id']
    assert state['test']['response_forecasts'] == experiment['predictions']
    assert not state.get('fit_bindings')
    assert len(state['response_research_history']) == 1
    assert state['response_context'] == before['response_context']
    assert agent.costs['research_compute_units'] == initial_cost


def test_explicit_initial_recall_requires_an_archive(tmp_path):
    agent, lab, _ = _public_initial_state(tmp_path)
    # Initialize the real public role table without first running a researcher.
    state = {'completed': [], 'rejected': {}, 'target_numbers': [4, 5],
             'reference_numbers': [1, 2, 3]}
    (agent.workspace / recipe.STATE).write_text(json.dumps(state))
    (agent.workspace / KNOWLEDGE_PATH).unlink()

    with pytest.raises(ValueError, match='archive'):
        execute_reference_response(agent, lab, {'path': PATH, 'stage': 'frozen_recall'})


def test_bundle_refresh_installs_initial_recall_route_in_embedded_controller(tmp_path, monkeypatch):
    source = Path(recipe.__file__).read_text()
    old = next(node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == '_extend_hypotheses')
    lines = source.splitlines(keepends=True)
    lines[old.lineno - 1:old.end_lineno] = [
        "def _extend_hypotheses(state, memory, number, path):\n"
        "    return {'stop': 'Archived routing predates initial recall'}\n"]
    bundle = {'files': {PATH: ''.join(lines)}}
    retained = deepcopy(bundle)

    refreshed = prepare_bundle(bundle)
    namespace = {}
    exec(compile(refreshed['files'][PATH], PATH, 'exec'), namespace)
    proposal = _route(tmp_path, monkeypatch, namespace=namespace)

    assert bundle == retained
    assert proposal['arguments']['stage'] == 'frozen_recall'
    assert 'stage' not in proposal['arguments']['context']


def _late_refutation_state():
    return {
        'response_research': {
            'question_id': 'reference_response_5_2',
            'status': 'confirmation_trial',
            'frozen_transfer': {'status': 'selected'},
            'hypotheses': [{'id': 'old_a'}, {'id': 'old_b'}]},
        'tests': [{'number': 5, 'status': 'refuted', 'family': 'old_square',
                   'response_question': 'reference_response_5_2'}],
        'response_feedback': {'refuted_hypotheses': ['old_a', 'old_b'],
                              'remaining_hypotheses': []},
    }


def test_actual_late_refutation_rechecks_archive_before_pending_local_fits(tmp_path, monkeypatch):
    prior = _late_refutation_state()
    result = _route(tmp_path, monkeypatch, prior=prior)

    assert result['arguments']['operator'] == 'reference_response'
    assert result['arguments']['stage'] == 'frozen_recall'
    assert 'stage' not in result['arguments']['context']


@pytest.mark.parametrize('change', ['earlier_question', 'exhausted'])
def test_unrelated_or_already_exhausted_feedback_preserves_full_response_route(tmp_path, monkeypatch, change):
    prior = _late_refutation_state()
    if change == 'earlier_question':
        prior['tests'][-1]['response_question'] = 'reference_response_4_1'
    else:
        prior['response_research']['frozen_transfer']['status'] = 'exhausted'

    result = _route(tmp_path, monkeypatch, prior=prior)

    assert result['arguments']['operator'] == 'reference_response'
    assert 'stage' not in result['arguments']


@pytest.mark.parametrize('transfer_status, stage', [
    ('refuted', 'withdrawal_recall'), ('exhausted', 'continuation_recall')])
def test_exhausted_local_fit_after_archive_screening_still_enters_full_research(
        tmp_path, monkeypatch, transfer_status, stage):
    screening = _route(tmp_path, monkeypatch, prior=_late_refutation_state())
    prior = {
        'response_context': screening['arguments']['context'],
        'fit_bindings': {'reference_fit_quadratic_5': []},
        'response_research': {
            'status': 'response_evidence_needed', 'selection': None,
            'frozen_transfer': {'status': transfer_status, 'research_stage': stage}},
    }

    following = _route(tmp_path, monkeypatch, prior=prior)

    assert following['arguments']['operator'] == 'reference_response'
    assert following['arguments']['context'] == screening['arguments']['context']
    assert 'stage' not in following['arguments']


def test_supported_transfer_then_target_counterexample_resumes_real_fitting(tmp_path, monkeypatch):
    # One static quadratic law applies to references and both targets. The old
    # coefficient error is compatible near the references but fails far away.
    # These are environment parameters; the agent sees only public instruments.
    agent, lab, knowledge = _public_initial_state(tmp_path, archived_gain=10.04,
        target_densities={4: 9.6 ** .5, 5: 108.59 ** .5})
    monkeypatch.setattr(research, '_propose_response_members', _forbid_new_response_members)
    previous = None
    for _ in range(80):
        outcome = agent._run_procedure(lab, PATH, max_steps=1)
        state = _state(agent)
        response = state.get('response_research', {})
        transfer = response.get('frozen_transfer', {})
        if transfer.get('research_stage') == 'withdrawal_recall':
            break
        previous = deepcopy(state)
    else:
        pytest.fail('No target-driven withdrawal reached through the physical controller')

    assert lab.active(4) and not lab.active(5)
    assert state['completed'] == [4]
    assert [(test['number'], test['status']) for test in state['tests']] == [
        (4, 'supported'), (5, 'refuted')]
    assert previous['response_research']['frozen_transfer']['status'] == 'selected'
    assert transfer['status'] == 'refuted' and transfer['retained_count'] == 0
    assert not state.get('fit_bindings')
    assert state['response_feedback']['remaining_hypotheses'] == []
    assert outcome['steps'][0]['proposal']['arguments']['stage'] == 'frozen_recall'
    assert all('family_research' not in item for item in state['response_research_history'])
    actions = deepcopy(lab.actions)
    observations = deepcopy(agent.observations)

    following = agent._run_procedure(lab, PATH, max_steps=1)

    assert following['steps'][0]['proposal']['arguments']['operator'] == 'research_update'
    assert _state(agent)['fit_bindings']
    assert lab.actions == actions and agent.observations == observations
    assert lab.active(4) and _state(agent)['completed'] == [4]
    assert json.loads((agent.workspace / KNOWLEDGE_PATH).read_text()) == knowledge
    assert agent.costs['model_calls'] == 0


def test_partial_archive_keeps_its_trial_then_exhaustion_resumes_real_local_fit(tmp_path, monkeypatch):
    # Both laws fit the acquired references. The old wider band incorrectly
    # supports C5=1190. Both support C4=196, which should be confirmed first.
    # The narrow survivor's C5 interval contains no attainable integer setting.
    agent, lab, knowledge = _public_initial_state(tmp_path, archived_gain=10.04,
        target_densities={4: 9.6 ** .5, 5: 108.53 ** .5})
    survivor = deepcopy(knowledge['models'][0])
    survivor['program']['children'][1]['children'][0]['value'] = 10.01
    survivor.update(lower=-.1, upper=.1)
    knowledge['models'].append(survivor)
    (agent.workspace / KNOWLEDGE_PATH).write_text(json.dumps(knowledge))
    monkeypatch.setattr(research, '_propose_response_members', _forbid_new_response_members)
    partial = None

    for _ in range(40):
        outcome = agent._run_procedure(lab, PATH, max_steps=1)
        state = _state(agent)
        response = state.get('response_research', {})
        transfer = response.get('frozen_transfer', {})
        if transfer['status'] == 'selected' and partial is None:
            partial = deepcopy(state)
        if transfer['status'] == 'exhausted':
            break
    else:
        pytest.fail('Compatible archive never reached its physically exhausted continuation')

    # Confirm the shared target before discriminating the two original models.
    # No fitted replacement displaces either before the physical counterexample.
    assert partial is not None
    selected = partial['response_research']
    assert selected['frozen_transfer']['retained_count'] == 2
    assert selected['frozen_transfer']['rejected_count'] == 0
    assert any(h['program'] == survivor['program'] and (h['lower'], h['upper']) == (-.1, .1)
               for h in selected['hypotheses'])
    experiment = selected['selection']['selected_experiment']
    assert (experiment['number'], experiment['prediction']) == (4, 196.)
    assert not partial.get('fit_bindings')
    assert [(test['number'], test['status']) for test in state['tests']] == [
        (4, 'supported'), (5, 'refuted')]
    assert state['tests'][0]['response_question'] == selected['question_id']
    assert state['completed'] == [4] and lab.active(4) and not lab.active(5)
    assert transfer['retained_count'] == 1 and transfer['rejected_count'] == 1
    assert response['selection'] is None
    assert response['hypotheses'][0]['program'] == survivor['program']
    assert outcome['steps'][0]['proposal']['arguments']['stage'] == 'frozen_recall'
    assert not state.get('fit_bindings')
    actions, observations = deepcopy(lab.actions), deepcopy(agent.observations)

    following = agent._run_procedure(lab, PATH, max_steps=1)

    assert following['steps'][0]['proposal']['arguments']['operator'] == 'research_update'
    assert _state(agent)['fit_bindings']
    assert _state(agent)['response_research']['hypotheses'] == response['hypotheses']
    assert lab.actions == actions and agent.observations == observations
    assert lab.active(4) and _state(agent)['completed'] == [4]
    assert json.loads((agent.workspace / KNOWLEDGE_PATH).read_text()) == knowledge
    assert agent.costs['model_calls'] == 0
