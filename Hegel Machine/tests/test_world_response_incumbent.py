"""Stalled authored research compares its retained controller on current evidence."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import world_reference_response as response
from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration
from test_world_bound_response import BINDING, _compare, _ready, _state
from test_world_premise_dispatch import AUTHOR, PREMISE, finish_provider, stalled
from test_world_reference_hypothesis import PATH


DENSITY = {'id': 'retained_density', 'basis': [{'op': 'feature', 'name': 'density'}]}


def retained_controller(tmp_path):
    agent, lab = _ready(tmp_path)
    source = agent.workspace / PATH
    agent._execute('write', {'path': PATH,
        'code': _retain_declaration(source.read_text(), DENSITY)}, lab)
    compared = agent._execute('run', {'operator': 'reference_response', 'path': PATH}, lab)
    assert compared['selection']['selected_experiment']
    return agent, lab, compared


def stale_controller(tmp_path):
    agent, lab, old = retained_controller(tmp_path)
    # Buy a target failure without using the computed binding. The later bound
    # comparison includes it, while the retained raw comparison is still old.
    trial = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'experiment': {'number': 4, 'setting': 500.}}, lab)
    assert trial['requested_setting_reached'] and trial['prediction_status'] == 'refuted'
    _, current = stalled(agent, lab)
    assert current['response_count'] > old['response_count']
    return agent, lab, old, current


def test_stalled_binding_refreshes_retained_controller_once_on_the_same_paid_evidence(tmp_path):
    agent, lab, old, bound = stale_controller(tmp_path)
    provider = finish_provider(agent)
    before = deepcopy(agent.costs), len(agent.observations), len(agent.questions)

    assert response.dispatch_response_premise(agent, lab) is False

    state = _state(agent)
    raw = state['response_research']
    assert raw['response_count'] == bound['response_count'] > old['response_count']
    selected = raw['selection']['selected_experiment']
    assert agent.research_advice['diagnosis'] == 'retained_response_experiment'
    assert agent.research_advice['next_action'] == {'action': 'run', 'arguments': {
        'operator': 'reference_response', 'path': PATH,
        'experiment': {'number': selected['number'], 'setting': selected['prediction']}}}
    assert agent.costs['research_compute_units'] > before[0]['research_compute_units']
    assert len(agent.observations) == before[1]
    assert agent.costs['model_calls'] == before[0]['model_calls']
    assert len(agent.questions) == before[2] + 1
    assert not provider.requests and PREMISE not in agent.llm_inquiries
    inputs = deepcopy(state['response_research_inputs'])
    paid, questions = deepcopy(agent.costs), deepcopy(agent.questions)

    assert response.dispatch_response_premise(agent, lab) is False
    assert agent.costs == paid and agent.questions == questions

    # A repeated read adds record indices, but no new physical response or
    # measurement. It must not purchase the identical raw comparison again.
    agent._execute('query', {'actions': [
        {'action': 'TALK', 'arg1': 101}, {'chosen_dialog_option_int': 1}]}, lab)
    after_read = deepcopy(agent.costs)
    assert response.dispatch_response_premise(agent, lab) is False
    assert agent.costs == after_read and agent.questions == questions
    assert _state(agent)['response_research_inputs'] == inputs
    assert not provider.requests


def test_current_counterexample_refutes_retained_family_before_any_reuse(tmp_path):
    agent, lab, old = retained_controller(tmp_path)
    stalled(agent, lab)
    provider = finish_provider(agent)
    assert response.dispatch_response_premise(agent, lab) is False
    assert agent.research_advice['diagnosis'] == 'retained_response_experiment'
    # C1 and C2 have equal density. C2's positive at 103 and this actual C1
    # negative cannot share any density-only deterministic response family.
    trial = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'experiment': {'number': 1, 'setting': 103.}}, lab)
    assert trial['requested_setting_reached'] and trial['prediction_status'] == 'refuted'
    _, bound = stalled(agent, lab)
    observations = len(agent.observations)

    assert response.dispatch_response_premise(agent, lab) is True

    raw = _state(agent)['response_research']
    assert raw['response_count'] == bound['response_count'] > old['response_count']
    assert {family['id']: family['status'] for family in raw['family_research']['families']} == {
        DENSITY['id']: 'UNSAT_VERIFIED'}
    assert raw['selection'] is None
    assert (agent.research_advice or {}).get('diagnosis') != 'retained_response_experiment'
    assert len(provider.requests) == 1 and PREMISE in agent.llm_inquiries
    assert len(agent.observations) == observations
    paid, questions = deepcopy(agent.costs), deepcopy(agent.questions)
    assert response.dispatch_response_premise(agent, lab) is False
    assert agent.costs == paid and agent.questions == questions


def test_retained_offer_does_not_override_explicit_authored_physical_experiment(tmp_path):
    agent, lab, _, bound = stale_controller(tmp_path)
    provider = finish_provider(agent)
    assert response.dispatch_response_premise(agent, lab) is False
    offered = agent.research_advice['next_action']['arguments']['experiment']
    requested = {'number': 1, 'setting': 103.}
    assert offered != requested
    raw = deepcopy(_state(agent)['response_research'])
    declaration = bound['revision_action']['arguments']['question']

    trial = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'experiment': requested, 'response_comparison': {
            'binding_path': BINDING, 'question': deepcopy(declaration)}}, lab)

    assert trial['selected_number'] == requested['number']
    assert trial['selected_setting'] == requested['setting']
    assert trial['requested_setting_reached'] and trial['prediction_status'] == 'refuted'
    assert trial['response_comparison']['binding_path'] == BINDING
    assert trial['response_comparison']['response_count'] > bound['response_count']
    assert agent.llm_inquiries[AUTHOR]['proposal']['comparison_result'] == trial['response_comparison']
    assert _state(agent)['response_research'] == raw
    assert not provider.requests


@pytest.mark.parametrize('change', ['basis', 'paid_response'])
def test_new_retained_basis_or_paid_response_updates_cached_comparison(tmp_path, change):
    agent, lab, _, _ = stale_controller(tmp_path)
    provider = finish_provider(agent)
    assert response.dispatch_response_premise(agent, lab) is False
    previous = deepcopy(_state(agent))
    if change == 'basis':
        source = agent.workspace / PATH
        agent._execute('write', {'path': PATH, 'code': _retain_declaration(source.read_text(),
            {'id': 'retained_gap', 'basis': [{'op': 'feature', 'name': 'gap'}]})}, lab)
    else:
        trial = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
            'experiment': {'number': 4, 'setting': 499.}}, lab)
        assert trial['requested_setting_reached']
        stalled(agent, lab)
    paid = deepcopy(agent.costs)
    observations = len(agent.observations)

    assert response.dispatch_response_premise(agent, lab) is False

    current = _state(agent)
    assert current['response_research_inputs'] != previous['response_research_inputs']
    assert agent.costs['research_compute_units'] > paid['research_compute_units']
    assert len(agent.observations) == observations and not provider.requests
    raw = current['response_research']
    if change == 'basis':
        assert 'retained_gap' in {family['id'] for family in raw['family_research']['families']}
        assert raw['response_count'] == previous['response_research']['response_count']
    else:
        assert raw['response_count'] > previous['response_research']['response_count']


def test_available_authored_experiment_keeps_retained_controller_from_taking_over(tmp_path):
    agent, lab, _, _ = stale_controller(tmp_path)
    available = _compare(agent, lab)
    assert available['experiment_action']
    agent.llm_inquiries[AUTHOR]['proposal']['comparison_result'] = deepcopy(available)
    provider = finish_provider(agent)
    costs, state = deepcopy(agent.costs), deepcopy(_state(agent))

    assert response.dispatch_response_premise(agent, lab) is False

    assert agent.costs == costs and _state(agent) == state
    assert not provider.requests and PREMISE not in agent.llm_inquiries


def test_revisiting_paid_positive_updates_reference_fit_inputs(tmp_path, monkeypatch):
    agent, lab, _ = retained_controller(tmp_path)
    narrow_activation = lab.active
    # This physical fixture permits two neighboring successful settings, as in
    # the existing wide-reference controller tests. Neither is an exact label.
    monkeypatch.setattr(lab, 'active', lambda number: (
        number in lab.installed and abs(lab.frequencies[number] - lab.required[number]) < 1.1
        if number == 1 else narrow_activation(number)))

    def move(choice):
        agent._execute('query', {'actions': [
            {'action': 'TALK', 'arg1': 101}, {'chosen_dialog_option_int': choice},
            {'chosen_dialog_option_int': 1}]}, lab)

    move(5)  # Acquire the second positive setting, 101.
    assert lab.active(1) and lab.frequencies[1] == 101.
    move(9)  # Revisit the original paid positive, 100.
    estimators, before, _ = response._reference_state(agent, lab, PATH)
    before_inputs = response._reference_response_inputs(before, estimators)
    old_reference = next(row for row in before['research_table']['references'] if row['crystal'] == 1)
    assert old_reference['frequency']['value'] == 100.

    move(5)  # This adds no new unique response, but changes the fitter's label.
    estimators, after, _ = response._reference_state(agent, lab, PATH)
    after_inputs = response._reference_response_inputs(after, estimators)
    new_reference = next(row for row in after['research_table']['references'] if row['crystal'] == 1)
    assert new_reference['frequency']['value'] == 101.
    assert after_inputs['responses'] == before_inputs['responses']
    assert after_inputs['measurements'] == before_inputs['measurements']
    assert after_inputs != before_inputs
