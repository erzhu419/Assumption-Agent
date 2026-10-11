"""Premise authors receive actual paid failures and the retained conditional model."""

from copy import deepcopy

from hegel_machine.cognition.world_reference_response import (
    response_premise_design_action,
    response_premise_proposal_inputs,
)
from test_world_bound_response import BINDING, _ready
from test_world_premise_dispatch import stalled
from test_world_reference_hypothesis import PATH


def offered_path(comparison):
    return response_premise_design_action(comparison)['arguments']['path']


def test_premise_inputs_include_paid_negative_responses_features_and_retained_model(tmp_path):
    agent, lab = _ready(tmp_path)
    trial = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'experiment': {'number': 1, 'setting': 102.}}, lab)
    assert trial['requested_setting_reached'] and trial['prediction_status'] == 'refuted'
    author, compared = stalled(agent, lab)
    before = deepcopy(author), deepcopy(agent.costs), deepcopy(lab.actions)
    sources = {path: (tmp_path / path).read_text() for path in (PATH, BINDING)}
    state_path = tmp_path / 'shared/reference_recalibration.json'
    state_before = state_path.read_text()

    packet = response_premise_proposal_inputs(agent, offered_path(compared))

    outcomes = {(row['crystal'], row['frequency'], row['activated']) for row in packet['responses']}
    assert (1, 100., True) in outcomes and (1, 102., False) in outcomes
    assert packet['responses_not_in_comparison'] == []
    assert all(row['evidence'] and set(row['evidence']) <= set(range(len(agent.observations)))
               for row in packet['responses'])
    assert packet['bound_rows'] == compared['bound_rows']
    assert any('log_density' in row['features'] for row in packet['bound_rows'])
    assert packet['estimators'] == compared['estimators']
    assert packet['assumptions'] == compared['assumptions']
    assert packet['paid_family_checks'] == compared['paid_family_checks']
    assert packet['comparison_evidence'] == compared['evidence']
    assert packet['unresolved_consequences']['families'] == compared['family_research']['families']
    assert packet['unresolved_consequences']['research_obligation'] == (
        compared['family_research']['research_obligation'])
    assert packet['measurements'] and packet['control_grids']
    assert packet['binding_path'] == BINDING and packet['controller_path'] == PATH
    assert {path: (tmp_path / path).read_text() for path in sources} == sources
    assert state_path.read_text() == state_before
    assert (author, agent.costs, lab.actions) == before


def test_unrelated_proposal_does_not_receive_another_authors_premise_inputs(tmp_path):
    agent, lab = _ready(tmp_path)
    stalled(agent, lab)
    before = deepcopy(agent.costs), deepcopy(lab.actions)

    assert response_premise_proposal_inputs(agent, 'shared/unrelated_proposal.py') is None

    assert (agent.costs, lab.actions) == before


def test_fresh_physical_response_is_exposed_without_rewriting_old_comparison_or_state(tmp_path):
    agent, lab = _ready(tmp_path)
    author, compared = stalled(agent, lab)
    old_comparison = deepcopy(author['proposal']['comparison_result'])
    trial = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'experiment': {'number': 1, 'setting': 102.}}, lab)
    assert trial['requested_setting_reached'] and trial['prediction_status'] == 'refuted'
    state_path = tmp_path / 'shared/reference_recalibration.json'
    before = (state_path.read_text(), deepcopy(agent.costs), deepcopy(lab.actions),
              deepcopy(agent.observations))

    packet = response_premise_proposal_inputs(agent, offered_path(compared))

    unexamined = packet['responses_not_in_comparison']
    negative = next(row for row in unexamined if
                    (row['crystal'], row['frequency'], row['activated']) == (1, 102., False))
    assert set(negative['evidence']) & set(trial['new_observation_indices'])
    assert not set(negative['evidence']) & set(old_comparison['evidence'])
    assert negative in packet['responses']
    assert packet['comparison_evidence'] == old_comparison['evidence']
    assert author['proposal']['comparison_result'] == old_comparison
    assert (state_path.read_text(), agent.costs, lab.actions, agent.observations) == before
