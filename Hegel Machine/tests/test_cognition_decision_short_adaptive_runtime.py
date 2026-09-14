"""A short visit buys useful followup without consuming its reporting budget."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.decision_frontier import _policy
from hegel_machine.cognition.decision_research import decision_quote, run_decision_research
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.workspace import ResearchWorkspace
from test_cognition_decision_frontier import _calibration
from test_cognition_decision_probing import ResponseApparatus, acquired, policies
from test_cognition_decision_research_runtime import Response, prepared


@pytest.mark.parametrize('compute_budget,queries,adopted', [(207, 2, False), (208, 4, True)])
def test_additional_pair_requires_its_comparison_and_reporting_cost(compute_budget, queries, adopted):
    session, original, workspace = prepared(0.)

    class ChangingResponse(Response):
        def observe(self, probe):
            self.target = [0., 1., 1., 1.][len(self.queries)]
            return super().observe(probe)

    environment = ChangingResponse(original.spec, 0.)
    config = replace(session.config, observation_budget=32, compute_budget=compute_budget)
    quote = decision_quote(workspace)
    result = run_decision_research(environment, config, KnowledgeLibrary(), research_state=workspace.to_dict())
    assert len(environment.queries) == queries
    comparisons = [action for action in result.actions if action.action == 'compare_decision_question']
    assert len(comparisons) == queries // 2
    assert all(action.compute_cost == 32 for action in comparisons)
    assert comparisons[-1].details['compute_cost'] == 16 * queries
    assert result.observation_cost == 8 * queries
    assert result.compute_cost == (208 if adopted else quote['compute_cost'])
    assert result.compute_cost <= config.compute_budget
    feedback = next(action.details for action in result.actions if action.action == 'continuation_feedback')
    assert feedback['execution_complete'] is adopted
    assert result.uncertainty_model == (_calibration() if adopted else workspace.uncertainty_model)
    if adopted:
        assert result.research_state['decision_frontier'] == {}
    else:
        pending = result.research_state['decision_frontier']['evidence_states'][0]
        assert pending['comparison']['trajectories'] == 2
        assert pending['candidate_model'] == _calibration()


def test_multiple_unresolved_calibrations_retain_their_own_paid_comparisons():
    session, original, workspace = prepared(0.)
    frontier = workspace.decision_frontier
    frontier['policies'].append(deepcopy(frontier['policies'][1]))
    frontier['policies'][-1]['model'] = _calibration(.6)
    frontier['comparison_compute_cost'] = frontier['compute_cost'] = 64

    class ChangingResponse(Response):
        def observe(self, probe):
            self.target = [0., 1., 1., 1.][len(self.queries)]
            return super().observe(probe)

    environment = ChangingResponse(original.spec, 0.)
    config = replace(session.config, observation_budget=16)
    first = run_decision_research(environment, config, KnowledgeLibrary(), research_state=workspace.to_dict())
    pending = first.research_state['decision_frontier']
    assert len(pending['evidence_states']) == 2 and len(pending['policies']) == 3
    assert [state['candidate_model'] for state in pending['evidence_states']] == [_calibration(), _calibration(.6)]
    assert pending['comparison_compute_cost'] == 64
    second = run_decision_research(environment, config, KnowledgeLibrary(),
        research_state=json.loads(json.dumps(first.research_state)))
    compared = [action for action in second.actions if action.action == 'compare_decision_question']
    assert len(compared) == 2 and all(action.compute_cost == 32 for action in compared)
    assert all(action.details['trajectories'] == 4 for action in compared)
    for prior, action in zip(pending['evidence_states'], compared):
        assert action.details['trajectory_comparisons'][:2] == prior['comparison']['trajectory_comparisons']
    assert second.uncertainty_model == _calibration()
    assert second.research_state['decision_frontier'] == {}


@pytest.mark.parametrize('feedback', [0., .5])
def test_short_visit_learns_real_history_contrasts_but_scores_only_representative_trials(feedback):
    class RecordingApparatus(ResponseApparatus):
        def __init__(self):
            super().__init__(feedback)
            self.queries, self.responses = [], []

        def observe(self, probe):
            measured = super().observe(probe)
            self.queries.append(probe)
            self.responses.append(measured)
            return measured

    environment = RecordingApparatus()
    initial, pairs = acquired(environment), policies()
    environment.queries.clear()
    environment.responses.clear()
    workspace = ResearchWorkspace(environment.spec.task_id, observations=initial, reference=list(initial),
        incumbent=pairs[0][0], bank=[pairs[1][0]], fallback=0., initial_loss=.9,
        uncertainty_model=pairs[0][1])
    # A saved structural question, before any prospective paired comparison.
    workspace.decision_frontier = {'policies': [_policy(*pair) for pair in pairs],
        'design': {'policy': 'unresolved_decision_comparison', 'alternatives': []},
        'probe_lengths': [8, 8], 'observation_cost': 16, 'comparison_compute_cost': 32,
        'compute_cost': 32, 'question': 'Does prepared history distinguish the two fixed responses?'}
    library = KnowledgeLibrary()
    result = run_decision_research(environment, ResearchConfig(seed=7, observation_budget=64, compute_budget=2000),
        library, research_state=workspace.to_dict())
    design = next(action for action in result.actions if action.action == 'plan_decision_probe_package')
    assert design.details['selected'] and 0 < design.compute_cost <= 500
    acquired_actions = [action for action in result.actions if action.action == 'observe']
    targeted_ids = {action.details['probe_id'] for action in acquired_actions if action.details['role'] == 'targeted'}
    representative_ids = {action.details['probe_id'] for action in acquired_actions if action.details['role'] == 'representative'}
    assert len(targeted_ids) == 2 and len(representative_ids) in (2, 4)
    new_reference = [row for probe, rows in zip(environment.queries, environment.responses)
                     if probe.probe_id in representative_ids for row in rows]
    saved = ResearchWorkspace.from_dict(result.research_state)
    assert saved.reference == initial + new_reference
    assert len(saved.observations) == len(initial) + result.observation_cost
    assert result.observation_cost == sum(probe.cost for probe in environment.queries)
    compared = [action for action in result.actions if action.action == 'compare_decision_question']
    assert compared[-1].details['rows'] == len(new_reference)
    returned = next(action.details for action in result.actions if action.action == 'continuation_feedback')
    assert returned['reference_rows'] == len(new_reference)
    learned = [action for action in result.actions if action.action == 'learn_experiment']
    assert learned and library.experiments.records
    assert all(action.compute_cost > 0 and action.details['experiment']['observed_loss_gap'] > 0 for action in learned)
    assert all(record['source_task'] == environment.spec.task_id for record in library.experiments.records)
    assert result.compute_cost == sum(action.compute_cost for action in result.actions) + 2 * len(saved.reference)
    assert result.compute_cost <= 2000
    assert result.program == pairs[int(feedback > 0)][0].program
