"""An unresolved decision is actually compared before more fitting is bought."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.decision_frontier import plan_decision_frontier
from hegel_machine.cognition.decision_research import decision_quote, run_decision_research
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.session import ResearchSession
from test_cognition_decision_frontier import _question, _calibration


class Response:
    def __init__(self, spec, target):
        self.spec, self.target, self.queries = spec, target, []

    def observe(self, probe):
        self.queries.append(probe)
        rows, history = [], []
        for point in probe.sequence:
            rows.append(Observation(dict(point), self.target, tuple(history)))
            history.append({**point, '__target__': self.target})
        return rows


def prepared(target):
    spec, workspace, config = _question()
    workspace.decision_frontier = plan_decision_frontier(spec, workspace, config,
        SearchBudget(2000), proposed_model=_calibration())
    environment = Response(spec, target)
    session = ResearchSession(config, [environment], observation_budget=128, compute_budget=10000, max_visits=4)
    session.state['workspaces']['0'] = workspace.to_dict()
    session.state['tasks'][0].update(visits=1, unresolved=True, validation_loss=.9,
        initial_loss=.9, normalized_excess_loss=1., latest_episode=None)
    return session, environment, workspace


@pytest.mark.parametrize('target,adopted', [(0., False), (1., True)])
def test_session_runs_only_the_frozen_short_comparison_and_prices_actual_decisions(target, adopted):
    session, environment, workspace = prepared(target)
    quote = decision_quote(workspace)
    saved = session.to_dict()
    assert saved['tasks'][0]['next_research']['kind'] == 'decision'
    assert saved['tasks'][0]['next_research']['expected_gain'] is None
    result = session.run(max_steps=1)
    episode = result['episodes'][0]
    assert episode['scheduler']['reason'] == 'unresolved_decision_question'
    assert len(environment.queries) == 2
    assert episode['observation_cost'] == quote['observation_cost'] == 16
    assert episode['compute_cost'] == quote['compute_cost']
    assert not any(a['action'] in {'compare', 'execute_method_step', 'execute_research_question'} for a in episode['actions'])
    action = next(a['details'] for a in episode['actions'] if a['action'] == 'execute_decision_question')
    assert action['adopted'] is adopted
    outcome = episode['scheduler']['research_return']
    assert outcome['condition']['kind'] == 'decision' and outcome['execution_complete']
    assert outcome['reference_rows'] == 16
    assert (outcome['gain'] > 0.) is adopted
    assert episode['program'] == workspace.incumbent.program.to_dict()
    assert episode['uncertainty_model'] == (_calibration() if adopted else workspace.uncertainty_model)
    if not adopted:
        before = len(environment.queries)
        restored = ResearchSession(session.config, [environment], state=json.loads(json.dumps(result)),
            observation_budget=128, compute_budget=10000, max_visits=4)
        done = restored.run()
        assert len(environment.queries) == before and done['completed_visits'] == 1
        assert done['tasks'][0]['unresolved']


def test_short_decision_comparison_requires_its_whole_quote_before_any_query():
    session, environment, workspace = prepared(1.)
    quote = decision_quote(workspace)
    session.limits['compute_budget'] = quote['compute_cost'] - 1
    result = session.run()
    assert result['status'] == 'budget_exhausted'
    assert result['pending'] == [0] and not environment.queries
    assert result['workspaces']['0']['decision_frontier'] == workspace.decision_frontier
    restored = ResearchSession(session.config, [environment], state=json.loads(json.dumps(result)),
        observation_budget=128, compute_budget=quote['compute_cost'], max_visits=4)
    done = restored.run(max_steps=1)
    assert len(environment.queries) == 2
    assert done['total_compute_cost'] == quote['compute_cost']
