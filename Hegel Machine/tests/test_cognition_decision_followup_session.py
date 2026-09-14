"""An unresolved paid comparison survives an actual session continuation."""

from dataclasses import replace
import json

from hegel_machine.cognition.research_methods import ResearchMethods
from hegel_machine.cognition.session import ResearchSession
from test_cognition_decision_research_runtime import prepared, Response
from test_cognition_method_returns import method_evidence, measured_step


def test_conflicting_decisions_remain_fundable_after_json_restore():
    session, original, workspace = prepared(0.)

    class ChangingResponse(Response):
        def observe(self, probe):
            self.target = [0., 1., 1., 1.][len(self.queries)]
            return super().observe(probe)

    environment = ChangingResponse(original.spec, 0.)
    config = replace(session.config, observation_budget=16)
    previous_task = dict(session.state['tasks'][0])
    session = ResearchSession(config, [environment],
        observation_budget=32, compute_budget=10000, max_visits=3)
    session.state['workspaces']['0'] = workspace.to_dict()
    session.state['tasks'][0].update(previous_task)
    first = session.run(max_steps=1)
    outcome = first['episodes'][0]['scheduler']['research_return']
    assert not outcome['execution_complete'] and outcome['gain'] is None
    pending = first['workspaces']['0']['decision_frontier']
    evidence = pending['evidence_states'][0]
    assert evidence['comparison']['trajectories'] == 2
    assert evidence['previous_model'] == workspace.uncertainty_model
    assert first['tasks'][0]['next_research']['kind'] == 'decision'
    assert first['workspaces']['0']['uncertainty_model'] == workspace.uncertainty_model
    restored = ResearchSession(config, [environment], state=json.loads(json.dumps(first)),
        observation_budget=32, compute_budget=10000, max_visits=3)
    final = restored.run(max_steps=1)
    assert len(environment.queries) == 4 and final['total_observation_cost'] == 32
    second = final['episodes'][1]
    compared = [action for action in second['actions']
                if action['action'] == 'compare_decision_question'][-1]
    assert compared['details']['trajectories'] == 4
    assert compared['details']['trajectory_comparisons'][:2] == evidence['comparison']['trajectory_comparisons']
    assert compared['compute_cost'] == 32
    assert second['scheduler']['research_return']['execution_complete']
    assert second['scheduler']['research_return']['gain'] > 0.
    assert final['workspaces']['0']['uncertainty_model'] == evidence['candidate_model']
    assert final['workspaces']['0']['decision_frontier'] == {}


def test_unresolved_complete_decision_does_not_erase_a_method_with_zero_return():
    _, _, profile, context, condition = method_evidence()
    methods = ResearchMethods()
    identifier = methods.observe(context, [measured_step('representation', .9, context, profile)],
        condition=condition, source_task='source')
    before = json.loads(json.dumps(methods.to_dict()))
    recorded = methods.observe_continuation(identifier, profile,
        {'execution_complete': False, 'normalized_gain': 0., 'gain': 0.,
         'reference_rows': 16, 'observation_cost': 80, 'compute_cost': 20000},
        source_task='unresolved_target')
    assert recorded is None and methods.to_dict() == before
