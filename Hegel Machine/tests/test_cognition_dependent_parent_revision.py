"""Dependent research can revise a parent's measured contribution."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _fit, _terms
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
from test_cognition_unified_action_confirmation import _assert_fresh_comparison
from test_cognition_unified_research import PublicLaboratory


class RecordedDependentResearch(UnifiedResearchAgent):
    """Observe the actual parent and graph execution without replacing either."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dependent_executions = []

    def _dependent_residual_claims(self, spec, training, budget, parent, graph, actions):
        before = budget.spent
        claims = super()._dependent_residual_claims(spec, training, budget, parent, graph, actions)
        self.dependent_executions.append({'parent': deepcopy(parent), 'claims': deepcopy(claims),
            'training': training, 'graph': graph.to_dict(), 'compute_cost': budget.spent - before})
        return claims


def _acquire(laboratory, seed, length, *, channels=(), probe_id='paid_response'):
    rng = Random(seed)
    return laboratory.acquire(Probe(probe_id, tuple(
        {'u': rng.uniform(-1., 1.)} for _ in range(length))), channels).observations


def test_real_wrong_long_history_parent_can_be_reweighted_before_fresh_confirmation(record_property):
    seed = 94701901
    config = ResearchConfig(observation_budget=192, compute_budget=120000,
        minimum_improvement=1e-7, max_rounds=4, max_candidates=16,
        active_probes=True, seed=seed)
    laboratory = UnknownTaskLab(2, seed)
    agent = RecordedDependentResearch(config)
    report = agent.run(PublicLaboratory(laboratory))
    executions = [item for item in agent.dependent_executions
                  if item['parent']['candidate'].program.history_depth >= 3]
    assert executions
    revised = next(item for item in executions
                   if len(item['claims']) > 1 and min(claim['training_loss'] for claim in item['claims']) < 1e-12)
    assert max(claim['training_loss'] for claim in revised['claims']) > config.minimum_improvement
    assert report['status'] == 'completed'
    assert report['selected_kind'] == 'dependent_residual_revision'
    confirmations = [event for event in report['actions']
                     if event['action'] == 'research_action_confirmation'
                     and event['research_action'] == 'dependent_residual_revision']
    assert confirmations
    for event in confirmations:
        before, after = _assert_fresh_comparison(report, event)
        assert event['status'] == 'supported'
        assert before > after and after < 1e-12
    fresh = laboratory.evaluation_rows(report['selected_channels'], seed=733,
                                      length=40, trajectories=2)
    assert loss(Program.from_dict(report['program']), fresh, report['fallback'], .5) < 1e-12
    revisions = [event for event in report['actions']
                 if event['action'] == 'execute_dependent_research'
                 and event['parent_revision'] == 'reestimated_parent'
                 and event['parent_candidate_id'] == revised['parent']['candidate'].candidate_id]
    assert revisions
    graph = ResearchGraph.from_dict(revised['graph'])
    corrected = next(event for event in revisions
                     if event['question_id'] in graph.nodes and event['parent_coefficient'] == 0.)
    node = graph.nodes[corrected['question_id']]
    assert node.residual_base.op == 'mul'
    assert node.residual_base.children[1] == revised['parent']['candidate'].program
    answer = graph.nodes[corrected['question_id'] + ':combined']
    assert answer.depends_on == [node.question_id]
    assert loss(answer.answer, fresh, 0., .5) < 1e-12
    fixed = revised['claims'][0]
    assert loss(fixed['candidate'].program, fresh, 0., .5) > config.minimum_improvement
    assert report['costs']['observation_cost'] == laboratory.observation_cost
    assert report['costs']['measurement_cost'] == laboratory.measurement_cost
    assert report['costs']['compute_cost'] <= config.compute_budget
    record_property('revised_source_costs', json.dumps(report['costs']))
    record_property('parent_depth', revised['parent']['candidate'].program.history_depth)
    record_property('candidate_losses', json.dumps([item['training_loss'] for item in revised['claims']]))


def test_correct_paid_parent_remains_executable_when_new_measurements_update_the_graph():
    source = UnknownTaskLab(0, 113)
    source_rows = _acquire(source, 17, 48)
    parent_budget = SearchBudget(1000)
    parent = _fit(_terms(('u',)), source_rows, parent_budget, 'paid_current_parent', q=.5)
    assert parent['training_loss'] < 1e-12
    target = UnknownTaskLab(2, 113)
    training = _acquire(target, 31, 48, channels=('sensor_a',), probe_id='new_measured_coordinates')
    config = ResearchConfig(compute_budget=12000, minimum_improvement=1e-7)
    graph, actions, budget = ResearchGraph(), [], SearchBudget(config.compute_budget)
    claims = UnifiedResearchAgent(config)._dependent_residual_claims(
        target.spec_for(('sensor_a',)), training, budget, parent, graph, actions)
    exact = min(claims, key=lambda item: item['training_loss'])
    assert exact['training_loss'] < 1e-12
    assert all(event['parent_revision'] == 'fixed_parent' and event['parent_coefficient'] == 1.
               for event in actions if event['action'] == 'execute_dependent_research')
    parent_program = parent['candidate'].program
    child = next(node for node in graph.nodes.values() if node.residual_base == parent_program)
    combined = graph.nodes[child.question_id + ':combined']
    assert child.status == combined.status == 'answered'
    assert combined.depends_on == [child.question_id]
    assert combined.answer == exact['candidate'].program
    # New paid evidence arrives after both explanations have been frozen.
    fresh = _acquire(target, 59, 24, channels=('sensor_a',), probe_id='later_measured_coordinates')
    assert loss(exact['candidate'].program, fresh, 0., .5) < 1e-12
    restored = ResearchGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert restored.supply_measurements(fresh, question_id=child.question_id) == [child.question_id]
    followup_budget = SearchBudget(1000)
    answered = restored.execute_ready(fresh, followup_budget, max_nodes=2)
    revised = next(candidate for candidate in answered
                   if candidate.candidate_id == exact['candidate'].candidate_id)
    assert restored.nodes[child.question_id].residual_base == parent_program
    assert loss(revised.program, fresh, 0., .5) < 1e-12
    assert followup_budget.spent > 0
