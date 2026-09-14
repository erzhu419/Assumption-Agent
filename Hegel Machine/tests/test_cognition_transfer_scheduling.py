"""Reuse should save observations while a changed mechanism still gets revised."""

from dataclasses import replace

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig


def test_supported_transfer_stops_before_purchasing_redundant_queries():
    config = ResearchConfig(seed=401, partition='continual')
    library = KnowledgeLibrary()
    CognitiveAgent(config, library).run(make_suite(401, 'continual')[0])
    saved = library.to_dict()
    results = []
    for schedule in (False, True):
        env = make_suite(401, 'continual')[5]
        result = CognitiveAgent(replace(config, transfer_scheduling=schedule),
                                KnowledgeLibrary.from_dict(saved)).run(env)
        assert loss(result.program, env.evaluation_rows(), result.fallback, .5) < 1e-12
        assert any(a.action == 'compare' for a in result.actions)
        results.append(result)
    assert results[1].observation_cost < results[0].observation_cost


def test_same_context_mechanism_change_is_relearned_after_a_failed_transfer():
    config = ResearchConfig(seed=401, partition='continual')
    library = KnowledgeLibrary()
    CognitiveAgent(config, library).run(make_suite(401, 'continual')[0])
    env = make_suite(401, 'continual')[-1]
    result = CognitiveAgent(config, library).run(env)
    assert loss(result.program, env.evaluation_rows(), result.fallback, .5) < 1e-12
    assert any(a.action == 'revise' for a in result.actions)
    assert result.observation_cost <= config.observation_budget


def test_state_revision_stays_accurate_with_adaptive_experiment_choice():
    config = ResearchConfig(seed=401, partition='continual', use_memory=False)
    env = make_suite(401, 'continual')[9]
    result = CognitiveAgent(config).run(env)
    assert loss(result.program, env.evaluation_rows(), result.fallback, .5) < .03
    adverse = [a.round for a in result.actions if a.action == 'targeted_prediction_check'
               and a.details['further_revision_needed']]
    for turn in adverse:
        assert any(a.action == 'compare' and a.round > turn for a in result.actions)
    assert all(a.round < config.max_rounds - 1 for a in result.actions
               if a.action == 'observe' and a.details['probe_id'].startswith('transition_'))


def test_falsified_transfer_uses_a_short_next_action_and_revises_within_budget():
    config = ResearchConfig(seed=503, partition='continual')
    library = KnowledgeLibrary()
    agent = CognitiveAgent(config, library)
    for env in make_suite(503, 'continual')[:8]:
        agent.run(env)
    env = make_suite(503, 'continual')[8]
    result = agent.run(env)
    assert loss(result.program, env.evaluation_rows(), result.fallback, .5) < 1e-12
    deferred = [a for a in result.actions if a.action == 'plan_experiment'
                and a.details.get('action_policy', {}).get('context', '').endswith(':transfer_counterexample')]
    assert deferred
    assert deferred[0].details['action_policy']['quoted_observation_cost'] <= 8
    assert any(a.action == 'compare' and a.round > deferred[0].round for a in result.actions)
    assert result.observation_cost <= 96 and result.compute_cost <= 24000
