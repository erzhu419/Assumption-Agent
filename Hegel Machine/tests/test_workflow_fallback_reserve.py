"""A retained DAG must leave a complete legal continuation, not just a start."""

from dataclasses import replace

from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import (
    UnifiedResearchAgent, _conditional_experiment_entry_quote,
)
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
from hegel_machine.cognition.workflow_execution import WorkflowEpisode
from test_cognition_unified_workflows import _learned_episode
from test_cognition_workflow_budget import _two_fit_workflow


def test_replay_prices_whole_dag_confirmation_and_continuation_before_any_purchase():
    source, _ = _learned_episode()
    lab = UnknownTaskLab(0)
    episode = WorkflowEpisode(source.book, lab.spec, lab.channels)
    method = source.book.methods[0]
    reserve = {'observation_rows': 10, 'measurement_cost': 0, 'compute_cost': 40}
    arguments = dict(legal_views=[()], projectable=None,
        confirmation_length=2, continuation_reserve=reserve)
    plan, reason = episode._replay_plan(method, **arguments,
        limits={'observation_remaining': 17, 'measurement_remaining': 0, 'compute_remaining': 1000})
    assert plan is None and reason == 'execution_confirmation_and_continuation_not_funded'
    plan, reason = episode._replay_plan(method, **arguments,
        limits={'observation_remaining': 18, 'measurement_remaining': 0, 'compute_remaining': 55})
    assert plan is None and reason == 'execution_confirmation_and_continuation_not_funded'
    plan, reason = episode._replay_plan(method, **arguments,
        limits={'observation_remaining': 18, 'measurement_remaining': 0, 'compute_remaining': 56})
    assert reason is None and plan['observation_rows'] == 6
    assert plan['continuation_reserve'] == reserve
    assert lab.observation_cost == lab.measurement_cost == 0


def test_control_pack_quote_includes_its_shared_reference_and_boundary_capacity():
    spec = replace(UnknownTaskLab(0).spec, feature_names=('p', 'q', 'r'),
        feature_bounds={name: (-1., 1.) for name in ('p', 'q', 'r')})
    quote = _conditional_experiment_entry_quote(spec, 48, 24)
    assert quote['observation_rows'] == 4 * 12 + 12
    assert quote['compute_cost'] == (48 + 60) * 44 + 24 * 8


def test_unavailable_matched_experiment_does_not_block_a_complete_old_workflow():
    class ShortLab(UnknownTaskLab):
        def spec_for(self, channels=()):
            return replace(super().spec_for(channels), max_probe_length=4)

    old, _ = _learned_episode()
    library = KnowledgeLibrary()
    library.research_workflows = old.book
    lab = ShortLab(0)
    lab.channels = ()
    config = ResearchConfig(observation_budget=64, compute_budget=6000,
        minimum_improvement=1e-7, max_candidates=8)
    report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(lab)
    replay = [event for event in report['actions'] if event['action'] == 'replay_research_workflow']
    assert replay and replay[0]['continuation_reserve'] == {}
    assert report['status'] == 'completed'
    assert report['costs']['observation_cost'] == lab.observation_cost <= 64


def test_failed_control_only_workflow_keeps_real_fallback_compute_and_confirmation(monkeypatch):
    library = KnowledgeLibrary()
    library.research_workflows = _two_fit_workflow()
    lab = UnknownTaskLab(0, seed=531)
    lab.channels = ()
    config = ResearchConfig(observation_budget=240, compute_budget=60000,
        minimum_improvement=1e-7, max_candidates=16)
    active, paid_fallback = False, []
    replay = WorkflowEpisode.replay
    research = UnifiedResearchAgent._research_view

    def scoped_replay(self, **kwargs):
        nonlocal active
        active = True
        try:
            return replay(self, **kwargs)
        finally:
            active = False

    def expensive(self, spec, training, validation, budget, *args, **kwargs):
        before = budget.spent
        claims = research(self, spec, training, validation, budget, *args, **kwargs)
        if active:
            assert budget.spend(budget.remaining)
        else:
            paid_fallback.append(budget.spent - before)
        return claims

    monkeypatch.setattr(WorkflowEpisode, 'replay', scoped_replay)
    monkeypatch.setattr(UnifiedResearchAgent, '_research_view', expensive)
    report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(lab)
    allocations = [event for event in report['actions'] if event['action'] == 'budget_workflow_fit']
    assert allocations and all(event['continuation_reserve']['compute_cost'] > 0 for event in allocations)
    assert all(not event['measured_alternative_funded'] for event in allocations)
    assert sum(event['compute_cost'] for event in allocations) == allocations[0]['workflow_compute_limit']
    assert any(value > 0 for value in paid_fallback)
    assert any(event['action'] == 'validate_frozen_program' and event['accepted']
               for event in report['actions'])
    assert report['status'] == 'completed'
    assert report['costs']['observation_cost'] == lab.observation_cost <= config.observation_budget
    assert report['costs']['compute_cost'] <= config.compute_budget
    assert report['costs']['measurement_cost'] == 0
