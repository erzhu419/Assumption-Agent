"""A costly replay cannot spend the next investigation's compute allowance."""

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.growth_methods import ResearchMethodBook
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
from hegel_machine.cognition.workflow_execution import WorkflowEpisode
from test_cognition_unified_workflows import _claim, _paid, _true_program


def _two_fit_workflow():
    """A genuine counterexample links an initial fit to a repaired second fit."""
    lab = UnknownTaskLab(0, seed=316)
    episode = WorkflowEpisode(ResearchMethodBook(), lab.spec, lab.channels)
    training = _paid(episode, lab, "restricted_training", (0., 0., 0., 0.))
    validation = _paid(episode, lab, "restricted_validation", (0., 0.))
    initial = Program("const", value=.15)
    episode.fitted([_claim(initial, training, validation)], training, validation, lab.spec)
    counterexample = _paid(episode, lab, "fresh_counterexample", (-1., 1.))
    episode.tested(initial, counterexample, loss(initial, counterexample, 0., .5), accepted=False)
    repaired_training = [*training, *counterexample]
    repaired = _true_program()
    episode.fitted([_claim(repaired, repaired_training, validation)],
                   repaired_training, validation, lab.spec)
    fresh = _paid(episode, lab, "fresh_repair_confirmation", (-.5, .7))
    episode.tested(repaired, fresh, loss(repaired, fresh, 0., .5), accepted=True)
    method = episode.finish(repaired, confirmed=True, threshold=1e-7, actions=[])
    assert sum(node["action"] == "fit" for node in method["nodes"]) == 2
    return episode.book


@pytest.mark.parametrize("exhaust_first_fit", [False, True])
def test_all_replay_fits_share_one_cap_and_leave_native_confirmation_funded(monkeypatch, exhaust_first_fit):
    library = KnowledgeLibrary()
    library.research_workflows = _two_fit_workflow()
    config = ResearchConfig(observation_budget=192, compute_budget=12000,
                            minimum_improvement=1e-7, max_candidates=16)
    lab = UnknownTaskLab(0, seed=531)
    active = False
    fits, fallback_searches = [], []
    replay = WorkflowEpisode.replay
    research_view = UnifiedResearchAgent._research_view

    def tracked_replay(self, **kwargs):
        nonlocal active
        active = True
        try:
            return replay(self, **kwargs)
        finally:
            active = False

    def costly_search(self, spec, training, validation, budget, *args, **kwargs):
        start, available = budget.spent, budget.remaining
        claims = research_view(self, spec, training, validation, budget, *args, **kwargs)
        if active:
            # Model search and losses use the real purchased observations.
            # Artificially costly continuation exposes the controller's actual
            # budget boundary; it neither supplies labels nor changes a claim.
            first = not fits
            extra = budget.remaining // 2 if first and not exhaust_first_fit else budget.remaining
            assert budget.spend(extra)
            before = budget.spent
            assert not budget.spend(budget.remaining + 1)
            assert budget.spent == before
            fits.append({"available": available, "spent": budget.spent - start})
        else:
            fallback_searches.append({"available": available, "spent": budget.spent - start})
        return claims

    monkeypatch.setattr(WorkflowEpisode, "replay", tracked_replay)
    monkeypatch.setattr(UnifiedResearchAgent, "_research_view", costly_search)
    report = UnifiedResearchAgent(config, library=library).run(lab)
    assert len(fits) == 2
    if exhaust_first_fit:
        assert fits[1]["available"] == fits[1]["spent"] == 0
    else:
        assert fits[1]["available"] > 0
    allocations = [item for item in report["actions"] if item["action"] == "budget_workflow_fit"]
    assert len(allocations) == 2
    cap = allocations[0]["workflow_compute_limit"]
    assert all(item["workflow_compute_limit"] == cap for item in allocations)
    assert sum(item["spent"] for item in fits) == cap
    assert allocations[-1]["workflow_compute_spent"] == cap
    assert cap < config.compute_budget
    assert all(item["measured_alternative_funded"] for item in allocations)
    # The reserve must permit real work after the expensive replay, not merely
    # leave an unspent number in a ledger.
    final_fit = max(index for index, item in enumerate(report["actions"])
                    if item["action"] == "budget_workflow_fit")
    after = report["actions"][final_fit + 1:]
    assert any(item["action"] == "acquire" and item["purpose"] == "confirm_frozen_program"
               for item in after)
    if exhaust_first_fit:
        assert any(item["spent"] > 0 for item in fallback_searches)
    assert report["status"] == "completed"
    assert any(item["action"] in {"validate_frozen_program", "confirm_observation_research"}
               and item["accepted"] for item in after)
    assert report["costs"]["compute_cost"] <= config.compute_budget
    assert report["costs"]["observation_cost"] == lab.observation_cost
    assert report["costs"]["measurement_cost"] == lab.measurement_cost
