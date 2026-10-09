"""Paid measurement recovery keeps its complete fit-and-confirmation allocation."""

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.observation_research import research_observation_actions
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _fit, _terms
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab

from test_cognition_observation_research import ActualPurchases


@pytest.mark.parametrize("observation_limit,pilot_lengths,first_action", [
    (192, (8, 8, 24, 24, 24), "remeasure_inputs"),
    (240, (24, 24), "revise_mechanism"),
])
def test_paid_recovery_packages_preserve_funded_measurement_and_fresh_confirmation(
        observation_limit, pilot_lengths, first_action):
    config = ResearchConfig(observation_budget=observation_limit, compute_budget=120000,
        minimum_improvement=1e-7, max_rounds=4, max_candidates=16, active_probes=False, seed=11813)
    agent = UnifiedResearchAgent(config, measurement_budget=768)
    budget, actions = SearchBudget(config.compute_budget, reserve=48), []
    with UnknownTaskLab(task_index=10, seed=11813) as laboratory:
        paid = ActualPurchases(laboratory, budget, actions, observation_budget=observation_limit,
                               measurement_budget=768)
        for rows in pilot_lengths:
            paid.acquire((), "paid_command_pilot", rows_count=rows)
        batches = paid.command_batches()
        training = [row for batch in batches[:-1] for row in batch]
        parent = _fit(_terms(laboratory.spec.feature_names), training, budget, "paid_parent")
        start_observations = laboratory.costs["observation_cost"]
        research_observation_actions(agent, laboratory, acquire=paid.acquire,
            paired_for=paid.paired_for, command_batches=paid.command_batches, budget=budget,
            actions=actions, transfer=[], graph=ResearchGraph(), remaining=paid.remaining,
            length=24, parent=parent, failed_probe=paid.batches[-1]["probe"])
        choices = [action for action in actions if action["action"] == "choose_observation_research"]
        assert choices[0]["selected_action"] == first_action
        measured = next(action for action in choices if action["selected_action"] == "remeasure_inputs")
        assert measured["quoted_observation_cost"] == 96
        if first_action == "remeasure_inputs":
            assert measured["observation_remaining"] == 104
            assert measured["deferred_actions"] == [{"action": "revise_mechanism",
                "reason": "preserve_funded_measurement_and_response_package"}]
        else:
            assert measured["observation_remaining"] >= 96
        packages = [action for action in actions if action["action"] == "finish_observation_research_package"]
        assert len(packages) == len(choices)
        for package in packages:
            for resource in ("observation", "measurement", "compute"):
                assert package[resource + "_cost"] <= package["quoted_" + resource + "_cost"]
        assert sum(item["observation_cost"] for item in packages) == (
            laboratory.costs["observation_cost"] - start_observations)
        assert laboratory.costs["observation_cost"] <= observation_limit
        assert laboratory.costs["measurement_cost"] <= 768
        assert budget.reserve == 48 and budget.spent <= config.compute_budget - 48
        confirmations = [action for action in actions if action["action"] == "confirm_observation_research"]
        assert any(action["research_action"] == "remeasure_inputs" for action in confirmations)
        for action in confirmations:
            purchased = next(batch for batch in paid.batches if batch["probe"].probe_id == action["probe_id"])
            assert purchased["purpose"] == "confirm_observation_research"
            actual = loss(Program.from_dict(action["program"]),
                [row for _, row in purchased["pairs"]], action["fallback"], action["risk_quantile"])
            assert action["after_loss"] == pytest.approx(actual)
