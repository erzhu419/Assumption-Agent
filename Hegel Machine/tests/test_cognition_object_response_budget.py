"""Paid role discovery must leave evidence and resources for response learning."""

from dataclasses import asdict

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.object_research_actions import ObjectResearchActions
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_object_interventions import discover


def test_paid_object_bank_fits_response_before_fresh_comparison_and_confirmation(monkeypatch):
    config = ResearchConfig(observation_budget=192, compute_budget=120000,
        minimum_improvement=1e-7, max_candidates=16, max_rounds=4,
        active_probes=True, seed=71)
    measurement_budget = 1728
    response_inputs = []
    original = UnifiedResearchAgent._research_view

    def record_response_inputs(self, spec, training, validation, budget, actions, *args, **kwargs):
        if any(action["action"] == "reuse_object_evidence_for_response" for action in actions):
            response_inputs.append({"features": set(spec.feature_names),
                "training": [asdict(row) for row in training],
                "validation": [asdict(row) for row in validation],
                "mapped_instruments": any(action["action"] == "research_instrument_mapping"
                    for action in actions)})
        return original(self, spec, training, validation, budget, actions, *args, **kwargs)

    monkeypatch.setattr(UnifiedResearchAgent, "_research_view", record_response_inputs)
    with AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013,
                            name_seed=42114, wiring_seed=1780, seed=71) as laboratory:
        agent = UnifiedResearchAgent(config, measurement_budget=measurement_budget)
        report = agent.run(laboratory)
        assert report["object_research"]["complete"]
        assert report["status"] == "completed"
        actions = report["actions"]
        reused_at = next(index for index, action in enumerate(actions)
                         if action["action"] == "reuse_object_evidence_for_response")
        role_purchases = [action for action in actions[:reused_at]
                          if action["action"] == "acquire" and action["channels"] == ["fine"]]
        assert role_purchases
        assert any(action["purpose"] == "observe_anonymous_dependencies" for action in role_purchases)
        batches = {batch["probe"]["probe_id"]: batch for batch in report["batches"]}
        paid_bank = [row for action in role_purchases
                     for row in batches[action["probe_id"]]["observations"]]
        response = next(item for item in response_inputs
                        if item["features"] == set(laboratory.spec_for(("fine",)).feature_names))
        # Observed roles suffice to research the response. Calibration is paid
        # only when a later input-repair or deployment question requests it.
        assert not response["mapped_instruments"]
        assert any(action["action"] == "defer_object_instrument_mapping"
                   for action in actions[:reused_at])
        mapped_at = next(index for index, action in enumerate(actions)
                         if action["action"] == "research_instrument_mapping")
        assert mapped_at > reused_at
        model = report["object_research"]["sensor_model"]
        saved_objects = next(item for item in agent.library.object_models
                             if item["model_id"] == report["object_research"]["model_id"])
        assert saved_objects["sensor_model"] == model
        # Reuse all paid role observations, including their reset
        # histories. Buying two new fitting trajectories would waste this bank.
        assert response["training"] == paid_bank
        comparison = next(action for action in actions[reused_at:]
                          if action["action"] == "acquire"
                          and action["purpose"] == "compare_on_new_trajectory")
        compared = batches[comparison["probe_id"]]["observations"]
        assert response["validation"] == compared
        assert all(row not in paid_bank for row in compared)
        confirmation = next(action for action in actions[reused_at:]
                            if action["action"] == "acquire"
                            and action["purpose"] == "confirm_frozen_program")
        confirmed = batches[confirmation["probe_id"]]["observations"]
        assert all(row not in paid_bank and row not in compared for row in confirmed)
        assert report["costs"]["observation_cost"] == laboratory.observation_cost <= config.observation_budget
        assert report["costs"]["measurement_cost"] == laboratory.measurement_cost <= measurement_budget
        assert report["costs"]["compute_cost"] <= config.compute_budget
        # New independently generated trajectories check actual delivered
        # behavior after the fitting/comparison/confirmation workflow ends.
        rows = laboratory.evaluation_rows(report["selected_channels"],
            seed=8191, length=32, trajectories=2)
        assert loss(Program.from_dict(report["program"]), rows, report["fallback"], .5) < 1e-7


def test_failed_short_transfer_leaves_current_response_comparison_and_confirmation_funded():
    config = ResearchConfig(observation_budget=192, compute_budget=120000,
        minimum_improvement=1e-7, max_candidates=16, max_rounds=4,
        active_probes=True, seed=71)
    measurement_budget, library = 1728, KnowledgeLibrary()
    with AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013,
                            name_seed=42114, wiring_seed=1780, seed=71) as source:
        learned = UnifiedResearchAgent(config, library=library,
            measurement_budget=measurement_budget).run(source)
    assert learned["status"] == "completed" and learned["concepts_added"]

    # A changed instrument and an event response make the learned focus
    # predictor fail. The paid object bank makes a short trial affordable,
    # but must not substitute for either subsequent independent trajectory.
    with AnonymousObjectLab(objects=3, readout="event", apparatus_seed=9913,
                            name_seed=12114, wiring_seed=1980, seed=171) as target:
        report = UnifiedResearchAgent(config, library=library,
            measurement_budget=measurement_budget).run(target)
        actions = report["actions"]
        trial_at = next(index for index, action in enumerate(actions)
                        if action["action"] == "test_learned_acquisition_method")
        trial = actions[trial_at]
        assert trial["validation_loss"] > config.minimum_improvement
        assert set(trial["source_ids"]) & set(learned["concepts_added"])
        assert report["object_research"]["complete"]
        acquired_before = [action for action in actions[:trial_at] if action["action"] == "acquire"]
        assert config.observation_budget - sum(action["observation_cost"]
            for action in acquired_before) >= 48
        assert measurement_budget - sum(action["measurement_cost"]
            for action in acquired_before) >= 48 * 9

        batches = {batch["probe"]["probe_id"]: batch for batch in report["batches"]}
        paid_before = [row for action in acquired_before
                       for row in batches[action["probe_id"]]["observations"]]
        comparison = next(action for action in actions[trial_at:]
            if action["action"] == "acquire" and action["purpose"] == "compare_on_new_trajectory")
        confirmation = next(action for action in actions[trial_at:]
            if action["action"] == "acquire" and action["purpose"] == "confirm_frozen_program")
        compared = batches[comparison["probe_id"]]["observations"]
        confirmed = batches[confirmation["probe_id"]]["observations"]
        assert len(compared) == len(confirmed) == 24
        assert all(row not in paid_before for row in compared)
        assert all(row not in paid_before and row not in compared for row in confirmed)
        assert report["status"] == "completed"
        assert report["costs"]["observation_cost"] == target.observation_cost <= config.observation_budget
        assert report["costs"]["measurement_cost"] == target.measurement_cost <= measurement_budget
        assert report["costs"]["compute_cost"] <= config.compute_budget
        fresh = target.evaluation_rows(report["selected_channels"], seed=8191, length=32, trajectories=2)
        assert loss(Program.from_dict(report["program"]), fresh, report["fallback"], .5) < 1e-7


@pytest.mark.parametrize("fresh_comparison", [True, False])
def test_short_calibration_batch_is_fitting_evidence_not_a_response_holdout(monkeypatch, fresh_comparison):
    config = ResearchConfig(compute_budget=120000, minimum_improvement=1e-7,
                            max_candidates=16, max_question_depth=2)
    budget, actions, purchased, researched = SearchBudget(config.compute_budget), [], [], []
    with AnonymousObjectLab(objects=3, readout="focus", apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as laboratory:
        acquire_original = laboratory.acquire

        def record_acquisition(probe, channels=()):
            batch = acquire_original(probe, channels)
            pairs = [(Observation(point, row.target, tuple(probe.sequence[:index])), row)
                     for index, (point, row) in enumerate(zip(probe.sequence, batch.observations))]
            purchased.append({"channels": tuple(channels), "pairs": pairs,
                              "transitions": batch.transitions})
            return batch

        monkeypatch.setattr(laboratory, "acquire", record_acquisition)
        objects, _, _ = discover(laboratory)
        assert objects["complete"]
        laboratory.acquire(laboratory.random_probe(7117, length=24), ("fine",))
        point = laboratory.random_probe(8103, length=1).sequence[0]
        focus = objects["schema"]["focus_control"]
        laboratory.acquire(Probe("short_calibration_cut", (
            {**point, focus: .49}, {**point, focus: .51})), ("fine",))
        bank = [row for batch in purchased for _, row in batch["pairs"]]
        assert len(purchased[-1]["pairs"]) == 2
        length = 24
        rows_left = length * (2 if fresh_comparison else 1)
        observation_limit = laboratory.observation_cost + rows_left
        measurement_limit = laboratory.measurement_cost + 9 * rows_left

        def remaining():
            return {"observation_remaining": observation_limit - laboratory.observation_cost,
                "measurement_remaining": measurement_limit - laboratory.measurement_cost,
                "compute_remaining": budget.remaining}

        def acquire(channels, purpose, *, probe=None, rows_count=None):
            assert purpose == "compare_on_new_trajectory"
            probe = probe or laboratory.random_probe(9713, length=rows_count or length)
            return laboratory.acquire(probe, channels).observations

        def paired_for(channels):
            return [batch["pairs"] for batch in purchased if batch["channels"] == tuple(channels)]

        def transitions_for(channels, rows):
            return [transition for batch in purchased if batch["channels"] == tuple(channels)
                    for (_, measured), transition in zip(batch["pairs"], batch["transitions"])
                    if measured in rows]

        agent = UnifiedResearchAgent(config)
        agent._object_schemas[("fine",)] = objects["schema"]
        research_original = agent._research_view

        def record_research(spec, training, validation, *args, **kwargs):
            researched.append((list(training), list(validation)))
            return research_original(spec, training, validation, *args, **kwargs)

        monkeypatch.setattr(agent, "_research_view", record_research)
        controller = ObjectResearchActions(agent, laboratory, acquire=acquire,
            paired_for=paired_for, command_batches=lambda: [[command for command, _ in batch["pairs"]]
                for batch in purchased], view_rows=lambda rows, source, target: laboratory.project_rows(rows, source, target),
            transitions_for=transitions_for, budget=budget, actions=actions, transfer=[],
            graph=ResearchGraph(), remaining=remaining, length=length)
        result = controller.propose(None, ("fine",))
        assert researched and result["selected"] is not None
        training, validation = researched[0]
        assert training == bank
        if fresh_comparison:
            assert len(validation) == length
            assert all(row not in bank for row in validation)
        else:
            assert validation == []
            assert result["selected"]["selection_basis"] == "all_acquired_evidence"
        # Neither the independent selection comparison nor an all-paid fit
        # substitutes for the separately reserved frozen confirmation.
        assert not result["confirmed"]
        assert remaining()["observation_remaining"] >= length
        assert remaining()["measurement_remaining"] >= 9 * length
        assert not agent.library.concepts
