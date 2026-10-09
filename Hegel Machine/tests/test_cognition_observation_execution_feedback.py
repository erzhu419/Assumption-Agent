"""An unfunded mechanism attempt is not evidence that its method failed."""

from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.observation_research import research_observation_actions
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _fit, _terms
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
from test_cognition_observation_research import ActualPurchases


CONFIG = ResearchConfig(observation_budget=192, compute_budget=120000,
    minimum_improvement=1e-7, max_rounds=4, max_candidates=16,
    active_probes=False, seed=11811)


def test_unfunded_mechanism_attempt_cannot_suppress_the_next_tasks_real_attempt(record_property):
    library = KnowledgeLibrary()
    with UnknownTaskLab(task_index=8, seed=11811) as laboratory:
        first = UnifiedResearchAgent(replace(CONFIG, compute_budget=512),
            measurement_budget=768, library=library).run(laboratory)
        assert first["costs"]["observation_cost"] == laboratory.costs["observation_cost"] > 0
        assert first["costs"]["measurement_cost"] == laboratory.costs["measurement_cost"]
    assert 0 < first["costs"]["compute_cost"] <= 512
    choice = next(action for action in first["actions"]
                  if action["action"] == "choose_observation_research")
    assert choice["selected_action"] == "revise_mechanism"
    assert any(action["action"] == "choose_research_action"
               and action["reason"] == "no_affordable_action"
               for action in first["actions"])
    context = choice["context"]
    record = library.action_experience.get(context, {}).get("revise_mechanism", {})
    assert record.get("trials", 0) == 0
    assert "expected_gain" not in record and not record.get("trial_outcomes")
    assert not record.get("unscored_attempts")
    assert not any(action["action"] == "observation_research_feedback"
                   and action["research_action"] == "revise_mechanism"
                   and action["context"] == context for action in first["actions"])
    unexecuted = next(action for action in first["actions"]
                      if action["action"] == "observation_research_unexecuted"
                      and action["research_action"] == "revise_mechanism")
    assert unexecuted["reason"] == "unfunded_mechanism_research"
    assert unexecuted["execution"]["search_compute_cost"] == 0
    assert unexecuted["execution"]["compared_candidates"] == 0
    assert 0 < unexecuted["compute_cost"] <= first["costs"]["compute_cost"]
    assert not any(action["action"] == "confirm_observation_research"
                   and action["research_action"] == "revise_mechanism" for action in first["actions"])
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert restored.action_experience == library.action_experience
    with UnknownTaskLab(task_index=8, seed=11812) as laboratory:
        second = UnifiedResearchAgent(replace(CONFIG, seed=11812),
            measurement_budget=768, library=restored).run(laboratory)
    next_choice = next(action for action in second["actions"]
                       if action["action"] == "choose_observation_research")
    assert next_choice["context"] == context
    assert next_choice["selected_action"] == "revise_mechanism"
    assert next_choice["reason"] == "untried_action"
    record_property("limited_costs", first["costs"])
    record_property("successor_costs", second["costs"])
    record_property("successor_status", second["status"])
    record_property("successor_concepts_added", second["concepts_added"])
    record_property("successor_confirmation_loss", second["confirmation_loss"])


def test_real_refitting_to_the_same_program_still_records_measured_zero_return():
    # Fund the parent fit, one fresh affine fit/comparison, and confirmation.
    # The actual continuous-state response remains imperfectly explained.
    config = replace(CONFIG, compute_budget=400, max_question_depth=0)
    agent = UnifiedResearchAgent(config, measurement_budget=0, library=KnowledgeLibrary())
    budget, actions = SearchBudget(config.compute_budget), []
    with UnknownTaskLab(task_index=8, seed=11811) as laboratory:
        paid = ActualPurchases(laboratory, budget, actions, measurement_budget=0,
                               observation_budget=config.observation_budget)
        for _ in range(2):
            paid.acquire((), "initial_paid_response", rows_count=24)
        training, validation = paid.command_batches()
        parent = _fit(_terms(laboratory.spec.feature_names), training, budget, "prior_affine")
        assert parent is not None
        result = research_observation_actions(agent, laboratory, acquire=paid.acquire,
            paired_for=paid.paired_for, command_batches=paid.command_batches, budget=budget,
            actions=actions, transfer=[], graph=ResearchGraph(), remaining=paid.remaining,
            length=24, parent=parent, failed_probe=paid.batches[-1]["probe"])
        confirmation = next(action for action in actions
                            if action["action"] == "confirm_observation_research")
        assert confirmation["program"] == confirmation["baseline_program"] == parent["candidate"].program.to_dict()
        assert not confirmation["accepted"] and not result["confirmed"]
        actual_rows = [row for batch in paid.batches if batch["purpose"] == "confirm_observation_research"
                       for _, row in batch["pairs"]]
        measured = loss(parent["candidate"].program, actual_rows, result["fallback"], .5)
        assert measured > config.minimum_improvement
        assert confirmation["before_loss"] == pytest.approx(measured)
        assert confirmation["after_loss"] == pytest.approx(measured)
    feedback = next(action for action in actions if action["action"] == "observation_research_feedback")
    record = agent.library.action_experience[feedback["context"]]["revise_mechanism"]
    assert record["trials"] == 1 and record["expected_gain"] == 0.
    assert record["trial_outcomes"]["1"]["status"] == "refuted"
    assert feedback["compute_cost"] > 0 and feedback["observation_cost"] == 24
    assert not any(action["action"] == "observation_research_unexecuted" for action in actions)


@pytest.mark.parametrize("route", ["observation", "measured_stage"])
def test_calibration_that_eliminates_the_source_confirms_without_transfer_credit(route, monkeypatch):
    class LinearMeterLab(UnknownTaskLab):
        def _experiment(self, probe):
            return [(fields, .4 + 1.7 * fields["sensor_a"])
                    for fields, _ in super()._experiment(probe)]

    config = replace(CONFIG, observation_budget=96, max_rounds=1, seed=491)
    library = KnowledgeLibrary()
    with LinearMeterLab(2, 491) as laboratory:
        feature = Program("feature", name="sensor_a")
        cubic = Program("mul", children=(feature, Program("mul", children=(feature, feature))))
        source_id, _ = library.learn(Candidate("cubic_source", cubic, "learn"),
            laboratory.spec_for(("sensor_a",)),
            [Observation({"u": 0., "sensor_a": i / 12}, (i / 12) ** 3) for i in range(-12, 13)])
        if route == "measured_stage":
            # Exercise the stage executor after the observation controller
            # declines a visit, using actual acquired instrument trajectories.
            monkeypatch.setattr("hegel_machine.cognition.observation_research.research_observation_actions",
                lambda *args, **kwargs: {"handled": False, "selected": None})
        result = UnifiedResearchAgent(config, measurement_budget=384, library=library).run(laboratory)
        assert result["status"] == "completed"
        assert result["selected_kind"] == "callable_transfer"
        program = Program.from_dict(result["program"])
        assert not _sources(program)
        assert not result["used_sources"]
        assert not any(event["accepted"] for event in result["transfer"])
        if route == "observation":
            assert any(event.get("kind") == "frozen_recalled_response_counterexample"
                and event["source_ids"] == [source_id]
                and event["validation_loss"] > config.minimum_improvement
                for event in result["transfer"])
        assert result["confirmation_loss"] <= config.minimum_improvement
        fresh = laboratory.acquire(Probe("held_out_linear_meter",
            tuple({"u": (i - 16) / 16} for i in range(32))), result["selected_channels"]).observations
        assert loss(program, fresh, result["fallback"], .5) <= config.minimum_improvement
        confirmation = "confirm_observation_research" if route == "observation" else "validate_frozen_program"
        assert any(action["action"] == confirmation for action in result["actions"])
