"""Shared research choices compare observation and mechanism changes on real data."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.dynamic_instrument_lab import DynamicInstrumentLab
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.instrument_models import instrument_binding
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


CONFIG = ResearchConfig(observation_budget=384, compute_budget=360000,
    minimum_improvement=1e-7, max_history=4, max_rounds=5, seed=113, active_probes=True)


class ActualPurchases:
    """The controller's ordinary acquisition interface, with actual shared limits."""

    def __init__(self, laboratory, budget, actions, *, measurement_budget=1536, observation_budget=384):
        self.laboratory, self.budget, self.actions = laboratory, budget, actions
        self.measurement_budget, self.observation_budget = measurement_budget, observation_budget
        self.batches, self.rng = [], Random(183)

    def remaining(self):
        return {"observation_remaining": self.observation_budget - self.laboratory.costs["observation_cost"],
                "measurement_remaining": self.measurement_budget - self.laboratory.costs["measurement_cost"],
                "compute_remaining": self.budget.remaining}

    def acquire(self, channels, purpose, candidates=(), *, rows_count=None, probe=None, **kwargs):
        length = rows_count or 24
        if probe is None:
            probe = Probe(f"paid:{len(self.batches)}", tuple(
                {name: self.rng.uniform(*bounds) for name, bounds in self.laboratory.spec.feature_bounds.items()}
                for _ in range(length)))
        rate = sum(channel.cost_per_row for channel in self.laboratory.channels if channel.name in channels)
        remaining = self.remaining()
        if probe.cost > remaining["observation_remaining"] or probe.cost * rate > remaining["measurement_remaining"]:
            return None
        batch = self.laboratory.acquire(probe, channels)
        history, pairs = [], []
        for point, measured in zip(probe.sequence, batch.observations):
            pairs.append((Observation(point, measured.target, tuple(history)), measured))
            history.append({**point, "__target__": measured.target})
        self.batches.append({"probe": probe, "channels": tuple(channels), "pairs": pairs, "purpose": purpose})
        self.actions.append({"action": "acquire", "purpose": purpose, "probe_id": probe.probe_id,
            "channels": list(channels), "observation_cost": batch.observation_cost,
            "measurement_cost": batch.measurement_cost})
        return batch.observations

    def paired_for(self, channels):
        return [batch["pairs"] for batch in self.batches if set(channels) <= set(batch["channels"])]

    def command_batches(self):
        return [[command for command, _ in batch["pairs"]] for batch in self.batches]


def restore(payload):
    agent = UnifiedResearchAgent(CONFIG, measurement_budget=1536,
        library=KnowledgeLibrary.from_dict(json.loads(json.dumps(payload))))
    agent._learn_instruments = agent._revise_inputs = True
    return agent


def learn_source(index):
    library = KnowledgeLibrary()
    with DynamicInstrumentLab(index, 113, 113, "confirmation") as laboratory:
        report = UnifiedResearchAgent(CONFIG, measurement_budget=1536, library=library).run(laboratory)
    assert report["status"] == "completed" and report["confirmation_loss"] <= 1e-7
    assert not report["selected_channels"]
    return {"library": library.to_dict(), "program": report["program"]}


@pytest.fixture(scope="module")
def source():
    return learn_source(0)


@pytest.fixture(scope="module")
def source_second_instrument():
    return learn_source(2)


def run_controller(source, index, *, measurement_budget=1536, paired_pilot=False):
    from hegel_machine.cognition.observation_research import research_observation_actions

    agent = restore(source["library"])
    budget, actions = SearchBudget(360000), []
    laboratory = DynamicInstrumentLab(index, 881, 113, "confirmation")
    paid = ActualPurchases(laboratory, budget, actions, measurement_budget=measurement_budget)
    for _ in range(2):
        paid.acquire(("meter",) if paired_pilot else (), "current_pilot", rows_count=24)
    program = Program.from_dict(source["program"])
    parent = {"candidate": Candidate("retained_source", program, "retained"), "kind": "retained_source"}
    before = deepcopy(agent.library.to_dict())
    result = research_observation_actions(agent, laboratory, acquire=paid.acquire,
        paired_for=paid.paired_for, command_batches=paid.command_batches, budget=budget,
        actions=actions, transfer=[], graph=ResearchGraph(), remaining=paid.remaining,
        length=24, parent=parent, failed_probe=paid.batches[-1]["probe"])
    return agent, laboratory, paid, actions, result, before


def assert_fresh_solution(laboratory, result):
    assert result["confirmed"] and result["selected"] is not None
    program = result["selected"]["candidate"].program
    fresh = laboratory.evaluation_rows(tuple(result["channels"]), seed=2911, length=48, trajectories=2)
    assert loss(program, fresh, result["fallback"], .5) <= 1e-7


def chosen(actions):
    return [item for item in actions if item["action"] == "choose_observation_research"]


def assert_common_response_feedback(paid, actions):
    confirmations = [item for item in actions if item["action"] == "confirm_observation_research"]
    purchased = [batch for batch in paid.batches if batch["purpose"] == "confirm_observation_research"]
    assert confirmations and len(confirmations) == len(purchased)
    for event, batch in zip(confirmations, purchased):
        rows = [measured for _, measured in batch["pairs"]]
        before = loss(Program.from_dict(event["baseline_program"]), rows, event["fallback"], .5)
        after = loss(Program.from_dict(event["program"]), rows, event["fallback"], .5)
        assert event["before_loss"] == pytest.approx(before)
        assert event["after_loss"] == pytest.approx(after)
        assert event["evidence_basis"] == "fresh_common_response_trajectory"
        assert event["observation_rows"] == len(rows)
    feedback = [item for item in actions if item["action"] == "observation_research_feedback"]
    assert len(feedback) == len(confirmations)
    for event, confirmed in zip(feedback, confirmations):
        assert event["before_loss"] == confirmed["before_loss"]
        assert event["after_loss"] == confirmed["after_loss"]


def test_instrument_change_competes_with_mechanism_revision_and_keeps_the_learned_response(source):
    agent, laboratory, paid, actions, result, before = run_controller(source, 2)
    assert_fresh_solution(laboratory, result)
    choices = chosen(actions)
    assert choices
    assert any({"revise_mechanism", "remeasure_inputs"} <= set(item["eligible_actions"]) for item in choices)
    # A control-history revision may already explain this changed apparatus;
    # buying meters is required only if the chosen hypothesis needs them.
    assert {item["selected_action"] for item in choices} <= {
        "revise_mechanism", "remeasure_inputs", "discriminate_inputs"}
    assert agent.library.to_dict()["definitions"] == before["definitions"]
    assert_common_response_feedback(paid, actions)
    restored = restore(agent.library.to_dict())
    assert restored.library.action_experience == agent.library.action_experience


def test_response_change_does_not_revoke_the_instrument_supported_by_actual_fields(source_second_instrument):
    agent, laboratory, paid, actions, result, before = run_controller(source_second_instrument, 3, paired_pilot=True)
    assert_fresh_solution(laboratory, result)
    assert_common_response_feedback(paid, actions)
    old_models = {item["model_id"]: item for item in before["instrument_models"]}
    assert not any(item["action"] == "instrument_mapping_counterexample" for item in actions)
    for model in agent.library.instrument_models:
        if model["model_id"] in old_models:
            assert model["status"] != "counterexample"
    # Field agreement is independently checked on these paid target readings.
    binding = agent._instrument_binding(laboratory.spec_for(("meter",)), laboratory.spec)
    assert "meter" in binding
    assert max(abs(binding["meter"].predict(command) - measured.features["meter"])
               for batch in paid.paired_for(("meter",)) for command, measured in batch) < 1e-7


def test_joint_change_requires_current_response_success_after_any_input_repair(source):
    _, laboratory, paid, actions, result, _ = run_controller(source, 3)
    assert_fresh_solution(laboratory, result)
    assert_common_response_feedback(paid, actions)
    assert result["confirmation_rows"] > 0 and result["confirmation_loss"] <= 1e-7


def test_no_meter_budget_still_allows_real_mechanism_revision(source_second_instrument):
    agent, laboratory, paid, actions, result, before = run_controller(source_second_instrument, 3,
                                                                    measurement_budget=0)
    assert_fresh_solution(laboratory, result)
    assert laboratory.costs["measurement_cost"] == 0
    assert chosen(actions) and all(item["selected_action"] == "revise_mechanism" for item in chosen(actions))
    assert all("remeasure_inputs" not in item["eligible_actions"] for item in chosen(actions))
    assert agent.library.instrument_models == before["instrument_models"]


def test_saved_instrument_ambiguity_is_priced_and_chosen_alongside_other_research_actions():
    from hegel_machine.cognition.ambiguous_instrument_lab import AmbiguousInstrumentLab
    from hegel_machine.cognition.observation_research import research_observation_actions

    agent = UnifiedResearchAgent(CONFIG)
    with AmbiguousInstrumentLab(0, 157, 739, "recurrent") as laboratory:
        compute, actions = SearchBudget(120000), []
        paid = ActualPurchases(laboratory, compute, actions)
        for _ in range(7):
            paid.acquire(("meter",), "fit_source_instrument", rows_count=3)
        model = agent._fit_instruments(laboratory.spec, laboratory.spec_for(("meter",)),
            paid.paired_for(("meter",)), compute, actions)
        assert model["status"] == "ambiguous_on_measured_trajectory"
    agent = restore(agent.library.to_dict())
    with AmbiguousInstrumentLab(1, 881, 739, "recurrent") as laboratory:
        compute, actions = SearchBudget(120000), []
        paid = ActualPurchases(laboratory, compute, actions, observation_budget=128, measurement_budget=384)
        research_observation_actions(agent, laboratory, acquire=paid.acquire, paired_for=paid.paired_for,
            command_batches=paid.command_batches, budget=compute, actions=actions, transfer=[],
            graph=ResearchGraph(), remaining=paid.remaining, length=8)
        first = chosen(actions)[0]
        assert set(first["eligible_actions"]) == {"revise_mechanism", "remeasure_inputs", "discriminate_inputs"}
        assert any(item["selected_action"] == "discriminate_inputs" for item in chosen(actions))
        purchase = next(batch for batch in paid.batches if batch["purpose"] == "distinguish_instrument_hypotheses")
        assert purchase["probe"].cost == 4
        checked = next(item for item in actions if item["action"] == "test_instrument_hypotheses")
        assert checked["evidence_role"] == "observation_fields_only"
        model = agent.library.instrument_models[0]
        assert model["status"] == "supported_on_measured_trajectory"
        assert "meter" in instrument_binding(model)
        assert {item["kind"] for item in model["fields"]["meter"]["hypotheses"]
                if item["status"] != "refuted"} == {"continuous_state"}
        # An instrument experiment's response reward, if available, must come
        # from another actual frozen-response trajectory rather than field loss.
        if any(item["action"] == "observation_research_feedback" for item in actions):
            assert_common_response_feedback(paid, actions)
