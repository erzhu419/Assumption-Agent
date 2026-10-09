"""Observation experiment procedures transfer through actual purchased readings."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.instrument_method_lab import InstrumentMethodLab
from hegel_machine.cognition.instrument_models import instrument_binding
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


class Purchased:
    def __init__(self, laboratory, batches=(), *, available=True):
        self.laboratory, self.batches = laboratory, list(batches)
        self.available, self.purchases = available, []

    def acquire(self, channels, purpose, *, probe):
        if not self.available:
            return None
        batch = self.laboratory.acquire(probe, channels)
        pairs = [(Observation(point, measured.target, tuple(probe.sequence[:index])), measured)
                 for index, (point, measured) in enumerate(zip(probe.sequence, batch.observations))]
        self.batches.append(pairs)
        self.purchases.append({"purpose": purpose, "probe": probe, "batch": batch})
        return batch.observations

    def paired_for(self, channels):
        assert channels == ("meter",)
        return self.batches


def agent_for(payload=None):
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(payload))) if payload else KnowledgeLibrary()
    return UnifiedResearchAgent(ResearchConfig(seed=157, max_history=4, minimum_improvement=1e-7),
                                library=library)


def fit_short(agent, laboratory, length):
    paid, random = Purchased(laboratory), Random(157)
    for index in range(7):
        probe = Probe(f"paid_short:{index}", tuple({"u": random.uniform(-1., 1.)} for _ in range(length)))
        paid.acquire(("meter",), "fit_instrument", probe=probe)
    model = agent._fit_instruments(laboratory.spec, laboratory.spec_for(("meter",)),
                                   paid.batches, SearchBudget(120000), [])
    assert model["status"] == "ambiguous_on_measured_trajectory"
    assert "meter" not in instrument_binding(model)
    return model, paid.batches


def distinguish(agent, laboratory, model, batches=(), *, budget=30000, available=True):
    paid, actions, compute = Purchased(laboratory, batches, available=available), [], SearchBudget(budget)
    result = agent._distinguish_instruments(laboratory, model, ("meter",), paid.acquire,
                                           paid.paired_for, compute, actions)
    return result, paid, actions, compute


def methods(agent):
    return [item for item in agent.library.program_experiment_methods.records
            if item["solver_strategy"] == "instrument_sequence"]


def plans(actions):
    return [item for item in actions if item["action"] == "plan_instrument_disagreement"
            and item.get("probe_id")]


@pytest.fixture(scope="module")
def source():
    agent = agent_for()
    with InstrumentMethodLab(0, 157, 739, "recurrent") as laboratory:
        model, batches = fit_short(agent, laboratory, 3)
    unresolved = deepcopy(model)
    with InstrumentMethodLab(1, 157, 739, "recurrent") as laboratory:
        result, paid, actions, _ = distinguish(agent, laboratory, model, batches)
        assert result["status"] == "supported_on_measured_trajectory"
        assert len(paid.purchases) == 1 and paid.purchases[0]["probe"].cost == 4
        assert any(item["action"] == "learn_instrument_experiment_method" for item in actions)
        assert len(methods(agent)) == 1 and len(methods(agent)[0]["support"]) == 1
        return {"library": agent.library.to_dict(), "unresolved": unresolved,
                "proposal": plans(actions)[0], "probe": paid.purchases[0]["probe"]}


def test_json_recalled_instrument_method_forecasts_current_candidates_before_real_purchase(source):
    agent = agent_for(source["library"])
    identifier = methods(agent)[0]["method_id"]
    with InstrumentMethodLab(0, 881, 1541, "recurrent") as laboratory:
        model, batches = fit_short(agent, laboratory, 3)
    candidates = deepcopy(model["fields"]["meter"]["hypotheses"])
    with InstrumentMethodLab(1, 881, 1541, "recurrent") as laboratory:
        result, paid, actions, compute = distinguish(agent, laboratory, model, batches)
        assert result["status"] == "supported_on_measured_trajectory"
        used = [item for item in actions if item["action"] == "use_instrument_experiment_method"]
        assert used and used[0]["method_id"] == identifier
        proposal, acquired = plans(actions)[0], paid.purchases[0]
        assert proposal["predictions"] != source["proposal"]["predictions"]
        sequence = acquired["probe"].sequence
        for hypothesis in candidates:
            program = Program.from_dict(hypothesis["program"])
            predicted = [program.predict(Observation(point, 0., tuple(sequence[:index])))
                         for index, point in enumerate(sequence)]
            assert proposal["predictions"]["meter"][hypothesis["hypothesis_id"]] == pytest.approx(predicted)
        assert laboratory.costs["observation_cost"] == acquired["probe"].cost == 4
        assert laboratory.costs["measurement_cost"] == 12
        assert 0 < compute.spent <= compute.limit
    restored = agent_for(agent.library.to_dict())
    assert len(methods(restored)) == 1
    assert len(methods(restored)[0]["support"]) == 2


def test_failed_short_procedure_is_extended_and_the_revision_is_supported_by_new_readings(source):
    agent = agent_for(source["library"])
    identifier = methods(agent)[0]["method_id"]
    before = deepcopy(agent.library.program_experiment_methods.to_dict())
    with InstrumentMethodLab(2, 881, 739, "recurrent") as laboratory:
        model, batches = fit_short(agent, laboratory, 4)
        _, paid, actions, _ = distinguish(agent, laboratory, model, batches)
        assert not paid.purchases
        assert not plans(actions)
        assert agent.library.program_experiment_methods.to_dict() == before
    with InstrumentMethodLab(3, 881, 739, "recurrent") as laboratory:
        result, paid, actions, _ = distinguish(agent, laboratory, model, batches)
        assert result["status"] == "supported_on_measured_trajectory"
        assert len(paid.purchases) == 1
        acquired = paid.purchases[0]["probe"]
        assert acquired.cost == 5 > source["probe"].cost
        assert acquired.sequence[:4] == source["probe"].sequence
        assert laboratory.costs["observation_cost"] == 5
        assert laboratory.costs["measurement_cost"] == 15
    revised = [item for item in methods(agent) if item.get("parent_method_id") == identifier]
    assert len(revised) == 1 and len(revised[0]["support"]) == 1
    assert len(next(item for item in methods(agent) if item["method_id"] == identifier)["support"]) == 1
    assert any(item["action"] == "learn_instrument_experiment_method"
               and item.get("parent_method_id") == identifier for item in actions)


def test_real_field_refutation_revises_the_used_method_without_recording_a_success(source):
    agent = agent_for(source["library"])
    identifier = methods(agent)[0]["method_id"]
    # The candidate set itself came from paid source data; it is now tested
    # against another real apparatus without pre-fitting away the mismatch.
    model = deepcopy(source["unresolved"])
    agent.library.instrument_models = [model]
    with InstrumentMethodLab(3, 881, 739, "recurrent") as laboratory:
        result, paid, actions, _ = distinguish(agent, laboratory, model)
        assert len(paid.purchases) == 1
        assert any(item["action"] == "test_instrument_hypotheses"
                   and "meter" in item["assessment"]["refuted_fields"] for item in actions)
        assert any(item["action"] == "revise_instrument_experiment_method"
                   and item["method_id"] == identifier for item in actions)
        assert result is None
    method = next(item for item in methods(agent) if item["method_id"] == identifier)
    assert len(method["support"]) == 1
    assert method["counterexamples"] and method["failure_scopes"]
    assert not any(item["action"] in {"learn_instrument_experiment_method", "use_instrument_experiment_method"}
                   for item in actions)


def test_unpaid_or_unfunded_method_attempts_never_become_new_support_after_json(source):
    agent = agent_for(source["library"])
    model = deepcopy(source["unresolved"])
    before = deepcopy(agent.library.program_experiment_methods.to_dict())
    with InstrumentMethodLab(1, 881, 739, "recurrent") as laboratory:
        for budget, available in ((0, True), (30000, False)):
            result, paid, actions, _ = distinguish(agent, laboratory, model, budget=budget, available=available)
            assert result["status"] == "ambiguous_on_measured_trajectory"
            assert not paid.purchases
            assert not any(item["action"] in {"learn_instrument_experiment_method",
                           "use_instrument_experiment_method", "revise_instrument_experiment_method"}
                           for item in actions)
        assert laboratory.costs["observation_cost"] == laboratory.costs["measurement_cost"] == 0
    restored = agent_for(agent.library.to_dict())
    assert restored.library.program_experiment_methods.to_dict() == before
