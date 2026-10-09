"""Unresolved paid observation models resume through the ordinary research loop."""

import json
from random import Random

import pytest

from hegel_machine.cognition.ambiguous_instrument_lab import AmbiguousInstrumentLab
from hegel_machine.cognition.instrument_models import instrument_binding
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def measured_batch(laboratory, random, index):
    probe = Probe(f"calibration_{index}", tuple({"u": random.uniform(-1., 1.)} for _ in range(3)))
    readings = laboratory.acquire(probe, ("meter",)).observations
    history, pairs = [], []
    for controls, measured in zip(probe.sequence, readings):
        pairs.append((Observation(controls, measured.target, tuple(history)), measured))
        history.append({**controls, "__target__": measured.target})
    return pairs


def unresolved(mode):
    library = KnowledgeLibrary()
    agent = UnifiedResearchAgent(ResearchConfig(max_history=4), library=library)
    with AmbiguousInstrumentLab(0, 157, 739, mode) as source:
        random = Random(157)
        batches = [measured_batch(source, random, index) for index in range(7)]
        model = agent._fit_instruments(source.spec, source.spec_for(("meter",)),
                                       batches, SearchBudget(60000), [])
    assert model["status"] == "ambiguous_on_measured_trajectory"
    assert "meter" not in instrument_binding(model)
    return KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))


@pytest.mark.parametrize("mode,expected", [("finite2", "finite_history_polynomial"),
                                           ("recurrent", "continuous_state")])
def test_longer_public_horizon_resumes_saved_ambiguity_with_a_real_discriminating_purchase(mode, expected):
    library = unresolved(mode)
    agent = UnifiedResearchAgent(ResearchConfig(seed=881, observation_budget=96,
        compute_budget=120000, max_history=4, max_rounds=3, minimum_improvement=1e-7,
        active_probes=True), measurement_budget=384, library=library)
    with AmbiguousInstrumentLab(1, 881, 739, mode) as target:
        report = agent.run(target)
        purchases = [action for action in report["actions"] if action["action"] == "acquire"
                     and action["purpose"] == "distinguish_instrument_hypotheses"]
        assert purchases and purchases[0]["observation_cost"] >= 4
        assert purchases[0]["measurement_cost"] == 3 * purchases[0]["observation_cost"]
        assert report["costs"]["observation_cost"] == target.costs["observation_cost"]
        assert report["costs"]["measurement_cost"] == target.costs["measurement_cost"]
    candidates = library.instrument_models[0]["fields"]["meter"]["hypotheses"]
    assert {item["kind"] for item in candidates if item["status"] != "refuted"} == {expected}
    assert any(item["status"] == "refuted" for item in candidates)
    assert any(action["action"] == "test_instrument_hypotheses" for action in report["actions"])


def test_no_measurement_budget_keeps_unresolved_models_across_run_and_json_restore():
    library = unresolved("recurrent")
    agent = UnifiedResearchAgent(ResearchConfig(seed=881, observation_budget=48,
        compute_budget=30000, max_history=4, max_rounds=2, minimum_improvement=1e-7),
        measurement_budget=0, library=library)
    with AmbiguousInstrumentLab(1, 881, 739, "recurrent") as target:
        report = agent.run(target)
        assert target.costs["measurement_cost"] == 0
    assert not any(action["action"] == "test_instrument_hypotheses" for action in report["actions"])
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert restored.instrument_models[0]["status"] == "ambiguous_on_measured_trajectory"
    assert "meter" not in instrument_binding(restored.instrument_models[0])
