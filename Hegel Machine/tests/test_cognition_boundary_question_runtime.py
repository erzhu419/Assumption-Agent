"""Ordinary goal confirmation must retain the current noise explanation."""

from copy import deepcopy

from test_cognition_boundary_noise_runtime import boundary_runtime
from test_cognition_sc_kg_environment import project_root


def test_noise_evidence_waits_for_a_constraint_question(boundary_runtime, monkeypatch):
    algorithm, adapter, _, _ = boundary_runtime
    goal = algorithm.objective_research
    goal.state["compute_cost"] = goal.state["compute_budget"] - 6000
    evidence = algorithm._boundary_noise_evidence(4)
    assert evidence is not None
    monkeypatch.setattr(algorithm, "_boundary_noise_evidence", lambda remaining: deepcopy(evidence))
    offer = {"features": {"x1": .22, "x2": .59, "x3": .34},
             "reason": "measured_terminal_incumbent", "purpose": "objective_confirmation"}
    monkeypatch.setattr(goal, "terminal_offers", lambda *args, **kwargs: [deepcopy(offer)])
    original = algorithm.variance_model
    algorithm._generate_candidates(28)
    assert algorithm._terminal_noise_evidence is None
    assert algorithm.variance_model is original
    assert not algorithm.objective_diagnostics["boundary_noise"]["models"]

    # The same evidence becomes relevant when the agenda poses an actual
    # constraint question. No observation-count or fitted-scale cutoff decides.
    offer["reason"] = "research_promising_constraint_certificate"
    algorithm._generate_candidates(28)
    installed = algorithm._terminal_noise_evidence
    assert installed["research_questions"] == [
        {"reason": offer["reason"], "features": offer["features"]}]
    assert installed["scales"] == evidence["scales"]
    assert len(algorithm.history) == 60 and adapter.simulator_calls == 0
