"""Paid response comparisons, not parent existence, choose a structure seed."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


@pytest.mark.parametrize("prior", ["worse", "better", "tied"])
def test_existing_parent_competes_with_paid_history_response(monkeypatch, prior):
    from hegel_machine.cognition import mechanism_revision, partial_object_knowledge, unified_structure

    q = Program("feature", name="q")
    history_response = Program("window_sum", lag=2, children=(q,))
    response = Program("add", children=(history_response, Program("mul", children=(
        Program("const", value=.5), Program("feature", name="x")))))

    def observations(seed):
        rng, history, rows = Random(seed), [], []
        for _ in range(16):
            features = {"q": rng.uniform(-2., 2.), "u": rng.uniform(-1., 1.),
                        "x": rng.uniform(-2., 2.), "f": 1.}
            row = Observation(features, 0., tuple(history))
            rows.append(Observation(features, response.predict(row), tuple(history)))
            history.append(features)
        return rows

    training, validation = observations(17), observations(39)
    proposed = {"candidate": Candidate("paid_history", history_response, "research_program"),
                "kind": "partial_object:window_2_totals", "family_id": "T03",
                "training_loss": loss(history_response, training, 0., .5)}
    old_program = (Program("const", value=20.) if prior == "worse" else
        history_response if prior == "tied" else
        Program("add", children=(response, Program("const", value=.05))))
    parent = {"candidate": Candidate("old_parent", old_program, "retained_response"),
              "kind": "retained_response", "family_id": "T03",
              "training_loss": 0., "validation_loss": 0.}
    original = deepcopy(parent)
    schema = {"objects": ["object"], "fields": {"object": {
        "q": "q", "u": "u", "x": "x", "f": "f"}}}
    spec = TaskSpec("paid_parent_choice", "test a measured response", tuple(training[0].features), (), (),
        context={"object_schema": json.dumps(schema), "object_model": {"groups": [{"object_id": "object",
            **schema["fields"]["object"]}], "unresolved": []}},
        feature_bounds={name: (-2., 2.) for name in training[0].features})
    agent = UnifiedResearchAgent(ResearchConfig(use_memory=False, learn_research_policy=False,
        state_search=False, max_history=2, minimum_improvement=1e-7))
    received = []

    def capture(spec, rows, budget, *, parent=None, **kwargs):
        received.append(parent)
        return []

    # These fixtures provide candidates, not their comparison scores. The
    # native research view must actually assess both on the same held rows.
    monkeypatch.setattr(partial_object_knowledge, "propose_partial_object_claims",
                        lambda *args, **kwargs: [proposed])
    monkeypatch.setattr(mechanism_revision, "propose_mechanism_revisions", lambda *args, **kwargs: [])
    monkeypatch.setattr(agent, "_dependent_residual_claims", lambda *args, **kwargs: [])
    monkeypatch.setattr(unified_structure, "propose_structures", capture)
    actions = []
    candidates = agent._research_view(spec, training, validation, SearchBudget(40000),
                                      actions, [], None, parent=parent)
    chosen, = received
    assert chosen["candidate"].candidate_id == ("paid_history" if prior == "worse" else "old_parent")
    event, = [item for item in actions if item["action"] == "select_structure_response_parent"]
    assert event["prior_loss"] == pytest.approx(loss(old_program, validation, 0., .5))
    assert event["selected_loss"] == pytest.approx(loss(chosen["candidate"].program, validation, 0., .5))
    assert event["observation_rows"] == len(validation) and event["compute_cost"] == 0
    assert {"old_parent", "paid_history"} <= {item["candidate"].candidate_id for item in candidates}
    assert parent == original
