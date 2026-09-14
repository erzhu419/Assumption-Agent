"""Integration checks for discovery, experiment costs and revisable memory."""

from dataclasses import replace

import pytest

from hegel_machine.cognition import experiments
from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.evaluation import loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import (
    Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec,
)
from hegel_machine.cognition.probing import choose_probe


def _threshold(op, value):
    return Program(op, children=(Program("feature", name="x"), Program("const", value=value)))


class _LinearEnvironment:
    """Small exact tasks isolate runtime behavior from discovery difficulty."""

    def __init__(self, task_id, slope=1.0, phase="acquire", evaluation_sign=1.0):
        initial = tuple(
            Probe(f"initial_{index}", tuple({"x": x, "z": z} for x, z in points))
            for index, points in enumerate((
                ((-1.0, 0.0), (0.0, 1.0), (1.0, -1.0)),
                ((-0.5, 0.5), (0.5, -1.0), (1.5, 0.0)),
                ((-1.5, -0.5), (0.25, 1.0), (0.75, 0.25)),
            ))
        )
        available = tuple(
            Probe(f"query_{index}", tuple({"x": x, "z": 0.0} for x in xs))
            for index, xs in enumerate(((-2.0, 0.0, 2.0), (-1.25, 0.1, 1.25)))
        )
        self.spec = TaskSpec(task_id, "Predict the next reading.", ("x", "z"), initial, available)
        self.evaluation_metadata = {"phase": phase, "mechanism": "test_linear", "related_task": ""}
        self.slope = slope
        self.evaluation_sign = evaluation_sign
        self.observed_cost = 0
        self.run_finished = False

    def observe(self, probe):
        self.observed_cost += probe.cost
        return [Observation(dict(features), self.slope * features["x"]) for features in probe.sequence]

    def evaluation_rows(self):
        assert self.run_finished, "evaluation was accessed before the learning episode finished"
        return tuple(
            Observation({"x": x, "z": 0.0}, self.slope * x * self.evaluation_sign)
            for x in (-2.5, -0.8, 0.8, 2.5)
        )


def _small_config(**overrides):
    return replace(
        ResearchConfig(observation_budget=24, compute_budget=6000, max_rounds=2),
        **overrides,
    )


def test_three_boolean_hypotheses_choose_a_probe_that_splits_the_explanations():
    candidates = [
        Candidate("a", _threshold("ge", 0.5), "relation"),
        Candidate("b", _threshold("ge", 1.0), "relation"),
        Candidate("c", _threshold("lt", -0.5), "relation"),
    ]
    uninformative = Probe("same_answer", ({"x": 0.0},))
    informative = Probe("different_answers", ({"x": 0.75},))
    budget = SearchBudget(6)
    selected, disagreement = choose_probe([uninformative, informative], candidates, 0.0, budget)
    assert selected == informative
    assert disagreement > 0
    assert budget.spent == 6


def test_retrieved_knowledge_executes_and_counterexamples_restrict_then_revoke(monkeypatch):
    library = KnowledgeLibrary()
    known = Program("feature", name="x")
    concept_id, _ = library.learn(Candidate("learned", known, "refit"), _LinearEnvironment("source").spec)
    called = []
    original_predict = Program.predict

    def traced_predict(program, observation, context=None):
        if program is known:
            called.append(observation.features["x"])
        return original_predict(program, observation, context=context)

    monkeypatch.setattr(Program, "predict", traced_predict)
    agent = CognitiveAgent(_small_config(strategy="fixed"), library)
    reused = agent.run(_LinearEnvironment("related", phase="transfer"))
    assert called
    assert reused.selected_memory_id == concept_id
    assert any(action.action == "reuse_check" for action in reused.actions)
    assert "related" in library.concepts[0].successful_tasks

    conflict_one = _LinearEnvironment("conflict_one", slope=-1.0, phase="conflict")
    first = agent.run(conflict_one)
    assert any(event["action"] == "restricted" and event["concept_id"] == concept_id
               for event in first.memory_events)
    assert all(candidate.memory_id != concept_id for candidate in library.retrieve(conflict_one.spec))
    unrelated = _LinearEnvironment("not_yet_tested").spec
    assert any(candidate.memory_id == concept_id for candidate in library.retrieve(unrelated))

    second = agent.run(_LinearEnvironment("conflict_two", slope=2.0, phase="conflict"))
    assert any(event["action"] == "restricted" and event["concept_id"] == concept_id
               for event in second.memory_events)
    assert any(candidate.memory_id == concept_id for candidate in library.retrieve(unrelated))
    # Failing on an unrelated task narrows scope. A new counterexample inside
    # a previously successful scope can invalidate the stored concrete program.
    supported_scope = agent.run(_LinearEnvironment("related", slope=-1.0, phase="conflict"))
    assert any(event["action"] == "revoked" and event["concept_id"] == concept_id
               for event in supported_scope.memory_events)
    assert all(candidate.memory_id != concept_id for candidate in library.retrieve(unrelated))
    assert len(library.concepts[0].counterexamples) == 3


def test_controller_obeys_cost_limits_and_never_requests_final_evaluation():
    underlying = make_suite()[1]

    class LearningOnly:
        spec = underlying.spec
        observed_cost = 0

        def observe(self, probe):
            self.observed_cost += probe.cost
            return underlying.observe(probe)

        def evaluation_rows(self):
            raise AssertionError("the controller cannot inspect final evaluation outcomes")

    environment = LearningOnly()
    config = _small_config(observation_budget=25, compute_budget=2500, max_rounds=3)
    result = CognitiveAgent(config).run(environment)
    assert result.observation_cost == environment.observed_cost == len(result.observations)
    assert result.observation_cost <= config.observation_budget
    assert result.compute_cost <= config.compute_budget
    assert sum(action.observation_cost for action in result.actions) == result.observation_cost
    assert sum(action.compute_cost for action in result.actions) <= result.compute_cost


def test_insufficient_compute_budget_stops_before_unfunded_scoring(monkeypatch):
    scored = []

    def traced_loss(*args, **kwargs):
        scored.append(True)
        return loss(*args, **kwargs)

    monkeypatch.setattr("hegel_machine.cognition.controller.loss", traced_loss)
    with pytest.raises(ValueError, match="compute budget"):
        CognitiveAgent(_small_config(compute_budget=1)).run(_LinearEnvironment("small_budget"))
    assert not scored


def test_zero_and_insufficient_history_rows_stay_in_the_loss_denominator():
    program = Program("lag", name="x", lag=2)
    rows = [
        Observation({"x": 0.0}, 3.0),
        Observation({"x": 1.0}, 5.0, ({"x": 0.0, "__target__": 3.0},)),
        Observation({"x": 2.0}, 7.0, (
            {"x": 7.0, "__target__": 3.0}, {"x": 1.0, "__target__": 5.0},
        )),
    ]
    assert [predict(program, row, fallback=1.0) for row in rows] == [1.0, 1.0, 7.0]
    assert loss(program, rows, fallback=1.0, q=0.5) == pytest.approx(20.0 / 3.0)


def _install_short_suite(monkeypatch, evaluation_sign=1.0):
    def suite(seed, partition="development"):
        return [
            _LinearEnvironment("learn", slope=1.0, evaluation_sign=evaluation_sign),
            _LinearEnvironment("reuse", slope=1.0, phase="transfer", evaluation_sign=evaluation_sign),
            _LinearEnvironment("revise", slope=-1.0, phase="conflict", evaluation_sign=evaluation_sign),
        ]

    monkeypatch.setattr(experiments, "make_suite", suite)
    return suite


def _mark_learning_finished(monkeypatch):
    original_run = CognitiveAgent.run

    def finished_run(agent, environment):
        result = original_run(agent, environment)
        environment.run_finished = True
        return result

    monkeypatch.setattr(CognitiveAgent, "run", finished_run)


def test_cross_task_sequence_performs_executable_revision_and_scores_after_learning(monkeypatch):
    _install_short_suite(monkeypatch)
    _mark_learning_finished(monkeypatch)
    result = experiments.run_sequence(_small_config(strategy="fixed"))
    assert result["task_count"] == 3
    assert result["evaluation_rows"] == 12
    revisions = [action for episode in result["episodes"] for action in episode["actions"]
                 if action["action"] == "revise"]
    assert revisions
    assert all(action["details"]["before"] != action["details"]["after"] for action in revisions)
    assert result["direct_reuse_tasks"] >= 1
    assert result["memory_revisions"] >= 1


def test_final_evaluation_outcomes_do_not_change_knowledge_or_learning_actions(monkeypatch):
    _mark_learning_finished(monkeypatch)
    _install_short_suite(monkeypatch, evaluation_sign=1.0)
    first = experiments.run_sequence(_small_config(strategy="fixed"))
    _install_short_suite(monkeypatch, evaluation_sign=-1.0)
    changed_outcomes = experiments.run_sequence(_small_config(strategy="fixed"))
    assert first["knowledge"] == changed_outcomes["knowledge"]
    assert [episode["actions"] for episode in first["episodes"]] == [
        episode["actions"] for episode in changed_outcomes["episodes"]
    ]
    assert first["metrics_by_loss"] != changed_outcomes["metrics_by_loss"]


def test_new_evidence_prevents_reusing_a_refuted_diagonal_explanation():
    class Interaction:
        initial = tuple(Probe(f"initial_{i}", tuple({"x": x, "z": x} for x in points))
                        for i, points in enumerate(((-1.0, 0.0, 1.0), (-0.5, 0.5, 1.5), (-1.5, 0.25, 0.75))))
        spec = TaskSpec("interaction", "Predict measured response", ("x", "z"), initial, (),
                        feature_bounds={"x": (-2.0, 2.0), "z": (-2.0, 2.0)})

        def observe(self, probe):
            return [Observation(point, -0.2 + 1.1 * point["x"] * point["z"]) for point in probe.sequence]

    x = Program("feature", name="x")
    square = Program("mul", children=(x, x))
    known = Program("add", children=(Program("const", value=-0.2),
                    Program("mul", children=(Program("const", value=1.1), square))))
    library = KnowledgeLibrary()
    library.learn(Candidate("diagonal", known, "composition"), _LinearEnvironment("old").spec)
    result = CognitiveAgent(ResearchConfig(), library).run(Interaction())
    assert any(a.action == "reuse_check" and not a.details["accepted"] for a in result.actions)
    assert result.program.predict(Observation({"x": 1.0, "z": -2.0}, -2.4)) == pytest.approx(-2.4, abs=1e-5)


def test_risk_decision_selection_gets_budget_after_program_search():
    environment = make_suite()[3]
    config = _small_config(compute_budget=4000, observation_budget=48, max_rounds=4)
    result = CognitiveAgent(config).run(environment)
    assert result.uncertainty_model["status"] == "measured_selection"
    assert len(result.uncertainty_model["alternatives"]) >= 2
    assert result.uncertainty_model["fixed_costs"] == list(environment.spec.decision_costs)
    assert result.validation_rows > 0
    assert result.compute_cost <= config.compute_budget
