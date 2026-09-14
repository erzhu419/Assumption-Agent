"""A whole research run must survive task boundaries and use real resources."""

from copy import deepcopy
from dataclasses import replace
import json
from math import sin

from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.session import ResearchSession


class MeasuredTask:
    def __init__(self, task_id, *, slope=1.0, nonlinear=False):
        self.spec = TaskSpec(
            task_id, "Predict this measured response.", ("x",),
            (Probe("initial", tuple({"x": x} for x in (-1., -.4, .3, .8))),), (),
            context={"system_id": "same_apparatus"}, feature_bounds={"x": (-1., 1.)},
            max_probe_length=8)
        self.slope, self.nonlinear = slope, nonlinear
        self.observed = 0

    @property
    def evaluation_metadata(self):
        raise AssertionError("the research scheduler asked for evaluator labels")

    def evaluation_rows(self):
        raise AssertionError("the research scheduler asked for final outcomes")

    def observe(self, probe):
        self.observed += probe.cost
        history, rows = [], []
        for features in probe.sequence:
            x = features["x"]
            target = sin(11 * x) if self.nonlinear else self.slope * x + .25
            rows.append(Observation(dict(features), target, tuple(history)))
            history.append({**features, "__target__": target})
        return rows


def config(**kwargs):
    return replace(ResearchConfig(observation_budget=32, compute_budget=6000,
                                  max_rounds=2, seed=61), **kwargs)


def test_pause_json_load_resume_matches_continuous_learning_and_revises_memory():
    factories = [lambda: MeasuredTask("source"), lambda: MeasuredTask("related"),
                 lambda: MeasuredTask("source", slope=-1.)]
    options = dict(observation_budget=120, compute_budget=24000, max_visits=1)
    whole = ResearchSession(config(), factories, **options).run()
    first = ResearchSession(config(), factories, **options).run(max_steps=1)
    assert first["status"] == "paused" and first["completed_visits"] == 1
    resumed = ResearchSession(config(), factories, state=json.loads(json.dumps(first)), **options).run()
    # Candidate proposal serials are run-local. Resume must preserve learning
    # behavior, the knowledge used next, and actual resource expenditure.
    for key in ("knowledge", "tasks", "total_observation_cost", "total_compute_cost", "scheduler_events"):
        assert json.loads(json.dumps(resumed[key])) == json.loads(json.dumps(whole[key]))
    for expected, actual in zip(whole["episodes"], resumed["episodes"]):
        for key in ("program", "validation_loss", "observation_cost", "compute_cost",
                    "selected_operation", "selected_memory_id", "memory_events", "concepts_added"):
            assert actual[key] == expected[key]
    assert resumed["status"] == "completed" and resumed["prediction_coverage"] == 1.
    assert resumed["episodes"][1]["selected_memory_id"] is not None
    assert resumed["memory_revisions"] >= 1
    assert any(concept["status"] == "revoked" for concept in resumed["knowledge"]["concepts"])


def test_all_tasks_are_seen_before_revisiting_the_unresolved_question():
    factories = [lambda: MeasuredTask("unresolved", nonlinear=True), lambda: MeasuredTask("linear")]
    result = ResearchSession(config(), factories, observation_budget=160,
                             compute_budget=30000, max_visits=2).run()
    assert [row["task_index"] for row in result["episodes"]] == [0, 1, 0]
    revisit = result["scheduler_events"][-1]
    assert revisit["reason"] == "new_knowledge"
    assert any(event["reason"] == "try_new_knowledge" and event["useful_candidates"] > 0
               for event in result["scheduler_events"])
    assert revisit["validation_loss"] > result["config"]["minimum_improvement"]
    assert revisit["query_seed"] != result["scheduler_events"][0]["query_seed"]
    assert result["tasks"][0]["questions"]
    assert result["unresolved_tasks"] == [0]
    assert result["status"] == "completed"


def test_total_observation_budget_stops_work_without_dropping_pending_tasks():
    environments = [MeasuredTask("first", nonlinear=True), MeasuredTask("second", nonlinear=True)]
    result = ResearchSession(config(), environments, observation_budget=32,
                             compute_budget=12000, max_visits=2).run()
    assert result["total_observation_cost"] == sum(env.observed for env in environments)
    assert 0 < result["total_observation_cost"] <= 32
    assert result["total_compute_cost"] <= 12000
    assert result["status"] == "budget_exhausted"
    assert result["task_count"] == 2 and result["pending"]
    assert result["prediction_coverage"] == result["visited_tasks"] / 2


def test_total_compute_budget_is_shared_across_visits():
    result = ResearchSession(config(), [lambda: MeasuredTask("hard", nonlinear=True)],
                             observation_budget=120, compute_budget=3000, max_visits=3).run()
    assert 0 < result["total_compute_cost"] <= 3000
    assert result["total_compute_cost"] == sum(row["compute_cost"] for row in result["episodes"]) + result.get("evaluator_compute_cost", 0)
    assert result["total_observation_cost"] <= 120
    assert all(event["allocated_compute_budget"] <= 3000 for event in result["scheduler_events"])


def test_completed_checkpoint_keeps_knowledge_without_buying_queries():
    environment = MeasuredTask("finished")
    options = dict(observation_budget=48, compute_budget=9000, max_visits=1)
    result = ResearchSession(config(), [environment], **options).run()
    observed = environment.observed
    restored = ResearchSession(config(), [environment], state=deepcopy(result), **options).run()
    assert restored == result
    assert environment.observed == observed


def test_more_resources_continue_an_unresolved_saved_question():
    factories = [lambda: MeasuredTask("hard", nonlinear=True)]
    first = ResearchSession(config(), factories, observation_budget=32,
                            compute_budget=6000, max_visits=1).run()
    assert first["unresolved_tasks"] == [0]
    restored = ResearchSession(config(), factories, observation_budget=96,
                               compute_budget=18000, max_visits=3, state=first).run(max_steps=1)
    assert restored["completed_visits"] == 2
    assert restored["episodes"][0] == first["episodes"][0]
    assert restored["scheduler_events"][-1]["reason"] == "largest_unresolved_relative_loss"
    assert restored["total_observation_cost"] > first["total_observation_cost"]


def test_checkpoint_callback_saves_each_completed_visit():
    saved = []
    result = ResearchSession(config(), [lambda: MeasuredTask("a"), lambda: MeasuredTask("b")],
                             observation_budget=80, compute_budget=16000, max_visits=1).run(on_checkpoint=saved.append)
    assert [checkpoint["completed_visits"] for checkpoint in saved] == [1, 2]
    assert saved[0]["episodes"] == result["episodes"][:1]
    assert saved[-1]["knowledge"] == result["knowledge"]


def test_nonzero_risk_loss_needs_revision_evidence_before_buying_another_visit():
    def risk_task():
        task = MeasuredTask("risk", nonlinear=True)
        task.spec = replace(task.spec, risk_quantile=.8, decision_costs=(4., 1.))
        return task

    options = dict(observation_budget=96, compute_budget=18000, max_visits=2)
    first = ResearchSession(config(), [risk_task], **options).run(max_steps=1)
    assert first["tasks"][0]["unresolved"]
    last = next(action["details"] for action in reversed(first["episodes"][0]["actions"])
                if action["action"] == "compare")
    last.update(evidence_gain=0., dispersion_margin=.02)
    for residual_loss in (.1, 10., 1000.):
        state = deepcopy(first)
        # Isolate the claim that a large residual alone cannot create value.
        # A separately retained fitted decision alternative is an executable
        # question and is covered by the decision-research runtime tests.
        state["workspaces"]["0"]["decision_frontier"] = {}
        state["tasks"][0].update(validation_loss=residual_loss, normalized_excess_loss=residual_loss)
        resumed = ResearchSession(config(), [risk_task], state=state, **options).run()
        assert resumed["completed_visits"] == 1
        assert resumed["total_observation_cost"] == first["total_observation_cost"]
        assert resumed["tasks"][0]["revisit_basis"] == "no_observed_revision_gain"
        assert resumed["unresolved_tasks"] == [0]
    last.update(evidence_gain=.1, dispersion_margin=.02)
    resumed = ResearchSession(config(), [risk_task], state=first, **options).run()
    assert resumed["completed_visits"] == 2
    assert resumed["scheduler_events"][-1]["reason"] == "recent_supported_risk_improvement"
