"""The ordinary research controller must execute and credit continuous states."""

from dataclasses import replace
import json
from random import Random

from hegel_machine.cognition.action_policy import ResearchActionPolicy
from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.measurement_research import InstrumentedEnvironment
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig
from hegel_machine.cognition.workspace import ResearchWorkspace


def measured_task():
    laboratory = make_lab(1, seed=916)
    environment = InstrumentedEnvironment(laboratory, ("sensor_a",))
    rng = Random(814)
    probes = tuple(Probe(f"initial_{index}", tuple({"u": rng.uniform(-1., 1.)} for _ in range(length)))
                   for index, length in enumerate((11, 19, 23)))
    environment.spec = replace(environment.spec, initial_probes=probes)
    return laboratory, environment


def prefer_continuous(library, spec):
    policy = ResearchActionPolicy([Observation({}, 0.), Observation({}, .3), Observation({}, .7)],
                                  spec, library.action_experience)
    for condition in ("model_residual", "new_counterexample", "transfer_counterexample", "adequate"):
        policy.set_condition(condition)
        for action in ("transition", "task_value", "representative", "refit", "representation", "continuous_state"):
            policy.update(action, before_loss=1., after_loss=.1 if action == "continuous_state" else 1.2,
                          observation_cost=8, compute_cost=1000)


def test_learned_action_returns_trigger_continuous_search_in_ordinary_controller():
    laboratory, environment = measured_task()
    library = KnowledgeLibrary()
    prefer_continuous(library, environment.spec)
    config = ResearchConfig(seed=82, max_rounds=3, compute_budget=60000, observation_budget=128)
    result = CognitiveAgent(config, library).run(environment)
    planning = next(action for action in result.actions if action.action == "plan_experiment"
                    and action.details.get("action_policy", {}).get("selected_action") == "continuous_state")
    execution = next(action for action in result.actions if action.action == "continuous_state")
    feedback = next(action for action in result.actions if action.action == "action_feedback"
                    and action.details["action"] == "continuous_state")
    assert execution.round > planning.round
    assert execution.details["allocated_compute"] >= 250 * execution.details["training_rows"]
    assert execution.compute_cost > 0 and execution.details["candidates"] > 0
    assert feedback.details["last_after_loss"] < feedback.details["last_before_loss"]
    assert feedback.details["mean_compute_cost"] * feedback.details["trials"] - 1000 >= execution.compute_cost
    assert feedback.details["reference_rows"] > 0
    assert any(plan["plan_id"] == "continuous_state" for plan in result.plans)
    fresh = laboratory.evaluation_rows(("sensor_a",), seed=75, length=64, trajectories=3)
    assert loss(result.program, fresh, result.fallback, .5) < 1e-6
    assert "accumulate" in result.program.describe()
    assert result.compute_cost <= config.compute_budget
    assert result.observation_cost == laboratory.observation_cost <= config.observation_budget
    laboratory.close()


def test_restored_pending_state_action_runs_before_refit_and_preserves_champion():
    laboratory, environment = measured_task()
    observations = [row for probe in environment.spec.initial_probes for row in environment.observe(probe)]
    fallback = initial_prediction(observations, .5)
    initial = Candidate("saved", Program("const", value=fallback), "refit")
    workspace = ResearchWorkspace(environment.spec.task_id, observations=observations,
                                  incumbent=initial, evidence_champion=initial, fallback=fallback,
                                  initial_loss=loss(initial.program, observations, fallback, .5))
    workspace.queried = list(environment.spec.initial_probes)
    workspace.pending_actions = [{"name": name, "context": "squared:numeric:multiple:model_residual",
        "program": initial.program.to_dict(), "compute_cost": 13, "observation_cost": 0}
        for name in ("refit", "continuous_state")]
    config = ResearchConfig(seed=82, max_rounds=2, compute_budget=40000, observation_budget=32)
    saved = json.loads(json.dumps(workspace.to_dict()))
    result = CognitiveAgent(config).run(environment, research_state=saved)
    searches = [action for action in result.actions if action.action in ("refit", "composition", "continuous_state")]
    assert searches[0].action == "continuous_state"
    assert any(action.action == "action_feedback" and action.details["action"] == "continuous_state"
               for action in result.actions)
    assert result.research_state["evidence_champion"] is not None
    fresh = laboratory.evaluation_rows(("sensor_a",), seed=75, length=64, trajectories=3)
    assert loss(result.program, fresh, result.fallback, .5) < 1e-6
    assert all(action.details.get("probe_id") not in {probe.probe_id for probe in environment.spec.initial_probes}
               for action in result.actions if action.action == "observe")
    laboratory.close()
