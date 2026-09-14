"""A paid counterexample can identify an existing transferred lag structure."""

import json
from dataclasses import replace
from pathlib import Path

from hegel_machine.cognition.action_policy import ResearchActionPolicy
from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.experiment_knowledge import research_context
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.method_conditions import method_profile
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.session import ResearchSession
from hegel_machine.cognition.workspace import ResearchWorkspace


FIXTURE = Path(__file__).parent / "fixtures" / "cognition_conflict_transfer_learning.json"


def conflict_learning():
    payload = json.loads(FIXTURE.read_text())

    def spec(record):
        return TaskSpec(**{**record, "feature_names": tuple(record["feature_names"]),
            "initial_probes": tuple(Probe(row["probe_id"], tuple(row["sequence"])) for row in record["initial_probes"]),
            "available_probes": tuple(Probe(row["probe_id"], tuple(row["sequence"])) for row in record["available_probes"])})

    def observations(records):
        return [Observation(row["features"], row["target"], tuple(row["history"])) for row in records]

    return (spec(payload["source_spec"]), spec(payload["target_spec"]),
        Program.from_dict(payload["source_program"]), observations(payload["initial_rows"]),
        observations(payload["first_new_trajectory"]), observations(payload["later_learning_trajectory"]))


def refit(library, identifier, spec, rows):
    budget = SearchBudget(500)
    candidates = library.propose(spec, rows, budget, source_concept_ids={identifier}, preserve_parameters=False)
    best = min(candidates, key=lambda candidate: loss(candidate.program, rows, 0., .5))
    return best, budget


def test_first_discriminating_trajectory_identifies_the_already_available_lag_structure():
    source_spec, target, source, initial, fresh, later = conflict_learning()
    library = KnowledgeLibrary()
    identifier, _ = library.learn(Candidate("learned_source", source, "representation"), source_spec)
    assert len(initial) == 9 and len(fresh) == len(later) == 12
    mature = [row for row in initial if len(row.history) >= 2]
    assert mature and all(row.features["x"] == row.history[-2]["x"] for row in mature)
    ambiguous, first_budget = refit(library, identifier, target, initial)
    assert loss(ambiguous.program, initial, 0., .5) < 1e-12
    assert loss(ambiguous.program, later, 0., .5) > .1

    # The first real representative trajectory already breaks the aliasing.
    # No new measurements or new state operators are necessary to identify it.
    assert any(row.features["x"] != row.history[-2]["x"] for row in fresh if len(row.history) >= 2)
    revised, budget = refit(library, identifier, target, initial + fresh)
    assert revised.memory_id == identifier
    assert identifier in revised.parents
    assert loss(revised.program, initial + fresh, 0., .5) < 1e-12
    assert loss(revised.program, later, 0., .5) < 1e-12
    assert revised.program.history_depth == source.history_depth == 2
    assert first_budget.spent < budget.spent < 200
    assert library.last_search["compute_cost"] == budget.spent
    assert library.concepts[0].program == source
    assert target.task_id not in library.concepts[0].successful_tasks


def test_source_limited_refit_after_json_resume_keeps_the_law_and_actual_work():
    source_spec, target, source, initial, fresh, later = conflict_learning()
    library = KnowledgeLibrary()
    identifier, _ = library.learn(Candidate("learned_source", source, "representation"), source_spec)
    refit(library, identifier, target, initial)
    saved = json.loads(json.dumps(library.to_dict()))
    restored = KnowledgeLibrary.from_dict(saved)
    expected, expected_budget = refit(library, identifier, target, initial + fresh)
    actual, actual_budget = refit(restored, identifier, target, initial + fresh)
    assert actual.program == expected.program
    assert actual.memory_id == expected.memory_id == identifier
    assert actual_budget.spent == expected_budget.spent
    assert loss(actual.program, later, 0., .5) < 1e-12
    assert restored.to_dict() == saved


class ConflictLagEnvironment:
    """Execute new requested controls for the lag relation in the paid fixture."""

    def __init__(self, spec):
        self.spec = spec
        self.observed = []
        self.queries = []

    def observe(self, probe):
        self.queries.append(probe)
        history, rows = [], []
        for features in probe.sequence:
            target = .55 * features["x"] + (1.2 * history[-2]["x"] if len(history) >= 2 else 0.)
            rows.append(Observation(dict(features), target, tuple(history)))
            history.append({**features, "__target__": target})
        self.observed.extend(rows)
        return rows


def pending_continuous(library, spec, workspace, *, condition_name="transfer_counterexample"):
    policy = ResearchActionPolicy(workspace.observations, spec, library.action_experience)
    context = policy.set_condition(condition_name)
    condition = research_context(spec, workspace.observations)
    profile = method_profile(spec, workspace.observations, workspace.incumbent,
        residual_loss=loss(workspace.incumbent.program, workspace.observations, workspace.fallback, .5),
        baseline_loss=workspace.initial_loss)
    method = library.methods.observe(context, [{"action": "continuous_state", "context": context,
        "gain": .9, "compute_cost": 1000, "observation_cost": 12, "profile": profile}],
        condition=condition, source_task="earlier_continuous_problem")
    workspace.pending_actions = [{"name": "continuous_state", "method_id": method,
        "method_condition": condition, "context": context, "profile": profile,
        "program": workspace.incumbent.program.to_dict(), "compute_cost": 0, "observation_cost": 0}]
    workspace.requested_method = {"name": "continuous_state", "method_id": method,
        "steps": ["continuous_state"], "context": context, "condition": condition, "profile": profile}
    return method


def test_actual_parameter_repair_replaces_scheduled_state_search_and_earns_its_own_feedback():
    source_spec, target, source, initial, fresh, later = conflict_learning()
    library = KnowledgeLibrary()
    identifier, _ = library.learn(Candidate("learned_source", source, "representation"), source_spec)
    ambiguous, _ = refit(library, identifier, target, initial)
    fallback = initial_prediction(initial, .5)
    workspace = ResearchWorkspace(target.task_id, observations=initial + fresh, reference=fresh,
        incumbent=ambiguous, evidence_champion=ambiguous, bank=[ambiguous], fallback=fallback,
        initial_loss=loss(Program("const", value=fallback), initial, fallback, .5), rounds_completed=1,
        queried=[*target.initial_probes, Probe("already_paid_trajectory", tuple(row.features for row in fresh))])
    method = pending_continuous(library, target, workspace)
    restored_library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    environment = ConflictLagEnvironment(target)
    config = ResearchConfig(seed=1237, max_rounds=1, compute_budget=18000, observation_budget=24)
    result = CognitiveAgent(config, restored_library).run(environment,
        research_state=json.loads(json.dumps(workspace.to_dict())))
    repaired = next(action for action in result.actions if action.action == "refit_transfer")
    changed = next(action for action in result.actions if action.action == "revise_research_step")
    comparison = next(action for action in result.actions if action.action == "compare")
    feedback = next(action for action in result.actions if action.action == "action_feedback")
    assert repaired.details["residual_resolved"] and 0 < repaired.compute_cost <= 500
    assert changed.details["deferred"] == [{"action": "continuous_state", "method_id": method}]
    assert any(action.action == "defer_method_step" and action.details["method_id"] == method for action in result.actions)
    assert not any(action.action == "execute_method_step" and action.details["method_id"] == method for action in result.actions)
    assert comparison.details["adopt"] and comparison.details["fresh_gain"] > 0
    assert feedback.details["action"] == "refit" and feedback.details["last_gain"] > 0
    assert not any(action.action == "continuous_state" for action in result.actions)
    assert not any(action.action == "action_feedback" and action.details["action"] == "continuous_state"
                   for action in result.actions)
    original_method = next(item for item in restored_library.methods.methods if item["method_id"] == method)
    assert original_method["trials"] == 0
    assert len(original_method["step_outcomes"]) == 1
    assert loss(result.program, later, result.fallback, .5) < 1e-12
    assert loss(result.program, environment.observed, result.fallback, .5) < 1e-12
    assert result.observation_cost == len(environment.observed) > 0
    assert all(probe.probe_id not in {old.probe_id for old in target.initial_probes} for probe in environment.queries)


def test_an_inadequate_transferred_structure_does_not_cancel_real_continuous_state_research():
    from test_cognition_continuous_runtime import measured_task

    laboratory, environment = measured_task()
    observations = [row for probe in environment.spec.initial_probes for row in environment.observe(probe)]
    library = KnowledgeLibrary()
    program = Program("feature", name=environment.spec.feature_names[0])
    source_spec = replace(environment.spec, task_id="prior_current_input_law")
    identifier, _ = library.learn(Candidate("current_only", program, "representation"), source_spec)
    incumbent = Candidate("transferred", program, "reuse", parents=(identifier,), memory_id=identifier)
    fallback = initial_prediction(observations, .5)
    workspace = ResearchWorkspace(environment.spec.task_id, observations=observations,
        incumbent=incumbent, evidence_champion=incumbent, bank=[incumbent], fallback=fallback,
        initial_loss=loss(Program("const", value=fallback), observations, fallback, .5),
        queried=list(environment.spec.initial_probes))
    pending_continuous(library, environment.spec, workspace)
    result = CognitiveAgent(ResearchConfig(seed=82, max_rounds=2, compute_budget=40000,
        observation_budget=32), library).run(environment, research_state=json.loads(json.dumps(workspace.to_dict())))
    attempt = next(action for action in result.actions if action.action == "refit_transfer")
    assert not attempt.details["residual_resolved"]
    assert not any(action.action == "revise_research_step" and action.round == attempt.round for action in result.actions)
    execution = next(action for action in result.actions if action.action == "continuous_state")
    assert execution.compute_cost > 0 and execution.details["candidates"] > 0
    feedback = next(action for action in result.actions if action.action == "action_feedback"
                    and action.details["action"] == "continuous_state")
    assert feedback.details["last_after_loss"] < feedback.details["last_before_loss"]
    assert "accumulate" in result.program.describe()
    assert any(action.action == "execute_method_step" for action in result.actions)
    assert result.compute_cost <= 40000
    laboratory.close()


def test_session_does_not_credit_a_deferred_method_for_the_actual_refit_continuation():
    source_spec, target, source, _, _, later = conflict_learning()
    library = KnowledgeLibrary()
    library.learn(Candidate("known_source", source, "representation"), source_spec)
    config = ResearchConfig(seed=1237, max_rounds=1, compute_budget=18000, observation_budget=24)
    limits = {"observation_budget": 72, "compute_budget": 36000, "max_visits": 2}
    session = ResearchSession(config, [lambda: ConflictLagEnvironment(target)], library=library, **limits)
    first = session.run(max_steps=1)
    assert first["completed_visits"] == 1 and first["tasks"][0]["unresolved"]
    workspace = ResearchWorkspace.from_dict(first["workspaces"]["0"])
    restored_library = KnowledgeLibrary.from_dict(first["knowledge"])
    method = pending_continuous(restored_library, target, workspace, condition_name="model_residual")
    first["knowledge"] = restored_library.to_dict()
    first["workspaces"]["0"] = workspace.to_dict()
    continued = ResearchSession(config, [lambda: ConflictLagEnvironment(target)],
        state=json.loads(json.dumps(first)), **limits).run()
    assert continued["completed_visits"] == 2
    last = continued["episodes"][-1]
    assert last["scheduler"]["reason"] == "learned_method_opportunity"
    assert last["scheduler"]["method_opportunity"]["method_id"] == method
    assert any(row["action"] == "defer_method_step" for row in last["actions"])
    assert not any(row["action"] == "execute_method_step" for row in last["actions"])
    assert continued["tasks"][0]["continuation_return"]["gain"] > 0
    original_method = next(item for item in continued["knowledge"]["methods"]["methods"] if item["method_id"] == method)
    assert original_method.get("continuation_outcomes", []) == []
    assert original_method["trials"] == 0
    assert loss(Program.from_dict(last["program"]), later, last["fallback"], .5) < 1e-12
