"""A failed source method must lead to funded, executed structural research."""

from dataclasses import replace
import json
from pathlib import Path

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.controller import CognitiveAgent, _refit_transferred, _representation_work
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.experiment_knowledge import research_context
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.method_conditions import method_profile
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.workspace import ResearchWorkspace
from test_cognition_conflict_transfer_refit import ConflictLagEnvironment, conflict_learning, refit
from test_cognition_failed_method_revision import problem


class LongerLagEnvironment:
    """Execute every requested probe, including new prospective trajectories."""

    def __init__(self, spec):
        self.spec, self.observed, self.queries = spec, [], []

    def observe(self, probe):
        self.queries.append(probe)
        rows, past = [], []
        for features in probe.sequence:
            target = .3 * features["x"] - (.8 * past[-4]["x"] if len(past) >= 4 else 0.)
            rows.append(Observation(dict(features), target, tuple(past)))
            past.append({**features, "__target__": target})
        self.observed.extend(rows)
        return rows


def saved_workspace(spec, incumbent, rows, reference):
    fallback = initial_prediction(rows, .5)
    return ResearchWorkspace(spec.task_id, observations=rows, reference=reference,
        incumbent=incumbent, evidence_champion=incumbent, bank=[incumbent], fallback=fallback,
        initial_loss=loss(Program("const", value=fallback), rows, fallback, .5))


def test_failed_lag_source_executes_complete_representation_search_and_learns_its_new_trajectory_return(monkeypatch):
    import hegel_machine.cognition.controller as controller

    spec, source, rows = problem()
    initial = rows[:28]
    library = KnowledgeLibrary()
    identifier, _ = library.learn(replace(source, memory_id=None), spec)
    source = replace(source, memory_id=identifier)
    config = ResearchConfig(seed=71, max_rounds=3, compute_budget=24000, observation_budget=80)
    initial_budget = SearchBudget(500)
    fitted = _refit_transferred(library, source, spec, initial, initial_budget, config, 0.)
    incumbent = fitted[0].candidate
    assert incumbent.memory_id == identifier and fitted[0].evidence_loss > .01
    assert initial_budget.spent > 0

    # A real fitted lag-2 candidate is restored with another parameter repair
    # scheduled. No failure is prefilled: both failures must occur in this run.
    workspace = saved_workspace(spec, incumbent, initial, initial[-16:])
    policy = ResearchActionPolicy(initial, spec, library.action_experience)
    context = policy.set_condition("model_residual")
    before = method_profile(spec, initial, incumbent, residual_loss=fitted[0].evidence_loss,
                            baseline_loss=workspace.initial_loss)
    workspace.pending_actions = [{"name": "refit", "context": context,
        "program": incumbent.program.to_dict(), "profile": before,
        "compute_cost": 0, "observation_cost": 0, "method_condition": research_context(spec, initial)}]
    environment = LongerLagEnvironment(spec)
    search_calls = []
    original_search = controller.sparse_candidates

    def observed_search(observations, *, budget, **kwargs):
        required = _representation_work(observations, spec, config.max_history)
        candidates = original_search(observations, budget=budget, **kwargs)
        search_calls.append({"rows": len(observations), "required": required, "allocated": budget.limit,
                             "spent": budget.spent, "programs": [candidate.program for candidate in candidates]})
        return candidates

    monkeypatch.setattr(controller, "sparse_candidates", observed_search)
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    result = CognitiveAgent(config, restored).run(environment,
        research_state=json.loads(json.dumps(workspace.to_dict())))

    revision = next(action for action in result.actions if action.action == "revise_research_method")
    assert revision.details["selected_action"] == "representation"
    assert revision.details["applicability"]["condition"]["status"] == "parameter_repair_exhausted"
    assert search_calls and search_calls[0]["rows"] > len(initial)
    assert search_calls[0]["allocated"] >= search_calls[0]["required"] > 0
    assert search_calls[0]["spent"] > 0
    assert any(program.history_depth == 4 for program in search_calls[0]["programs"])
    assert any(action.action == "composition" and action.round == revision.round + 1
               and action.compute_cost > 0 for action in result.actions)
    assert any(action.action == "compare" and action.round == revision.round + 1
               and action.details["adopt"] and action.details["fresh_gain"] > 0 for action in result.actions)
    assert result.program.history_depth == 4
    assert loss(result.program, environment.observed, result.fallback, .5) < 1e-12
    assert len(environment.observed) == result.observation_cost

    trace = result.research_state["method_trace"]
    diagnostics = [step for step in trace if step["action"] == "refit_transfer"]
    paid_diagnostics = [step for step in diagnostics if step["execution_complete"]]
    assert paid_diagnostics
    assert all(step["source_id"] == identifier and step["compute_cost"] > 0 and step["candidate_count"]
               and not step["residual_resolved"] and step["gain"] == 0. and "fit_gain" in step
               for step in paid_diagnostics)
    assert any(step["action"] == "refit" and step["execution_complete"] and step["gain"] == 0. for step in trace)
    completed = next(step for step in trace if step["action"] == "representation" and step["gain"] > 0)
    assert completed["profile"]["source_revision"]["status"] == "parameter_repair_exhausted"
    learned = next(method for method in restored.methods.methods if method["method_id"] == completed["produced_method_id"])
    assert learned["steps"] == ["representation"]
    assert learned["step_outcomes"][0]["gain"] == completed["gain"]
    assert learned["step_outcomes"][0]["compute_cost"] == completed["compute_cost"]


def test_valid_source_parameters_still_repair_before_structural_search_and_receive_fresh_credit():
    source_spec, spec, source, initial, fresh, _ = conflict_learning()
    library = KnowledgeLibrary()
    identifier, _ = library.learn(Candidate("lag_source", source, "representation"), source_spec)
    incumbent, fitting_budget = refit(library, identifier, spec, initial)
    assert fitting_budget.spent > 0
    workspace = saved_workspace(spec, incumbent, initial + fresh, fresh)
    environment = ConflictLagEnvironment(spec)
    config = ResearchConfig(seed=71, max_rounds=3, compute_budget=18000, observation_budget=48)
    result = CognitiveAgent(config, library).run(environment,
        research_state=json.loads(json.dumps(workspace.to_dict())))
    repair = next(action for action in result.actions if action.action == "refit_transfer")
    assert repair.details["residual_resolved"] and 0 < repair.compute_cost <= 500
    assert not any(action.action == "revise_research_method" for action in result.actions)
    assert not any(action.action in {"composition", "continuous_state"} for action in result.actions)
    assert any(action.action == "compare" and action.details["adopt"]
               and action.details["fresh_gain"] > 0 for action in result.actions)
    trace = result.research_state["method_trace"]
    diagnosis = next(step for step in trace if step["action"] == "refit_transfer")
    assert diagnosis["residual_resolved"] and diagnosis["gain"] == 0.
    assert any(step["action"] == "refit" and step["execution_complete"] and step["gain"] > 0 for step in trace)
    assert result.program.history_depth == 2
    assert result.selected_memory_id == identifier
    assert loss(result.program, environment.observed, result.fallback, .5) < 1e-12


def test_paid_n09_failures_replace_coverage_fallback_with_an_executed_structural_successor(monkeypatch):
    import hegel_machine.cognition.controller as controller

    fixture = json.loads((Path(__file__).parent / "fixtures" / "cognition_n09_failed_parameter_learning.json").read_text())
    suite = make_suite(fixture["public_generator_seed"], "continual")
    environment = next(item for item in suite if item.spec.task_id == fixture["target_task"])
    source_spec = next(item.spec for item in suite if item.spec.task_id == fixture["source_task"])
    library = KnowledgeLibrary()
    identifier, _ = library.learn(Candidate("source", Program.from_dict(fixture["source_program"]), "representation"), source_spec)
    library.action_experience = fixture["action_experience"]
    rows = [Observation(row["features"], row["target"], tuple(row["history"])) for row in fixture["observations"]]
    incumbent = Candidate("paid_explanation", Program.from_dict(fixture["incumbent_program"]), "transfer", memory_id=identifier)
    trace = fixture["paid_method_trace"]
    for step in trace:
        if step["action"] == "refit_transfer":
            step["source_id"] = identifier
    refitting = next(step for step in trace if step["action"] == "refit")
    # The saved selection profile has only 12 rows. This refit actually ran
    # after the source diagnostic on the same 32 acquired rows.
    refitting["execution_profile"] = fixture["executed_source_refit_profile"]
    assert refitting["gain"] == 0. and refitting["profile"]["observation_count"] == 12
    assert refitting["execution_profile"]["observation_count"] == 32
    earliest = method_profile(environment.spec, rows[:32], incumbent, trace[:2],
        residual_loss=fixture["executed_source_refit_profile"]["residual"]["loss"],
        baseline_loss=fixture["initial_loss"])
    assert earliest["source_revision"]["status"] == "parameter_repair_exhausted"
    assert earliest["source_revision"]["failed_attempts"] == 2
    current = method_profile(environment.spec, rows, incumbent, trace,
        residual_loss=fixture["selection_profile"]["residual"]["loss"],
        baseline_loss=fixture["initial_loss"])
    assert current["source_revision"]["status"] == "parameter_repair_exhausted"

    # These are the actual two competing quotes from the paid round-2 plan.
    # The old learned coverage mean chooses representative; the revised plan
    # must reserve the available structure step rather than merely record it.
    policy = ResearchActionPolicy(rows, environment.spec, library.action_experience)
    context = policy.set_condition("model_residual")
    options = [ActionOption("representative", 8, 8), ActionOption("representation", 0, 1954)]
    ordinary = policy.choose(options, observation_remaining=14, compute_remaining=11016)
    assert ordinary.name == "representative"
    revision = library.methods.revise_plan(current, options, trace)
    assert ordinary.name in revision["excluded_actions"]
    assert revision["action"] == "representation"

    reference = [row for low, high in fixture["reference_ranges"] for row in rows[low:high]]
    workspace = saved_workspace(environment.spec, incumbent, rows, reference)
    workspace.initial_loss = fixture["initial_loss"]
    workspace.rounds_completed = 3
    workspace.method_trace = trace
    workspace.queried = [Probe(item["probe_id"], tuple(item["sequence"])) for item in fixture["queried"]]
    workspace.pending_actions = [{"name": revision["action"], "context": context,
        "program": incumbent.program.to_dict(), "compute_cost": 0, "observation_cost": 0,
        "profile": current, "method_condition": research_context(environment.spec, rows)}]
    searches = []
    original = controller.sparse_candidates

    def record_search(observations, *, budget, **kwargs):
        searches.append((budget.limit, _representation_work(observations, environment.spec, 4)))
        return original(observations, budget=budget, **kwargs)

    monkeypatch.setattr(controller, "sparse_candidates", record_search)
    result = CognitiveAgent(ResearchConfig(seed=fixture["query_seed"], max_rounds=2,
        compute_budget=11016, observation_budget=26), library).run(environment,
            research_state=json.loads(json.dumps(workspace.to_dict())))
    assert searches and searches[0][0] >= searches[0][1]
    assert result.validation_loss < 1e-12
    assert result.program.history_depth == 4
    assert any(action.action == "defer_source_refit" for action in result.actions)
    assert any(action.action == "compare" and action.details["adopt"] and action.details["fresh_gain"] > 0
               for action in result.actions)
    assert any(step["action"] == "representation" and step["gain"] > 0
               and step["profile"]["source_revision"]["status"] == "parameter_repair_exhausted"
               for step in result.research_state["method_trace"])
    assert result.observation_cost <= 26 and result.compute_cost <= 11016
