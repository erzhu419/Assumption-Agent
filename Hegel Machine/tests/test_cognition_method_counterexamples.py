"""New paid structure revises a method agenda rather than freezing old ranks."""

from copy import deepcopy
from dataclasses import replace
import gzip
import importlib
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.condition_response_actions import _observed_draft_claims
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.risk_research_actions import RiskResearchActions
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for


@pytest.fixture(scope="module")
def recorded_numeric():
    path = Path(__file__).parent / "fixtures/method_binding_counterexample_root09.json.gz"
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


@pytest.fixture(scope="module")
def laboratory_factory():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        return importlib.import_module("run_observed_condition_decoding_study").laboratory


def restore_batches(recorded):
    batches = deepcopy(recorded["paid_batches"])
    for batch in batches:
        history = []
        for row in batch["observations"]:
            row["history"] = list(history)
            history.append({**row["features"], "__target__": row["target"]})
    return batches


def numeric_agent(recorded):
    config = replace(ResearchConfig(**recorded["config"]), seed=recorded["task"]["seed"])
    return UnifiedResearchAgent(config,
        library=KnowledgeLibrary.from_dict(deepcopy(recorded["before_task_library"])),
        measurement_budget=recorded["measurement_budget"])


def test_current_paid_window_rewrites_and_validates_the_simpler_method(
        recorded_numeric, laboratory_factory):
    saved = deepcopy(recorded_numeric)
    old_selection = recorded_numeric["paid_actions"][13]
    assert old_selection["selected_draft_ids"] == ["research_draft_0001"] * 2
    assert any(item["training_loss"] < 1e-7 and item["validation_loss"] > 30.
               for item in recorded_numeric["recorded_post_support_assessments"])
    evidence = recorded_numeric["current_duration_evidence"]
    assert evidence[0]["width"] == 7 and evidence[0]["placement"] == "outside_window"
    batches = restore_batches(recorded_numeric)
    # This is the original 84/24 split: the duration rows are training, while
    # the earlier varied-control mapping comparison remains unfitted validation.
    training = [Observation(**row) for index in (0, 1, 2, 4, 5)
                for row in batches[index]["observations"]]
    validation = [Observation(**row) for row in batches[3]["observations"]]
    assert len(training) == 84 and len(validation) == 24
    budget, actions = SearchBudget(80000), []
    agent = numeric_agent(recorded_numeric)
    with laboratory_factory(recorded_numeric["task"]) as lab:
        claims = _observed_draft_claims(agent, agent._spec(lab, ("reading",)),
            training, validation, budget, actions, evidence=evidence)
        assert lab.costs["observation_cost"] == lab.costs["measurement_cost"] == 0
    revised = next(event for event in actions if event["action"] == "revise_research_draft_agenda")
    assert revised["evidence"] == evidence and revised["validation_rows"] == 24
    replays = [event for event in actions if event["action"] == "replay_research_draft"]
    simple = next(event for event in replays if event["draft_id"] == "research_draft_0002")
    assert simple["response_basis_terms"] == 3 and simple["current_duration_evidence"]
    assert simple["compute_cost"] > 0
    chosen = min((claim for claim in claims if claim["research_draft_id"] == "research_draft_0002"),
                 key=lambda claim: max(claim["training_loss"], claim["validation_loss"]))
    assert max(chosen["training_loss"], chosen["validation_loss"]) < 1e-7
    assert loss(chosen["candidate"].program, validation, initial_prediction(training, .5), .5) < 1e-7
    assert any(step.get("width") == 7 and step.get("placement") == "outside_window"
               and not step.get("historical") for step in chosen["research_program"]["selection_steps"])
    assert 0 < budget.spent <= 80000 and recorded_numeric == saved
    print(json.dumps({"paid_method_revision": {"compute": budget.spent,
        "sources": [event["draft_id"] for event in replays],
        "training_loss": chosen["training_loss"], "validation_loss": chosen["validation_loss"]}}))


def test_normal_numeric_run_compares_distinct_sources_without_false_completion(
        recorded_numeric, laboratory_factory):
    agent = numeric_agent(recorded_numeric)
    with laboratory_factory(recorded_numeric["task"]) as lab:
        report = agent.run(lab)
        physical_costs = dict(lab.costs)
    actions = report["actions"]
    print(json.dumps({"normal_method_trace": {"status": report["status"],
        "costs": report["costs"], "confirmation_loss": report["confirmation_loss"],
        "replays": [{key: event.get(key) for key in ("draft_id", "compute_cost",
            "training_loss", "current_duration_evidence")} for event in actions
            if event["action"] == "replay_research_draft"],
        "assessments": [{key: event[key] for key in ("draft_id", "training_loss", "validation_loss")}
            for event in actions if event["action"] == "assess_research_draft"],
        "duration": [[{key: item.get(key) for key in ("control", "width", "placement", "loss")}
            for item in event["evidence"]] for event in actions
            if event["action"] == "distinguish_paid_condition_duration"]}}))
    selections = [event for event in actions if event["action"] == "select_research_drafts_on_paid_response"]
    assert selections
    assert len(set(selections[0]["selected_draft_ids"])) == 2
    assert "research_draft_0001" in selections[0]["selected_draft_ids"]
    for event in selections:
        assert event["compute_cost"] == sum(item["compute_cost"] for item in event["alternatives"])
    simple_indices = [index for index, event in enumerate(actions)
        if event["action"] == "replay_research_draft" and event["draft_id"] == "research_draft_0002"]
    assert simple_indices
    assert any(event["action"] == "assess_research_draft"
               and event["draft_id"] == "research_draft_0002" for event in actions)
    validations = [(index, event) for index, event in enumerate(actions)
        if index > simple_indices[0] and event["action"] == "validate_frozen_program"]
    for validation_index, validation in validations:
        purchase_index, purchase = next((index, event) for index, event in reversed(
            list(enumerate(actions[:validation_index]))) if event["action"] == "acquire"
            and event["purpose"] == "confirm_frozen_program")
        assert purchase_index > simple_indices[0]
        batch = next(item for item in report["batches"]
                     if item["probe"]["probe_id"] == purchase["probe_id"])
        assert purchase["observation_cost"] == purchase["measurement_cost"] == len(batch["observations"])
        assert validation["observation_rows"] == len(batch["observations"])
        assert validation["loss"] == pytest.approx(loss(Program.from_dict(validation["program"]),
            [Observation(**row) for row in batch["observations"]], report["fallback"], .5), abs=1e-7)
    if not validations:
        assert report["status"] == "unresolved" and report["confirmation_loss"] is None
        assert not report["concepts_added"]
    for resource in ("observation_cost", "measurement_cost"):
        assert report["costs"][resource] == physical_costs[resource]
        assert report["costs"][resource] == sum(event[resource] for event in actions if event["action"] == "acquire")
    assert report["costs"]["compute_cost"] <= recorded_numeric["config"]["compute_budget"]
    if report["status"] == "completed":
        assert report["confirmation_loss"] < 1e-7
    print(json.dumps({"normal_method_revision": {"status": report["status"],
        "first_sources": selections[0]["selected_draft_ids"], "costs": report["costs"],
        "confirmation_loss": report["confirmation_loss"]}}))


@pytest.fixture(scope="module")
def real_risk_chain(tmp_path_factory):
    config = ResearchConfig(observation_budget=384, compute_budget=240000,
        minimum_improvement=1e-7, active_probes=False, max_rounds=4, seed=7907)
    tasks = [{"task_index": task, "seed": seed, "profile": profile}
        for task, seed, profile in ((12, 7907, "development"),
            (13, 7908, "development"), (13, 7909, "confirmation"))]
    labs = []

    def factory(task):
        lab = laboratory_for(task)
        labs.append(lab)
        return lab

    session = UnifiedResearchSession(config, tasks=tasks, measurement_budget=1536,
                                    lab_factory=factory)
    snapshots, rewrite_calls = [], []
    original_rewrite = RiskResearchActions._rewrite_method

    def record_rewrite(controller, support, retained_change, previous):
        counterexample = (support or {}).get("structural_counterexample") or {}
        captured = {"seed": controller.agent.config.seed,
            "support_reason": (support or {}).get("reason"),
            "support_adequate": (support or {}).get("adequate"),
            "counterexample": {key: counterexample.get(key) for key in ("kind", "gain", "margin")},
            "retained_change": retained_change,
            "previous_action": (previous or {}).get("action"),
            "previous_status": (previous or {}).get("status"),
            "attempted_before": sorted(controller.attempted),
            "steps_before": (deepcopy(controller.current_method) or {}).get("steps"),
            "resources": controller.remaining()}
        preferred = original_rewrite(controller, support, retained_change, previous)
        captured.update(preferred=preferred, attempt_length=controller.length,
            attempted_after=sorted(controller.attempted),
            steps_after=(deepcopy(controller.current_method) or {}).get("steps"))
        rewrite_calls.append(captured)
        return preferred

    # Observe the real decision without injecting evidence or a chosen action.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RiskResearchActions, "_rewrite_method", record_rewrite)
        for _ in tasks:
            state = json.loads(json.dumps(session.run(max_tasks=1)))
            snapshots.append(state)
            session = UnifiedResearchSession(state=state, lab_factory=factory)
        queued = snapshots[-1]
        if any(offer["candidate"].get("risk_method") is not None for offer in queued["pending_revisits"]):
            final = json.loads(json.dumps(session.run(max_tasks=1)))
        else:
            final = queued
    trace_path = tmp_path_factory.mktemp("risk_method_counterexamples") / "development_trace.json"
    trace_path.write_text(json.dumps({"rewrite_calls": rewrite_calls,
        "choices": [{"seed": tasks[index]["seed"], "events": [event for event in snapshot["results"][index]["research"]["actions"]
            if event["action"] in {"choose_risk_research", "revise_risk_research_method", "propose_risk_research"}]}
            for index, snapshot in enumerate(snapshots)],
        "natural_revisits": [{"task_position": visit["task_position"],
            "events": [event for event in final["results"][visit["task_position"]]["research"]["actions"][
                len(visit["previous"]["actions"]):] if event["action"] in {
                    "choose_risk_research", "revise_risk_research_method", "propose_risk_research"}],
            "fresh_comparison": (visit.get("method_visit_outcome") or {}).get("fresh_comparison"),
            "whole_visit_costs": (visit.get("method_visit_outcome") or {}).get("whole_visit_costs"),
            "screening_compute_cost": visit.get("screening_compute_cost")}
            for visit in final["revisits"][len(queued["revisits"]):]]}, indent=2))
    print("risk_method_development_trace=" + str(trace_path))
    print(json.dumps({"risk_method_chain": {
        "stages": [{"status": snapshot["results"][index]["research"]["status"],
            "costs": snapshot["results"][index]["research"]["costs"],
            "recipe_count": sum(bool(item.get("risk_method"))
                                for item in snapshot["knowledge"]["research_drafts"]),
            "replays": [event["draft_id"] for event in snapshot["results"][index]["research"]["actions"]
                        if event["action"] == "replay_risk_method"],
            "rewrites": [{key: event.get(key) for key in ("draft_id", "before_steps", "after_steps", "reason")}
                for event in snapshot["results"][index]["research"]["actions"]
                if event["action"] == "revise_risk_research_method"]}
            for index, snapshot in enumerate(snapshots)],
        "pending_risk_methods": sum(offer["candidate"].get("risk_method") is not None
                                     for offer in queued["pending_revisits"]),
        "revisits": [{"position": visit["task_position"], "basis": visit["basis"],
            "whole_costs": (visit.get("method_visit_outcome") or {}).get("whole_visit_costs"),
            "screening": visit.get("screening_compute_cost"),
            "fresh_before": ((visit.get("method_visit_outcome") or {}).get("fresh_comparison") or {}).get("before_loss"),
            "fresh_after": ((visit.get("method_visit_outcome") or {}).get("fresh_comparison") or {}).get("after_loss")}
            for visit in final["revisits"][len(queued["revisits"]):]]}}, sort_keys=True))
    return snapshots, queued, final, labs, rewrite_calls


def test_actual_risk_method_survives_json_and_rewrites_after_paid_counterevidence(real_risk_chain):
    snapshots, queued, final, _, rewrite_calls = real_risk_chain
    source = snapshots[1]
    records = [item for item in source["knowledge"]["research_drafts"]
               if item.get("risk_method") and item["source_seed"] == 7908]
    assert records and all(item["risk_quantile"] == .9 for item in records)
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(source["knowledge"])))
    without = library.without({concept.concept_id for concept in library.concepts})
    surviving = {item["draft_id"]: item["risk_method"] for item in without.research_drafts
                 if item.get("risk_method")}
    assert all(surviving[item["draft_id"]] == item["risk_method"] for item in records)
    report = queued["results"][2]["research"]
    assert report["risk_quantile"] == .9 and report["loss_kind"] == "pinball"
    actions = report["actions"]
    source_ids = {item["draft_id"] for item in records}
    assert any(event["action"] == "replay_risk_method" and event["draft_id"] in source_ids
               and event["compute_cost"] > 0 for event in actions)
    inconclusive = next(call for call in rewrite_calls if call["seed"] == 7909
        and call["support_reason"] == "noise_replication_inconclusive"
        and call["previous_action"] == "revise_uncertainty")
    assert not inconclusive["support_adequate"]
    assert inconclusive["counterexample"]["kind"] is None
    assert not any(event["action"] == "revise_risk_research_method"
                   and event["reason"] == "fresh_refuted_recorded_tail" for event in actions)
    stopped = [event for event in actions if event["action"] == "choose_risk_research"][-1]
    assert stopped["reason"] == "no_affordable_action" and not stopped["eligible_actions"]
    assert all(any(option[resource + "_cost"] > inconclusive["resources"][resource + "_remaining"]
                   for resource in ("observation", "measurement", "compute")) for option in stopped["options"])
    assert report["status"] == "unresolved"
    miscalibrated = next(call for call in rewrite_calls if call["seed"] == 7909
        and call["support_reason"] == "quantile_miscalibration"
        and call["previous_action"] == "revise_uncertainty")
    assert miscalibrated["preferred"] == "revise_uncertainty"
    assert "revise_uncertainty" not in miscalibrated["attempted_after"]
    resumed = final["results"][2]["research"]
    new_actions = resumed["actions"][len(report["actions"]):]
    rewrite_index, rewrite = next((index, event) for index, event in enumerate(new_actions)
        if event["action"] == "revise_risk_research_method"
        and event["reason"] == "fresh_refuted_recorded_tail")
    assert rewrite["fixed_quantile"] == .9 and rewrite["evidence_condition"] == "quantile_miscalibration"
    # The normal visit really refits the method before scoring it on new rows.
    assert any(event["action"] == "choose_risk_research"
               and event["selected_action"] == "revise_uncertainty"
               and event["reason"] == "execute_revised_paid_risk_method"
               for event in new_actions[:rewrite_index])
    assert any(event["action"] == "propose_risk_research"
               and event["direction"] == "revise_uncertainty" and event["compute_cost"] > 0
               for event in new_actions[:rewrite_index])
    # Reopening is genuine, but its larger comparison and confirmation do not
    # fit the remaining paid visit. Do not substitute a marker for execution.
    stopped = next(event for event in new_actions[rewrite_index + 1:]
                   if event["action"] == "choose_risk_research")
    assert stopped["reason"] == "no_affordable_action" and not stopped["eligible_actions"]
    calibration = next(option for option in stopped["options"]
                       if option["name"] == "revise_uncertainty")
    assert calibration["observation_cost"] >= 2 * miscalibrated["attempt_length"]
    assert calibration["observation_cost"] > miscalibrated["resources"]["observation_remaining"]
    assert not any(event["action"] == "propose_risk_research"
                   for event in new_actions[rewrite_index + 1:])
    assert resumed["status"] == "unresolved"
    paired = resumed["continuation"]["method_fresh_comparisons"][-1]
    assert (paired["program"], paired["fallback"]) != (paired["baseline_program"], paired["baseline_fallback"])
    assert paired["after_loss"] < paired["before_loss"]
    assert all(item["research"]["risk_quantile"] == .9 for item in final["results"])


def test_actual_risk_method_returns_use_new_fixed_quantile_rows_and_whole_visit_cost(real_risk_chain):
    _, queued, final, labs, _ = real_risk_chain
    paired_events = []
    for entry in final["results"]:
        report = entry["research"]
        for event in report["actions"]:
            if event["action"] != "risk_research_feedback" or event["before_loss"] is None:
                continue
            batch = next(item for item in report["batches"]
                         if item["probe"]["probe_id"] == event["probe_id"])
            rows = [Observation(**item) for item in batch["observations"]]
            assert event["risk_quantile"] == .9
            assert event["before_loss"] == pytest.approx(loss(Program.from_dict(event["baseline_program"]),
                rows, event["baseline_fallback"], .9))
            assert event["after_loss"] == pytest.approx(loss(Program.from_dict(event["program"]),
                rows, event["fallback"], .9))
            assert event["observation_rows"] == len(rows)
            assert any(item["action"] == "acquire" and item["probe_id"] == event["probe_id"]
                       for item in report["actions"])
            paired_events.append(event)
    assert paired_events
    visits = [visit for visit in final["revisits"][len(queued["revisits"]):]
              if visit["candidate"].get("risk_method") is not None]
    assert visits
    for visit in visits:
        report = final["results"][visit["task_position"]]["research"]
        outcome = visit["method_visit_outcome"]
        fresh = outcome["fresh_comparison"]
        assert fresh is not None and fresh["risk_quantile"] == .9
        assert report["continuation"]["method_fresh_comparisons"][-1] == fresh
        new = report["actions"][len(visit["previous"]["actions"]):]
        acquire = next(item for item in new if item["action"] == "acquire"
                       and item["probe_id"] == fresh["probe_id"])
        assert acquire["purpose"] == "confirm_continuation_program"
        assert fresh["probe_id"] not in {item["probe"]["probe_id"] for item in visit["previous"]["batches"]}
        for resource in ("observation_cost", "measurement_cost", "compute_cost"):
            assert outcome["whole_visit_costs"][resource] == visit["visit_costs"][resource] + (
                visit["screening_compute_cost"] if resource == "compute_cost" else 0)
            assert visit["visit_costs"][resource] <= (
                visit["resource_limits"][resource] - visit["previous"]["costs"][resource])
    for resource in ("observation_cost", "measurement_cost"):
        assert final["costs"][resource] == sum(lab.costs[resource] for lab in labs)
    assert final["costs"]["compute_cost"] == sum(entry["research"]["costs"]["compute_cost"]
        for entry in final["results"]) + final["screening_compute_cost"]
