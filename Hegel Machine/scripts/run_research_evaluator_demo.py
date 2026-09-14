"""Constructed functional example using actual paid continuation returns.

This one fixed family demonstrates the evaluator's execution path. It is not
a replacement for the retained frozen study or new generalization evidence.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import __version__
from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_conditions import method_profile
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_opportunities import ResearchOpportunities
from hegel_machine.cognition.synthesis import synthesize
from hegel_machine.cognition.workspace import ResearchWorkspace


class LinearResponse:
    def __init__(self, task_id, intercept):
        self.intercept, self.observation_cost = intercept, 0
        self.spec = TaskSpec(task_id, "Predict the measured response", ("x",), (), (),
                            feature_bounds={"x": (-1., 1.)}, max_probe_length=8)

    def observe(self, probe):
        self.observation_cost += probe.cost
        rows, history = [], []
        for features in probe.sequence:
            target = features["x"] + self.intercept
            rows.append(Observation(dict(features), target, tuple(history)))
            history.append({**features, "__target__": target})
        return rows


def prepare(identifier, offset):
    """Fit an old source response; measure a new target before continuing it."""
    source = LinearResponse(f"source_{identifier}", .25 + offset)
    target = LinearResponse(identifier, .25)
    probe = Probe("already_measured", tuple({"x": value} for value in (-1., -.5, 0., .5, 1.)))
    source_rows, target_rows = source.observe(probe), target.observe(probe)
    work = SearchBudget(500)
    fitted = synthesize(source_rows, operations=("refit",), budget=work, max_history=0)[0]
    work.spend(len(source_rows))
    source_loss = loss(fitted.program, source_rows, 0., .5)
    fallback = mean(row.target for row in target_rows)
    work.spend(2 * len(target_rows))
    baseline_loss = loss(Program("const", value=fallback), target_rows, fallback, .5)
    prior_loss = loss(fitted.program, target_rows, fallback, .5)
    prior = Candidate(f"transferred_{identifier}", fitted.program, "refit")
    workspace = ResearchWorkspace(identifier, observations=target_rows, reference=target_rows,
        queried=[probe], incumbent=prior, bank=[prior], fallback=fallback, initial_loss=baseline_loss)
    profile = method_profile(target.spec, target_rows, prior, residual_loss=prior_loss,
                             baseline_loss=baseline_loss)
    offer = {"question_id": identifier, "kind": "revision", "action": "refit",
             "profile": profile, "observation_cost": 16, "compute_cost": 5000}
    preparation = {"source_observations": source.observation_cost, "target_observations": target.observation_cost,
        "compute_cost": work.spent, "source_fit_loss": source_loss,
        "source_program": fitted.program.to_dict(), "prior_target_loss": prior_loss,
        "target_constant_baseline_loss": baseline_loss}
    return target, workspace, offer, preparation


def execute(prepared, selected, book, config):
    target, workspace, _, preparation = prepared
    result = CognitiveAgent(config).run(target, research_state=workspace.to_dict())
    actual = next(action.details for action in reversed(result.actions)
                  if action.action == "continuation_feedback")
    feedback = {"execution_complete": not bool(result.research_state["decision_frontier"]
                                              or result.research_state["experiment_frontier"]), **actual}
    update_budget = SearchBudget(1200)
    recorded = book.observe(selected, feedback, observation_cost=result.observation_cost,
                            compute_cost=result.compute_cost, evaluator_budget=update_budget)
    return {"question_id": result.task_id, "preparation": preparation,
        "predicted_gain": selected.get("expected_gain"), "baseline_expected_gain": selected.get("baseline_expected_gain"),
        "basis": selected["basis"], "actual_feedback": feedback,
        "final_program": result.program.to_dict(), "final_validation_loss": result.validation_loss,
        "observation_cost": result.observation_cost, "compute_cost": result.compute_cost,
        "evaluator_update_compute": update_budget.spent,
        "evaluator_update": recorded.get("evaluator", {}).get("update")}


def run():
    config = ResearchConfig(seed=503, observation_budget=16, compute_budget=5000, max_rounds=1,
        strategy="fixed", use_memory=False, use_abstractions=False, active_probes=False,
        synthesize_probes=False, joint_planning=False, state_search=False, use_warm_start=False)
    offsets = (.1, .2, .3, .4, .5, .6, .1, .6, .1, .6)
    book, visits, prediction_compute = ResearchOpportunities(), [], 0
    for index, offset in enumerate(offsets):
        prepared = prepare(f"calibration_{index}", offset)
        budget = SearchBudget(256)
        selected = book.estimate(prepared[2], evaluator_budget=budget)
        prediction_compute += budget.spent
        visits.append(execute(prepared, selected, book, config))
        if index == 5:
            # Candidate comparison survives the same ordinary JSON boundary as
            # a selected offer; the subsequent returns have not arrived yet.
            book = ResearchOpportunities(json.loads(json.dumps(book.to_dict())))
    pending = [prepare("smaller_drift", .25), prepare("larger_drift", .55)]
    quotes = [item[2] for item in pending]
    old = book.rank(quotes, observation_remaining=16, compute_remaining=5000)
    budget = SearchBudget(256)
    revised = book.rank(quotes, observation_remaining=16, compute_remaining=5000, evaluator_budget=budget)
    prediction_compute += budget.spent
    before_execution = book.evaluator.to_dict()
    selected_visit = None
    if revised:
        selected = revised[0]
        prepared = next(item for item in pending if item[2]["question_id"] == selected["question_id"])
        selected_visit = execute(prepared, selected, book, config)
    all_visits = [*visits, *([selected_visit] if selected_visit else [])]

    def compact(quote):
        return {key: quote.get(key) for key in ("question_id", "expected_gain", "baseline_expected_gain", "basis", "value")}

    costs = {
        "preparation_observations": sum(visit["preparation"]["source_observations"]
            + visit["preparation"]["target_observations"] for visit in visits)
            + sum(item[3]["source_observations"] + item[3]["target_observations"] for item in pending),
        "preparation_compute": sum(visit["preparation"]["compute_cost"] for visit in visits)
            + sum(item[3]["compute_cost"] for item in pending),
        "continuation_observations": sum(visit["observation_cost"] for visit in all_visits),
        "continuation_compute": sum(visit["compute_cost"] for visit in all_visits),
        "evaluator_predict_compute": prediction_compute,
        "evaluator_update_compute": sum(visit["evaluator_update_compute"] for visit in all_visits)}
    return {"version": __version__, "kind": "constructed_research_evaluator_execution_demo",
        "scope": "One fixed linear-response family with observed source-parameter drift; functional demonstration only.",
        "protocol": {"seed": 503, "offsets": list(offsets), "proposal_visits": 6,
                     "subsequent_comparison_visits": 4, "pending_offsets": [.25, .55], "maximum_continuations": 11},
        "costs": costs, "visits": visits, "old_ranking": list(map(compact, old)),
        "revised_ranking": list(map(compact, revised)),
        "choice_changed": bool(old and revised and old[0]["question_id"] != revised[0]["question_id"]),
        "evaluator_before_selection": {"current": before_execution["current"], "events": before_execution["events"]},
        "selected_visit": selected_visit,
        "limitations": "These constructed examples do not alter the frozen study, prove generalization, or establish net long-term benefit."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research_evaluator_demo_20260909.json")
    args = parser.parse_args()
    report = run()
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "choice_changed": report["choice_changed"],
        "active_model": report["evaluator_before_selection"]["current"], "costs": report["costs"]}, ensure_ascii=False))
