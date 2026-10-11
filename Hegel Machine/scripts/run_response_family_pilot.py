#!/usr/bin/env python3
"""Small acquired-response diagnostic for the existing family researcher.

Three references and two targets share the public feature/control interface.
The evaluator owns the mechanism; research sees only measurements and responses.
The objective is prediction at reserved target settings, not activation at any
setting: a positive elsewhere does not complete this prediction task.
This is an offline mechanism pilot, not LLM discovery or a recursive-arm study.
"""

import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

from hegel_machine.cognition.world_response_constraints import query_response_family
from hegel_machine.cognition.world_response_hypotheses import propose_response_research


ESTIMATORS = [
    {"id": "shared_x", "basis": [{"op": "feature", "name": "x"}]},
    {"id": "shared_x_z", "basis": [{"op": "feature", "name": name}
                                        for name in ("x", "z")]},
]


class ResponseLab:
    """A directly controllable synthetic instrument; each read costs one action."""

    def __init__(self, case):
        if case == "parameter_sufficient":
            xs, zs = (0., .5, 2., 1., 1.5), (0.,) * 5
            self._centres = (0., 2., 8., 4., 6.)
            self.initial_settings, self.panel_settings = (0., 2., 10.), (4., 6.)
        elif case == "shared_feature_needed":
            xs, zs = (0., 0., 2., 1., 1.5), (0., 2., 0., 1., 0.)
            self._centres = (0., 4., 8., 6., 6.)
            self.initial_settings, self.panel_settings = (0., 4., 10.), (6., 6.)
        elif case == "shared_assumption_false":
            xs, zs = (0., 0., 2., 1., 1.5), (0.,) * 5
            self._centres = (0., 3., 8., 4., 6.)
            self.initial_settings, self.panel_settings = (0., 3., 10.), (4., 6.)
        else:
            raise ValueError(case)
        self.features = {number: {"x": x, "z": z}
                         for number, (x, z) in enumerate(zip(xs, zs), 1)}
        self.records = []

    def measure(self, number, setting):
        residual = setting - self._centres[number - 1]
        row = {"crystal": number, "frequency": float(setting),
               "activated": 0. <= residual <= 2.,
               "evidence": [5 + len(self.records)]}
        self.records.append(deepcopy(row))
        return row

    def initial_state(self):
        rows = [{"crystal": number, "measurements": {
                    name: {"value": value, "evidence": number - 1}
                    for name, value in features.items()}}
                for number, features in self.features.items()]
        responses = [self.measure(number, setting)
                     for number, setting in enumerate(self.initial_settings, 1)]
        for row, response in zip(rows, responses):
            row["frequency"] = {"value": response["frequency"],
                                "evidence": response["evidence"][0]}
        grid = {"lower": 0., "upper": 14., "step": 1., "origin": 0.}
        return {"research_table": {"references": rows[:3], "targets": rows[3:]},
                "target_numbers": [4, 5], "completed": [],
                "response_settings": responses,
                "control_grid": deepcopy(grid),
                "control_grids": {str(number): deepcopy(grid) for number in self.features},
                "consequence_panel": [{"crystal": number, "frequency": setting}
                                      for number, setting in zip((4, 5), self.panel_settings)]}


def consequence_snapshot(state, estimators=ESTIMATORS):
    """Query all four joint outcomes using the production family solver."""
    features = {row["crystal"]: {name: reading["value"]
                                 for name, reading in row["measurements"].items()}
                for kind in ("references", "targets")
                for row in state["research_table"][kind]}
    responses, panel = state["response_settings"], state["consequence_panel"]
    base = [query_response_family(features, responses, estimator) for estimator in estimators]
    cost = sum(result["compute_cost"] for result in base)
    signatures = []
    for outcomes in product((False, True), repeat=len(panel)):
        queries = [{**condition, "activated": value} for condition, value in zip(panel, outcomes)]
        results = [query_response_family(features, responses, estimator, queries=queries)
                   for estimator in estimators]
        statuses = [result["status"] for result in results]
        status = ("SAT_WITNESS" if "SAT_WITNESS" in statuses else
                  "UNSAT_VERIFIED" if all(value == "UNSAT_VERIFIED" for value in statuses)
                  else "UNKNOWN")
        signatures.append({"outcomes": list(outcomes), "status": status})
        cost += sum(result["compute_cost"] for result in results)
    return {"panel": deepcopy(panel), "signatures": signatures,
            "possible_signatures": [row["outcomes"] for row in signatures
                                    if row["status"] != "UNSAT_VERIFIED"],
            "families": {estimator["id"]: result["status"]
                         for estimator, result in zip(estimators, base)},
            "compute_cost": cost}


def changed_predictions(before, after, responses):
    """Freeze only new consequences at target settings not previously observed."""
    possible_before, possible_after = before["possible_signatures"], after["possible_signatures"]
    if not possible_after:
        return []
    observed = {(row["crystal"], row["frequency"]) for row in responses}
    changed = []
    for index, condition in enumerate(after["panel"]):
        old = {values[index] for values in possible_before}
        new = {values[index] for values in possible_after}
        if len(old) > 1 and len(new) == 1 and (condition["crystal"], condition["frequency"]) not in observed:
            changed.append({**condition, "predicted_activated": next(iter(new))})
    return changed


def run_case(case, *, max_research_actions=6):
    lab = ResponseLab(case)
    state = lab.initial_state()
    initial = consequence_snapshot(state)
    before = initial
    trace, confirmations, withdrawn_predictions = [], [], []
    pending = {}
    last_decision = None
    research_compute = evaluator_compute = 0
    evaluator_compute += initial["compute_cost"]
    while True:
        proposal = propose_response_research(state, deepcopy(ESTIMATORS), "public_lab.py")
        research_compute += proposal["compute_cost"]
        state["response_research"] = proposal
        family_research = proposal.get("family_research", {})
        last_decision = {"status": proposal["status"],
                         "selected_role": family_research.get("selected_role"),
                         "premise_test": deepcopy(family_research.get("premise_test")),
                         "research_obligation": deepcopy(family_research.get("research_obligation")),
                         "family_statuses": {row["id"]: row["status"]
                                             for row in family_research.get("families", [])},
                         "refuted_family_ids": deepcopy(family_research.get("refuted_family_ids", [])),
                         "no_progress_cycles": family_research.get("no_progress_cycles"),
                         "compute_cost": proposal["compute_cost"]}
        if proposal.get("selected_number") is None:
            break
        point = proposal["selected_number"], proposal["selected_setting"]
        # A reference-derived forecast survives intervening parent research.
        # The evaluator never chooses the action or spends an extra research
        # action once the original six-action limit has been reached.
        forecast = pending.get(point)
        if forecast is None and len(trace) >= max_research_actions:
            break
        response = lab.measure(*point)
        state["response_settings"].append(response)
        after = consequence_snapshot(state)
        evaluator_compute += after["compute_cost"]
        if forecast is not None:
            pending.pop(point)
            confirmations.append({**forecast, "response": response,
                                  "parent_selection_role": family_research.get("selected_role"),
                                  "correct": forecast["predicted_activated"] == response["activated"]})
        else:
            predictions = (changed_predictions(before, after, state["response_settings"])
                           if response["crystal"] <= 3 else [])
            trace.append({"selected_experiment": deepcopy(proposal["selection"]["selected_experiment"]),
                          "decision": deepcopy(last_decision),
                          "response": response, "before": before, "after": after,
                          "new_unobserved_target_predictions": deepcopy(predictions)})
            for prediction in predictions:
                pending[(prediction["crystal"], prediction["frequency"])] = {
                    **prediction, "source_reference_evidence": deepcopy(response["evidence"])}
        # A later observation may undo a conditional prediction. Preserve that
        # withdrawal rather than confirming a claim the parent no longer holds.
        for key, prediction in list(pending.items()):
            index = next(i for i, condition in enumerate(after["panel"])
                         if (condition["crystal"], condition["frequency"]) == key)
            possible = {values[index] for values in after["possible_signatures"]}
            if possible != {prediction["predicted_activated"]}:
                withdrawn_predictions.append({**pending.pop(key),
                                              "after_evidence": deepcopy(response["evidence"])})
        before = after
        if confirmations and not pending:
            break
        # An empty family set must reach the selector again so the genuine
        # research obligation is recorded, not turned into evaluator success.
    final = consequence_snapshot(state)
    evaluator_compute += final["compute_cost"]
    refuted = [family for family, status in final["families"].items()
               if status == "UNSAT_VERIFIED"]
    obligation = (last_decision or {}).get("research_obligation")
    all_refuted = len(refuted) == len(ESTIMATORS)
    confirmed = (bool(confirmations) and all(row["correct"] for row in confirmations)
                 and not pending and not withdrawn_predictions)
    return {"case": case, "initial": initial, "trace": trace,
            "confirmations": confirmations, "final": final,
            "pending_predictions": list(pending.values()),
            "withdrawn_predictions": withdrawn_predictions,
            "last_decision": last_decision,
            "refuted_family_ids": refuted,
            "scientific_diagnosis": {
                "status": "declared_families_refuted" if all_refuted else
                          "some_declared_families_refuted" if refuted else "declared_families_compatible",
                "research_obligation": obligation,
                "same_measurement_conflict": (obligation or {}).get("same_measurement_conflict"),
                "meaning": "Compatibility of declared families with acquired responses; not target confirmation"},
            "prediction_confirmation": {
                "status": "confirmed" if confirmed else "contradicted" if any(
                    not row["correct"] for row in confirmations) else "unconfirmed",
                "confirmed_count": sum(row["correct"] for row in confirmations),
                "meaning": "Reference evidence fixed a previously ambiguous unobserved target prediction, then the parent selected its actual test"},
            "reference_changed_target_prediction": any(
                row["new_unobserved_target_predictions"] for row in trace),
            "new_target_predictions_confirmed": confirmed,
            "costs": {"initial_response_actions": 3, "research_response_actions": len(trace),
                      "confirmation_response_actions": len(confirmations), "model_calls": 0,
                      "research_compute_units": research_compute,
                      "evaluator_compute_units": evaluator_compute},
            "controller_inputs": "public measurements, actual responses, common declared bases and fixed target panel"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-research-actions", type=int, default=6)
    args = parser.parse_args()
    results = [run_case(case, max_research_actions=args.max_research_actions)
               for case in ("parameter_sufficient", "shared_feature_needed", "shared_assumption_false")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"kind": "offline_family_mechanism_pilot", "cases": results}, indent=2) + "\n")
    for result in results:
        print(result["case"], "confirmed:", result["new_target_predictions_confirmed"],
              "families:", result["final"]["families"], "costs:", result["costs"])


if __name__ == "__main__":
    main()
