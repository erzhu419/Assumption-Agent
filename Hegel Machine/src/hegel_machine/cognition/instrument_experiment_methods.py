"""Reuse and revise purchased procedures for distinguishing observation laws."""

from copy import deepcopy

from .instrument_hypotheses import (
    instrument_candidates, forecast_instrument_sequence, instrument_boundary_sequences,
)
from .models import Probe, SearchBudget
from .program_experiment_methods import _control_roles


_MODES = {"pulse", "step", "random", "learned_boundary"}


def _recipe(sequence, roles, bounds, mode, target_strategy=None):
    if mode == "learned_boundary":
        return {"sequence_mode": mode, "control_role_count": len(roles), "target_strategy": target_strategy}
    return {"length": len(sequence), "sequence_mode": mode, "control_role_count": len(roles),
            "sequence_fractions": [[(point[name] - bounds[name][0]) / (bounds[name][1] - bounds[name][0])
                                    if bounds[name][1] > bounds[name][0] else .5
                                    for name in roles] for point in sequence]}


def _sequence(recipe, roles, bounds):
    center = {name: (low + high) / 2 for name, (low, high) in bounds.items()}
    return tuple({**center, **{name: bounds[name][0] + fraction * (bounds[name][1] - bounds[name][0])
                              for name, fraction in zip(roles, row)}}
                 for row in recipe["sequence_fractions"])


def _blocked(book, recipe, condition):
    return any(method.get("solver_strategy") == "instrument_sequence"
               and method["instrument_recipe"] == recipe
               and any(scope["condition"] == condition for scope in method.get("failure_scopes", []))
               for method in book.records)


def propose_instrument_method(book, model, command_spec, measured_spec, budget, *,
                              max_length=None, serial=0, tolerance=1e-7):
    """Resolve stored boundary questions or execute a stored numeric preparation.

    A boundary recipe uses the current candidates' cuts, not old coordinates.
    Numeric recipes may extend after a complete forecast finds no separation.
    Planning never changes measured support in the method book.
    """
    start = budget.spent
    fields, bounds = instrument_candidates(model, measured_spec), command_spec.feature_bounds
    roles = _control_roles([program for choices in fields.values() for program in choices.values()], {}, bounds)
    condition = {"programs": {name: {key: program.to_dict() for key, program in choices.items()}
                              for name, choices in fields.items()},
                 "bounds": {name: list(interval) for name, interval in bounds.items()}}
    result = {"probe": None, "predictions": {}, "field_names": [], "compute_cost": 0,
              "reason": "no_matching_instrument_method", "method_attempts": [],
              "instrument_condition": condition, "control_roles": roles}
    cap = min(command_spec.max_probe_length,
              command_spec.max_probe_length if max_length is None else max_length)
    supported = [method for method in reversed(book.records)
                 if method.get("solver_strategy") == "instrument_sequence" and method["status"] == "supported"]
    pending = []

    def finish():
        result["compute_cost"] = budget.spent - start
        book.last_proposal = deepcopy({key: result[key] for key in
            ("reason", "method_attempts", "compute_cost", "method_id", "parent_method_id", "preparation_revision")
            if key in result})
        book.last_proposal["status"] = "proposed" if result["probe"] is not None else "no_match"
        return result

    def forecast(method, sequence, recipe, *, revised=False):
        if _blocked(book, recipe, condition):
            reason, predicted = "measured_failure_under_current_condition", None
        elif len(sequence) * sum(map(len, fields.values())) > budget.remaining:
            reason, predicted = "insufficient_compute", None
        else:
            predicted = forecast_instrument_sequence(fields, sequence, budget, tolerance)
            reason = ("incomplete_forecast" if not predicted["complete"] else
                      "predicted_disagreement" if predicted["predictions"] else "no_predicted_disagreement")
        result["method_attempts"].append({"method_id": method["method_id"], "reason": reason,
                                           "length": len(sequence), "revised": revised})
        if reason == "predicted_disagreement":
            result.update(probe=Probe(f"instrument_method:{serial}:{method['method_id']}:{len(sequence)}", sequence),
                predictions=predicted["predictions"], field_names=sorted(predicted["predictions"]),
                predicted_separation=predicted["predicted_separation"],
                field_disagreements=predicted["field_disagreements"], sequence_mode=recipe["sequence_mode"],
                reason="revised_instrument_experiment_method" if revised else "learned_instrument_experiment_method")
            if revised:
                result.update(parent_method_id=method["method_id"], preparation_revision={
                    "reason": "extend_observation_prefix", "previous_recipe": deepcopy(method["instrument_recipe"]),
                    "revised_recipe": deepcopy(recipe)})
            else:
                result["method_id"] = method["method_id"]
            if recipe["sequence_mode"] == "learned_boundary":
                result["target_strategy"] = recipe["target_strategy"]
        return reason

    if not fields or not bounds:
        return finish()
    for method in supported:
        recipe = method["instrument_recipe"]
        if recipe["control_role_count"] != len(roles):
            result["method_attempts"].append({"method_id": method["method_id"], "reason":
                "control_roles_changed"})
            continue
        if recipe["sequence_mode"] == "learned_boundary":
            local = SearchBudget(budget.remaining // 4)
            choices = instrument_boundary_sequences(fields, command_spec, local, max_length=cap)
            budget.spend(local.spent)
            sequences = [sequence for strategy, sequence in choices if strategy == recipe["target_strategy"]]
            if not sequences:
                result["method_attempts"].append({"method_id": method["method_id"],
                    "reason": "no_current_boundary_sequence"})
            for sequence in sequences:
                forecast(method, sequence, recipe)
                if result["probe"] is not None:
                    return finish()
            continue
        if recipe["length"] > cap:
            result["method_attempts"].append({"method_id": method["method_id"],
                                             "reason": "preparation_exceeds_cap"})
            continue
        sequence = _sequence(recipe, roles, bounds)
        reason = forecast(method, sequence, recipe)
        if result["probe"] is not None:
            return finish()
        if reason == "no_predicted_disagreement":
            pending.append((method, sequence))
    depth = max([model.get("observed_history_depth", 0),
                 *(field.get("observed_history_depth", 0) for field in model.get("fields", {}).values())])
    for method, sequence in pending:
        length = min(cap, max(len(sequence) + 1, depth + 2))
        if length <= len(sequence):
            continue
        extended = (*sequence, *[sequence[-1]] * (length - len(sequence)))
        recipe = _recipe(extended, roles, bounds, method["sequence_mode"])
        forecast(method, extended, recipe, revised=True)
        if result["probe"] is not None:
            return finish()
    if pending:
        method = pending[0][0]
        result.update(reason="stored_preparation_no_longer_separates", parent_method_id=method["method_id"],
            preparation_revision={"reason": "no_predicted_disagreement",
                                  "previous_recipe": deepcopy(method["instrument_recipe"])})
    return finish()


def observe_instrument_method(book, plan, assessment, model, command_spec, *, task_id,
                              observation_rows, measurement_cost, tolerance=1e-7):
    """Support a procedure only when actual field readings eliminate alternatives.

    Rejecting every proposed law exposes a failed applicability condition. It
    cannot support a method that claims to leave a usable observation model.
    """
    probe, predicted = plan.get("probe"), plan.get("predictions", {})
    if not observation_rows or probe is None or not predicted or plan.get("sequence_mode") not in _MODES:
        return None
    scored = {key: value for values in assessment.get("field_losses", {}).values() for key, value in values.items()}
    actual = {item["hypothesis_id"]: item for field in model.get("fields", {}).values()
              for item in field.get("hypotheses", [])}
    planned = {key for choices in predicted.values() for key in choices}
    if not planned <= scored.keys() or not planned <= actual.keys():
        return None
    refuted = sorted(key for key in planned if actual[key]["status"] == "refuted")
    survivors = sorted(key for key in planned if actual[key]["status"] == "supported")
    success = bool(refuted) and all(any(key in survivors for key in choices) for choices in predicted.values())
    evidence = {"task_id": task_id, "probe_id": probe.probe_id,
                "observation_rows": observation_rows, "measurement_cost": measurement_cost,
                "compute_cost": plan.get("compute_cost", 0) + assessment.get("compute_cost", 0),
                "field_losses": deepcopy(assessment["field_losses"]), "refuted_hypotheses": refuted,
                "surviving_hypotheses": survivors,
                "status": "distinguished" if success else "all_refuted" if not survivors else "unresolved"}
    method = next((item for item in book.records if item["method_id"] == plan.get("method_id")), None)
    if method is not None and success:
        method["support"].append(evidence)
        return {"action": "use_instrument_experiment_method", "method_id": method["method_id"],
                "task_id": task_id, "source_task_id": method["source_task_id"],
                "cross_task": task_id != method["source_task_id"], "outcome": "distinguished"}
    recipe = _recipe(probe.sequence, plan["control_roles"], command_spec.feature_bounds,
                     plan["sequence_mode"], plan.get("target_strategy"))
    parent = next((item for item in book.records if item["method_id"] == plan.get("parent_method_id")), None)
    if method is None:
        method = {"method_id": f"program_experiment_{len(book.records) + 1}",
                  "solver_strategy": "instrument_sequence", "status": "supported" if success else "counterexample",
                  "source_task_id": task_id, "source_probe": probe.probe_id,
                  "sequence_mode": plan["sequence_mode"], "source_control_roles": list(plan["control_roles"]),
                  "instrument_recipe": recipe, "support": [], "counterexamples": [], "failure_scopes": [],
                  "steps": ["rebind_public_control_roles",
                            "resolve_current_candidate_boundaries" if plan["sequence_mode"] == "learned_boundary"
                            else "execute_fixed_preparation",
                            "forecast_current_instrument_hypotheses", "purchase_and_compare_fields"]}
        if parent is not None and recipe != parent["instrument_recipe"]:
            method.update(parent_method_id=parent["method_id"], preparation_revision={
                **deepcopy(plan.get("preparation_revision", {})),
                "previous_recipe": deepcopy(parent["instrument_recipe"]), "revised_recipe": deepcopy(recipe)})
        book.records.append(method)
    if not success:
        scope = {"reason": evidence["status"], "condition": deepcopy(plan["instrument_condition"])}
        method["counterexamples"].append(evidence)
        if scope not in method["failure_scopes"]:
            method["failure_scopes"].append(scope)
        return {"action": "revise_instrument_experiment_method", "method_id": method["method_id"],
                "task_id": task_id, "status": "condition_restricted", "reason": scope["reason"], "scope": scope}
    method["support"].append(evidence)
    return {"action": "learn_instrument_experiment_method", "method_id": method["method_id"],
            "task_id": task_id, "solver_strategy": "instrument_sequence",
            **({"parent_method_id": method["parent_method_id"],
                "preparation_revision": deepcopy(method["preparation_revision"])} if "parent_method_id" in method else {})}
