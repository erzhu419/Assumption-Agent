"""Observed applicability of a learned research step, without task identities.

Profiles describe evidence already bought by research. Prediction losses are
supplied by the caller's existing comparison; building a profile never runs a
second model evaluation or asks the environment for another observation.
"""

from copy import deepcopy
import json
from math import isclose, isfinite

from .method_revision import revision_condition, source_revision


def _record(value):
    return value if isinstance(value, dict) else vars(value)


def _program(value):
    if value is None:
        return None
    return value.program if hasattr(value, "program") else value


def _representation(program, feature_names):
    if program is None:
        return {"kind": "unmodeled", "families": [], "history_depth": 0,
                "structure": None, "program": None, "uses_calls": False}
    names = {name: index for index, name in enumerate(feature_names)}
    operators, accumulators, calls = set(), set(), False

    def walk(node):
        nonlocal calls
        operators.add(node.op)
        if node.op == "call":
            calls = True
            return walk(node.expanded())
        if node.op == "accumulate":
            accumulators.add(node)
        shape = {"op": node.op}
        if node.op in {"feature", "lag"}:
            shape["role"] = "observed_target" if node.name == "__target__" else names.get(node.name)
        if node.lag:
            shape["lag"] = node.lag
        if node.children:
            shape["children"] = [walk(child) for child in node.children]
        return shape

    structure = walk(program)
    families = ["raw"]
    if operators.intersection({"lag", "available", "window_sum", "event_delta", "accumulate"}):
        families.append("history")
    if "accumulate" in operators:
        families.append("continuous")
    if operators.intersection({"object_sum", "neighbor_sum", "window_sum", "event_delta"}):
        families.append("structure")
    return {"kind": families[-1], "families": families, "history_depth": program.history_depth,
            "continuous_coordinates": len(accumulators), "structure": structure,
            "program": program.to_dict(), "uses_calls": calls}


def _coverage(spec, observations, trace):
    # Experiment traces can retain the actual commanded controls even when a
    # coarse or selected-instrument predictor never received individual values.
    controls = []
    for raw in trace:
        record = _record(raw)
        details = record.get("details", {})
        if record.get("action") == "observe" and details.get("sequence"):
            controls.extend(details["sequence"])
    source = controls or [row.features for row in observations]
    coverage = []
    if spec.feature_bounds:
        for index, (name, bounds) in enumerate(spec.feature_bounds.items()):
            values = [float(row[name]) for row in source if name in row]
            low, high = bounds
            bins = sorted({min(3, max(0, int(4 * (value - low) / (high - low))))
                           for value in values}) if high != low else ([0] if values else [])
            coverage.append({"role": index, "observed": bool(values), "bins": bins,
                             "varying": bool(values and max(values) > min(values))})
    else:
        for index, name in enumerate(spec.feature_names):
            values = {row.features[name] for row in observations if name in row.features}
            coverage.append({"role": index, "observed": bool(values),
                             "distinct_support": min(4, len(values)), "varying": len(values) > 1})
    return coverage


def _conditional_coverage(spec, observations):
    """Retain which public controls were explored in each observed response state.

    Marginal coverage can be complete even when, for example, all high controls
    were tried only after the response was already on. Keep startup and the two
    acquired preceding response levels separate; no hidden state is inferred.
    """
    if not spec.feature_bounds or len({row.target for row in observations}) != 2:
        return []
    levels = sorted({row.target for row in observations})
    groups = {}
    for row in observations:
        if not row.history:
            key = "startup"
        elif row.history[-1].get("__target__") in levels:
            key = f"previous_response_{levels.index(row.history[-1]['__target__'])}"
        else:
            continue
        groups.setdefault(key, []).append(row)
    return [{"condition": key, "coverage": _coverage(spec, rows, ())}
            for key, rows in sorted(groups.items())]


def _condition(profile):
    """Keep the finite evidence needed to compare attempts, without nesting traces."""
    representation = profile.get("representation", {})
    data = profile.get("data", {})
    return {"objective": deepcopy(profile.get("objective")),
            "representation": {key: deepcopy(representation.get(key))
                               for key in ("kind", "families", "history_depth", "structure")},
            "data": {key: deepcopy(data.get(key)) for key in (
                "feature_dimensions", "control_dimensions", "coverage", "max_history",
                "independent_trajectories", "linear_data_support", "object_groups", "object_fields",
                "response_levels", "conditional_coverage")},
            "residual": deepcopy(profile.get("residual", {})),
            "boundary": deepcopy(profile.get("boundary", {})),
            "source_revision": revision_condition(profile)}


def method_profile(spec, observations, incumbent=None, trace=(), *, residual_loss=None,
                   baseline_loss=None, reference_loss=None, minimum_improvement=.002, boundary_evidence=None):
    """Summarize public controls, acquired coverage and the current executable model.

    The complete program remains reviewable. Its structural shape ignores
    fitted numerical constants and names roles by their public input order, so
    a renamed task or tiny parameter adjustment is not a new method condition.
    Counts are retained, while repeat detection uses coverage and residuals.
    """
    observations, trace = list(observations), list(trace)
    program = _program(incumbent)
    if spec.decision_costs is not None:
        under, over = spec.decision_costs
        objective = {"loss": "decision", "quantile": under / (under + over)}
    else:
        objective = {"loss": "squared" if spec.risk_quantile == .5 else "pinball",
                     "quantile": spec.risk_quantile}
    schema = spec.context.get("object_schema")
    if isinstance(schema, str):
        schema = json.loads(schema)
    max_history = max((len(row.history) for row in observations), default=0)
    trajectories = sum(not row.history for row in observations)
    # Prefixes can arrive without their startup row; a history reset still
    # distinguishes an additional acquired trajectory in that case.
    trajectories += sum(bool(row.history) and len(row.history) <= len(previous.history)
                        for previous, row in zip(observations, observations[1:]))
    residual = {"state": "unmeasured" if residual_loss is None else
                          "adequate" if residual_loss <= minimum_improvement else "unresolved",
                "loss": residual_loss, "reference_loss": reference_loss, "baseline_loss": baseline_loss,
                "normalized": residual_loss / max(baseline_loss, 1e-8)
                              if residual_loss is not None and baseline_loss is not None else None,
                "new_counterexample": (reference_loss is not None and reference_loss > minimum_improvement and
                                       residual_loss is not None and residual_loss <= minimum_improvement),
                "minimum_improvement": minimum_improvement}
    profile = {"observation_count": len(observations), "objective": objective,
        "representation": _representation(program, spec.feature_names),
        "data": {"feature_dimensions": len(spec.feature_names),
                 "control_dimensions": len(spec.feature_bounds),
                 "controllable": bool(spec.feature_bounds or spec.initial_probes or spec.available_probes),
                 "max_probe_length": spec.max_probe_length, "max_history": max_history,
                 "trajectory_count": trajectories, "independent_trajectories": trajectories > 1,
                 "response_levels": min(3, len({row.target for row in observations})),
                 "linear_data_support": len(observations) >= len(spec.feature_names) + 1,
                 "coverage": _coverage(spec, observations, trace),
                 "conditional_coverage": _conditional_coverage(spec, observations),
                 "object_groups": len(schema["objects"]) if schema else 0,
                 "object_fields": sorted({role for fields in schema["fields"].values() for role in fields})
                                   if schema else []},
        "residual": residual, "recent_actions": []}
    if boundary_evidence:
        profile["boundary"] = {"branches": [
            {"history_depth": row["history_depth"], "previous_state": row.get("previous_state"),
             "relative_width": row["relative_width"],
             "free_control_count": len(row.get("free_controls", []))}
            for row in boundary_evidence.get("boundary_support", [])],
            "unresolved_width": boundary_evidence.get("unresolved_boundary_width", 0.)}
    last = {}
    for raw in trace:
        record = _record(raw)
        if "gain" not in record or "action" not in record or record.get("execution_complete") is False:
            continue
        old = record.get("profile")
        last[record["action"]] = {"action": record["action"], "gain": record["gain"],
                                  "condition": _condition(old) if old else None}
    profile["recent_actions"] = list(last.values())
    revision = source_revision(profile, [_record(item) for item in trace], getattr(incumbent, "memory_id", None))
    if revision:
        profile["source_revision"] = revision
    return profile


def same_research_condition(before, after, next_action=None):
    """Whether another attempt would face the same representation and evidence.

    More rows inside already covered control regions do not themselves reopen
    a failed action. A newly covered region, longer observed dependence, new
    model structure or a material residual change does. Small residual drift
    uses a single relative tolerance in addition to the caller's loss tolerance.
    """
    if not before or not after:
        return False
    left, right = _condition(before), _condition(after)
    if next_action != "state_transition":
        left["data"].pop("response_levels", None)
        right["data"].pop("response_levels", None)
    if next_action not in {"state_transition", "transition", "representative", "task_value"}:
        left["data"].pop("conditional_coverage", None)
        right["data"].pop("conditional_coverage", None)
    for key in ("objective", "representation", "data"):
        if left[key] != right[key]:
            return False
    if left["boundary"] != right["boundary"] or left["source_revision"] != right["source_revision"]:
        return False
    old, new = left["residual"], right["residual"]
    if old.get("state") != new.get("state") or old.get("new_counterexample") != new.get("new_counterexample"):
        return False
    for field in ("loss", "reference_loss"):
        first, second = old.get(field), new.get(field)
        if first is None or second is None:
            if first != second:
                return False
        elif not isclose(first, second, rel_tol=.2,
                         abs_tol=max(old.get("minimum_improvement", .002), new.get("minimum_improvement", .002))):
            return False
    return True


def method_applicability(source_profile, target_profile, next_action):
    """Return supported, explore, or unsupported, with the actual reason.

    This judges a proposed next step; it does not choose actions, exchange
    observation and computation budgets, or import a source task's return as
    a target result. Missing old profiles leave a new trial available.
    """
    def result(status, *reasons):
        return {"status": status, "can_try": status != "unsupported", "transfer_supported": status == "supported",
                "next_action": next_action, "reasons": list(reasons)}

    if not target_profile:
        return result("explore", "target_evidence_not_profiled")
    data = target_profile.get("data", {})
    objective = target_profile.get("objective", {})
    residual = target_profile.get("residual", {})
    revision = revision_condition(target_profile)
    if next_action == "refit" and revision:
        return result("unsupported", "source_parameter_repairs_already_failed_on_the_current_coordinates")
    if next_action == "continuous_state" and not data.get("max_history", 0):
        return result("unsupported", "continuous_state_requires_acquired_history")
    if next_action == "state_transition":
        if not data.get("max_history", 0):
            return result("unsupported", "state_transition_requires_acquired_history")
        if data.get("response_levels") is None:
            return result("explore", "observed_response_levels_not_profiled")
        if data["response_levels"] != 2:
            return result("unsupported", "state_transition_requires_two_observed_response_levels")
    if next_action in {"structure", "structural_discovery"} and not data.get("object_groups", 0):
        return result("unsupported", "object_structure_requires_public_object_fields")
    if next_action in {"calibration", "decision_calibration"} and objective.get("loss") == "squared":
        return result("unsupported", "calibration_requires_a_risk_or_decision_objective")
    if next_action in {"representative", "task_value", "transition"} and not data.get("controllable", False):
        return result("unsupported", "experiment_requires_a_public_probe_interface")
    if next_action == "transition" and data.get("max_probe_length", 0) < 2:
        return result("unsupported", "transition_requires_a_multirow_probe")
    if next_action in {"refit", "representation", "continuous_state", "state_transition", "structure", "structural_discovery"}:
        if residual.get("state") == "adequate" and not residual.get("new_counterexample"):
            return result("unsupported", "no_current_prediction_residual_for_this_revision")
    prior_failure = next((item for item in reversed(target_profile.get("recent_actions", []))
                          if item["action"] == next_action and item["gain"] <= 0), None)
    if prior_failure and prior_failure.get("condition") and same_research_condition(
            prior_failure["condition"], target_profile, next_action):
        return result("unsupported", "same_action_already_unproductive_under_unchanged_evidence")
    if not source_profile:
        return result("explore", "source_method_has_no_saved_profile")
    required_revision = revision_condition(source_profile)
    if required_revision and required_revision != revision:
        return result("unsupported", "method_requires_matching_failed_parameter_repairs")
    if prior_failure:
        return result("explore", "changed_or_unprofiled_evidence_requires_a_new_trial")
    if source_profile.get("objective") != objective:
        return result("explore", "objective_differs_from_source_evidence")
    old_representation = source_profile.get("representation", {})
    representation = target_profile.get("representation", {})
    source_data = source_profile.get("data", {})
    if old_representation.get("families") != representation.get("families"):
        # Representation research is intended to leave the current family. A
        # source that already used lags does not require the target to have
        # adopted lags too; it requires the relevant history to be available.
        if next_action != "representation":
            return result("explore", "current_representation_differs_from_source")
        if data.get("max_history", 0) < old_representation.get("history_depth", 0):
            return result("explore", "source_representation_requires_more_acquired_history")
    if next_action in {"continuous_state", "state_transition", "transition"} and data.get("max_history", 0) < source_data.get("max_history", 0):
        return result("explore", "source_used_longer_observed_trajectories")
    if next_action in {"refit", "representation", "continuous_state"} and not data.get("linear_data_support", False):
        return result("explore", "observations_do_not_yet_identify_even_all_linear_coordinates")
    if data.get("object_fields") != source_data.get("object_fields"):
        return result("explore", "available_object_fields_differ_from_source")
    if next_action in {"transition", "representative", "task_value"}:
        def unresolved_branches(value):
            return {(row["history_depth"], row.get("previous_state"), row["free_control_count"])
                    for row in value.get("boundary", {}).get("branches", []) if row["relative_width"] > 0}
        if unresolved_branches(source_profile) != unresolved_branches(target_profile):
            return result("explore", "unresolved_boundary_conditions_differ_from_source")
    if (next_action in {"transition", "representative", "task_value", "state_transition"}
            and data.get("conditional_coverage", []) != source_data.get("conditional_coverage", [])):
        return result("explore", "response_conditioned_control_coverage_differs_from_source")
    if data.get("coverage") != source_data.get("coverage"):
        return result("explore", "observed_control_coverage_differs_from_source")
    if residual.get("state") == "unmeasured":
        return result("explore", "current_residual_has_not_been_measured")
    return result("supported", "public_objective_representation_and_observed_support_match")


def _completed_method_returns(method_ref, outcomes):
    """Read one measured return per actual invocation, retaining its original basis."""
    from .research_methods import common_method_process_return

    measured, unresolved = [], []
    for outcome in outcomes:
        if outcome.get("method_ref") != method_ref or not outcome.get("feedback_ref"):
            continue
        reference = outcome["feedback_ref"]
        comparison = outcome.get("working_comparison") or {}
        before, after = comparison.get("before_loss"), comparison.get("after_loss")
        complete = (outcome.get("status") == "completed"
            and outcome.get("execution_complete") is not False
            and not any(outcome.get(key) for key in (
                "execution_error", "administrative_censoring", "censored"))
            and comparison.get("complete") is True and comparison.get("same_rows") is True
            and comparison.get("data_role") == "exposed_development"
            and all(type(loss) in (int, float) and isfinite(loss) and loss >= 0
                    for loss in (before, after)))
        if not complete or not outcome.get("profile"):
            if reference not in unresolved:
                unresolved.append(deepcopy(reference))
            continue
        basis = {"kind": "working_comparison", "data_role": "exposed_development"}
        paired = outcome.get("paired_future") or {}
        if (outcome.get("execution_complete") is True
                and paired.get("execution_complete") is not False
                and not paired.get("administrative_censoring") and not paired.get("censored")
                and paired.get("same_rows") is True and paired.get("data_role") == "new_paid_readings"
                and type(paired.get("reference_rows")) is int and paired["reference_rows"] > 0
                and bool(paired.get("source_refs"))
                and all(type(paired.get(key)) in (int, float) and isfinite(paired[key]) and paired[key] >= 0
                        for key in ("before_loss", "after_loss"))):
            before, after = paired["before_loss"], paired["after_loss"]
            basis = {"kind": "paired_future", "data_role": "new_paid_readings",
                "reference_rows": paired["reference_rows"], "source_refs": deepcopy(paired["source_refs"])}
        helpful = after < before
        process = (common_method_process_return(outcome["process_comparison"])
            if outcome.get("process_comparison") else None)
        if process and process["helpful"] is not None:
            helpful = process["helpful"]
            basis = {"kind": "process_comparison", "data_role": "exposed_development",
                "process_return": process}
        measured.append({"feedback_ref": deepcopy(reference),
            "status": "local_improvement" if helpful else "complete_unhelpful",
            "return_basis": basis, "before_loss": before, "after_loss": after,
            "process_return": process, "profile": deepcopy(outcome["profile"]),
            "evidence_basis": deepcopy(outcome.get("evidence_basis"))})
    return measured, unresolved


def _method_observables(profile):
    """Finite, pre-use public state; method outcomes and task identities are absent."""
    selected = {"objective": profile.get("objective", {}),
        "representation": {key: profile.get("representation", {}).get(key)
            for key in ("kind", "families", "history_depth", "continuous_coordinates", "uses_calls")},
        "data": {key: profile.get("data", {}).get(key) for key in (
            "feature_dimensions", "control_dimensions", "controllable", "max_probe_length",
            "max_history", "trajectory_count", "independent_trajectories", "response_levels",
            "linear_data_support", "coverage", "conditional_coverage", "object_groups", "object_fields")},
        "residual": profile.get("residual", {}), "boundary": profile.get("boundary", {}),
        "source_revision": revision_condition(profile),
        "observation_count": profile.get("observation_count")}
    coordinates = {}

    def walk(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], (*path, key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, (*path, index))
        elif isinstance(value, (bool, str)) or type(value) in (int, float) and isfinite(value):
            coordinates[path] = value

    walk(selected, ())
    return coordinates


def _method_coordinates(views):
    coordinates = []
    for path in sorted(set().union(*(set(view) for view in views)), key=str):
        values = [view[path] for view in views if path in view]
        if all(type(value) in (int, float, bool) for value in values):
            coordinates.append({"path": list(path), "kind": "number"})
        else:
            for value in sorted(set(values), key=str):
                coordinates.append({"path": list(path), "kind": "equals", "value": value})
        if len(values) != len(views):
            coordinates.append({"path": list(path), "kind": "available"})
    return coordinates


def _method_features(view, coordinates):
    features = {}
    for index, coordinate in enumerate(coordinates):
        path = tuple(coordinate["path"])
        value = view.get(path)
        kind = coordinate["kind"]
        features[f"public_{index}"] = (float(path in view) if kind == "available" else
            float(value == coordinate["value"]) if kind == "equals" else
            float(value) if type(value) in (int, float, bool) and isfinite(value) else 0.)
    return features


def _method_conflict_basis(method_ref, measured, unresolved, public_experiments):
    return {"method_ref": deepcopy(method_ref), "returns": [
        {**{key: deepcopy(item[key]) for key in (
            "feedback_ref", "status", "return_basis", "before_loss", "after_loss", "process_return")},
         "public_profile": [{"path": list(path), "value": value}
             for path, value in _method_observables(item["profile"]).items()]}
        for item in measured], "unresolved_feedback_refs": deepcopy(unresolved),
        "public_experiments": deepcopy(list(public_experiments))}


def research_method_conflict(method_ref, outcomes, budget, *, target_profile=None,
                             public_experiments=(), previous=None):
    """Execute finite diagnoses and learn use conditions from method-level returns.

    A sample is a completed invocation, not a prediction row. The existing
    observable-condition learner supplies finite halfspaces; competing fits
    remain separate and must agree before they guide another visit. No method
    name is passed to a world-prediction experiment selector. All support here
    is exposed development evidence, including successful separation.
    """
    from .condition_learning import learn_condition
    from .models import Observation

    measured, unresolved = _completed_method_returns(method_ref, outcomes)
    basis = _method_conflict_basis(method_ref, measured, unresolved, public_experiments)
    if previous and previous.get("basis") == basis and not previous.get("budget_incomplete"):
        return {**deepcopy(previous), "compute_cost": 0, "reused": True}
    views = [_method_observables(item["profile"]) for item in measured]
    coordinates = _method_coordinates(views) if views else []
    features = [_method_features(view, coordinates) for view in views]
    rows = [Observation(value, float(item["status"] == "local_improvement"))
        for value, item in zip(features, measured)]
    names = [name for name in features[0] if len({row[name] for row in features}) > 1] if features else []
    support = [row for row in rows if row.target]
    counterexamples = [row for row in rows if not row.target]
    # Reserve the complete finite pass before its first diagnostic. Each
    # coordinate is scanned once; the existing affine learner has k+2 scans.
    required = len(outcomes) + len(rows) * (len(coordinates) + 1)
    required += len(rows) * (2 * len(names) + 2) if len(support) >= 2 and counterexamples else 0
    result = {"method_ref": deepcopy(method_ref), "basis": basis,
        "status": "unresolved", "condition": None, "candidates": [], "diagnostics": [],
        "coordinate_paths": coordinates, "feedback_refs": [item["feedback_ref"] for item in measured],
        "unresolved_feedback_refs": unresolved, "compute_cost": 0, "required_compute": required,
        "budget_incomplete": False, "scope": "exposed_development",
        "independent_support": False, "world_support_updated": False}
    if not support or not counterexamples:
        return {**result, "status": "no_mixed_returns", "required_compute": 0}
    required += len(rows) * (len(rows) - 1) // 2 * (len(coordinates) + 1)
    result["required_compute"] = required
    if not budget.can_spend(required):
        return {**result, "status": "insufficient_budget", "budget_incomplete": True,
            "denied_request": deepcopy(budget.last_denied_request)}
    start = budget.spent
    budget.spend(len(outcomes) + len(rows) * (len(coordinates) + 1)
        + len(rows) * (len(rows) - 1) // 2 * (len(coordinates) + 1))
    collisions = []
    for index, row in enumerate(rows):
        for other, prior in zip(measured[:index], rows[:index]):
            if prior.features == row.features and prior.target != row.target:
                collisions.append([other["feedback_ref"], measured[index]["feedback_ref"]])
    result["diagnostics"].append({"kind": "same_public_view_opposite_returns",
        "status": "conflict_observed" if collisions else "not_observed",
        "witnesses": collisions, "rows": len(rows)})
    # Existing finite coordinate and affine synthesis; no target return is read.
    if len(support) >= 2 and counterexamples:
        for name in [*names, None] if len(names) > 1 else names:
            selected = names if name is None else [name]
            positive = [Observation({key: row.features[key] for key in selected}, 1.) for row in support]
            negative = [Observation({key: row.features[key] for key in selected}, 0.) for row in counterexamples]
            learned = learn_condition(positive, negative, budget, scope_purpose="relative_utility", max_history=0)
            relevant = [coordinates[int(key.removeprefix("public_"))] for key in selected]
            result["diagnostics"].append({"kind": "observable_condition_test", "coordinates": relevant,
                "status": learned["status"], "positive_uses": len(support),
                "negative_uses": len(counterexamples), "feedback_refs": result["feedback_refs"]})
            condition = learned.get("condition")
            if condition and all(item["condition"] != condition for item in result["candidates"]):
                result["candidates"].append({"kind": "omitted_public_use_condition",
                    "condition": condition, "coordinates": relevant,
                    "predicted_returns": [{"feedback_ref": item["feedback_ref"],
                        "helpful": bool(row.target)} for item, row in zip(measured, rows)],
                    "prediction": "The method has local utility only on the true branch of this public-state condition.",
                    "scope": "fits_supplied_development_returns"})
    if result["candidates"]:
        result["condition"] = deepcopy(result["candidates"][0]["condition"])
        result["status"] = "development_conditions_found"
    else:
        result["status"] = "conflicting_public_view" if collisions else "insufficient_distinguishing_returns"
    result["measured_coverage"] = {name: [min(row.features[name] for row in rows),
        max(row.features[name] for row in rows)] for name in features[0]} if features else {}
    result["public_value_coverage"] = [{"path": list(path),
        "values": list(dict.fromkeys(view.get(path) for view in views))}
        for path in sorted(set().union(*(set(view) for view in views)), key=str)
        if any(isinstance(view.get(path), str) for view in views)] if views else []
    result["public_presence_coverage"] = [{"path": list(path),
        "values": sorted({path in view for view in views})}
        for path in sorted(set().union(*(set(view) for view in views)), key=str)] if views else []
    result["development_witnesses"] = [{"feedback_ref": item["feedback_ref"],
        "helpful": item["status"] == "local_improvement", "return_basis": item["return_basis"]}
        for item in measured]
    result["competing_explanations"] = [
        {"kind": "omitted_observable_condition", "status": "development_fit" if result["condition"] else "unresolved",
            "candidate_count": len(result["candidates"])},
        {"kind": "insufficient_method_feedback", "status": "not_excluded",
            "complete_positive_uses": len(support), "complete_negative_uses": len(counterexamples),
            "unresolved_feedback_refs": unresolved},
        {"kind": "method_step_inadequate", "status": "not_identified_by_whole_method_returns",
            "negative_feedback_refs": [item["feedback_ref"] for item in measured
                if item["status"] == "complete_unhelpful"]}]
    result["next_diagnostic"] = {"kind": "bounded_method_trial", "method_ref": deepcopy(method_ref),
        "question": ("Do opposite method returns persist under the same captured public state?"
            if collisions else "Does the development public-state condition predict a completed method return?"),
        "witness_refs": collisions[0] if collisions else deepcopy(result["feedback_refs"]),
        "comparison": {"before": "retained_complete_strong_pool", "after": "same_method_proposal",
            "same_paid_rows": True, "complete_required": True,
            "quality_coverage_then_resources": True},
        "required_evidence": (["complete_step_comparisons_at_the_conflicting_public_state"] if collisions else
            ["completed_return_on_a_new_public_state_before_condition_confirmation"]),
        "public_experiments": deepcopy(list(public_experiments)), "execution_status": "not_run"}
    if target_profile:
        result["next_diagnostic"]["candidate_predictions"] = _conflict_predictions(result, target_profile)
    result["compute_cost"] = budget.spent - start
    result["paid_compute_cost"] = result["compute_cost"]
    return result


def _conflict_predictions(research, profile):
    from .language import Program
    from .models import Observation

    view = _method_observables(profile)
    features = _method_features(view, research.get("coordinate_paths", []))
    return [{"candidate": index, "helpful": bool(Program.from_dict(item["condition"]).predict(
        Observation(features, 0.)))} for index, item in enumerate(research.get("candidates", []))]


def _conflict_use_decision(research, method_ref, profile, measured, unresolved):
    if not research or research.get("method_ref") != method_ref or research.get("budget_incomplete"):
        return None
    expected = _method_conflict_basis(method_ref, measured, unresolved,
        research.get("basis", {}).get("public_experiments", []))
    if expected != research.get("basis"):
        return {"status": "stale_conflict_diagnosis", "reason": "method_feedback_changed_requires_new_diagnosis"}
    predictions = _conflict_predictions(research, profile)
    view = _method_observables(profile)
    features = _method_features(view, research.get("coordinate_paths", []))
    outside = [name for name, (low, high) in research.get("measured_coverage", {}).items()
        if not low <= features[name] <= high]
    outside.extend(str(item["path"]) for item in research.get("public_value_coverage", [])
        if view.get(tuple(item["path"])) not in item["values"])
    outside.extend(str(item["path"]) for item in research.get("public_presence_coverage", [])
        if (tuple(item["path"]) in view) not in item["values"])
    if outside or not predictions or len({item["helpful"] for item in predictions}) != 1:
        return {"status": "unresolved_method_use", "candidate_predictions": predictions,
            "reason": "outside_measured_public_coverage" if outside else
                "competing_development_conditions_disagree" if predictions else "no_executable_condition",
            "outside_coverage": outside, "next_diagnostic": deepcopy(research.get("next_diagnostic"))}
    helpful = predictions[0]["helpful"]
    return {"status": "development_condition_prediction", "admission": "admit" if helpful else "defer",
        "next_action": "try_or_continue_method" if helpful else "ordinary_research",
        "reason": "finite_learned_method_use_policy_on_measured_public_coverage",
        "candidate_predictions": predictions, "next_diagnostic": deepcopy(research.get("next_diagnostic"))}


def method_use_decision(method_ref, profile, outcomes, *, evidence_basis=None,
                        next_action="representation", conflict_research=None):
    """Use completed method feedback to price another bounded research visit.

    Outcomes belong to the exact invoked method version. A complete paired
    working comparison measures a local development return. A completed paired
    comparison on new paid readings revises that return; a single predictor's
    confirmation cannot do so. Neither establishes general scientific support.
    An unhelpful visit defers only the same public research
    condition and binding/paid-evidence basis; changed evidence remains open to
    exploration. Unfinished, exceptional or administratively censored work
    retains its references without becoming a negative method example.
    """
    if not method_ref.get("id") or type(method_ref.get("version")) is not int:
        raise ValueError("method use needs the invoked method id and version")
    method_ref, profile = deepcopy(method_ref), deepcopy(profile)
    measured, unresolved = _completed_method_returns(method_ref, outcomes)
    for item in measured:
        item["same_condition"] = same_research_condition(item["profile"], profile, next_action)
        item["same_evidence"] = bool(evidence_basis) and item["evidence_basis"] == evidence_basis
        item["applicability"] = method_applicability(item["profile"], profile, next_action)

    result = {"method_ref": method_ref, "admission": "bounded_exploration",
        "next_action": "distinguish_method_use", "status": "unresolved_method_use",
        "feedback_refs": [item["feedback_ref"] for item in measured],
        "unresolved_feedback_refs": unresolved,
        "scope": {"kind": "method_use", "data_role": "exposed_development",
            "profile": profile, "evidence_basis": deepcopy(evidence_basis),
            "independent_support": False, "world_support_updated": False},
        "independent_support": False}
    exact = [item for item in measured if item["same_condition"] and item["same_evidence"]]
    # A later unknown attempt cannot erase an already measured return. Different
    # methods, versions or public conditions never inherit this deferral.
    if exact:
        last = exact[-1]
        result.update(status=last["status"], deciding_feedback_ref=last["feedback_ref"],
            applicability=last["applicability"], deciding_return_basis=deepcopy(last["return_basis"]))
        if last["status"] == "complete_unhelpful":
            result.update(admission="defer", next_action="defer_until_new_evidence",
                reason="completed_unhelpful_under_same_binding_and_paid_evidence")
        else:
            result.update(admission="admit", next_action="try_or_continue_method",
                reason=("paired_later_improvement_supports_bounded_use"
                    if last["return_basis"]["kind"] == "paired_future" else
                    "local_working_improvement_supports_bounded_use"))
        return result
    learned = _conflict_use_decision(conflict_research, method_ref, profile, measured, unresolved)
    if learned:
        result["conflict_research"] = learned
        if learned.get("admission"):
            result.update(learned)
            return result
    conditioned = [item for item in measured if item["same_condition"]]
    positive = next((item for item in reversed(conditioned)
        if item["status"] == "local_improvement"), None)
    mixed_returns = positive is not None and any(
        item["status"] == "complete_unhelpful" for item in conditioned)
    if positive and evidence_basis and not mixed_returns:
        result.update(admission="admit", next_action="try_or_continue_method",
            status="local_improvement", deciding_feedback_ref=positive["feedback_ref"],
            applicability=positive["applicability"], deciding_return_basis=deepcopy(positive["return_basis"]),
            reason="prior_local_return_supports_a_new_bounded_trial_not_confirmation")
        return result
    result["reason"] = ("mixed_local_returns_require_distinguishing_research_conditions"
        if mixed_returns else "changed_public_condition_or_evidence_requires_a_new_bounded_trial"
        if measured else "no_complete_comparable_method_return")
    result["question"] = {
        "kind": "method_use", "answer_type": "research_action",
        "question": "Under the current public research conditions, should this method be tried or continued, "
            "should ordinary research continue, or is distinguishing evidence or a method revision needed?",
        "method_ref": deepcopy(method_ref),
        "query_semantics": {"mode": "solution_relation"},
        "alternatives": ["try_or_continue_method", "ordinary_research",
            "acquire_distinguishing_evidence", "revise_method"],
        "evidence_refs": deepcopy(result["feedback_refs"] + unresolved),
        "parent_use": {"kind": "select_research_action", "method_ref": deepcopy(method_ref)},
        "required_output": "Compare executable research options under the declared budget; "
            "keep same-evidence working returns, unused-result checks and incomplete execution separate."}
    if learned:
        result["question"]["diagnostic_plan"] = deepcopy(learned.get("next_diagnostic"))
        result["question"]["candidate_predictions"] = deepcopy(learned.get("candidate_predictions", []))
        result["reason"] = learned["reason"]
    return result
