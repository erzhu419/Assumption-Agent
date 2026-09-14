"""Observed applicability of a learned research step, without task identities.

Profiles describe evidence already bought by research. Prediction losses are
supplied by the caller's existing comparison; building a profile never runs a
second model evaluation or asks the environment for another observation.
"""

from copy import deepcopy
import json
from math import isclose

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
