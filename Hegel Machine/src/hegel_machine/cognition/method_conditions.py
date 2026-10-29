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

    def walk(node, replaying=False):
        nonlocal calls
        operators.add(node.op)
        if node.op == "call":
            calls = True
            return walk(node.expanded(), replaying)
        if node.op == "accumulate":
            accumulators.add(node)
        shape = {"op": node.op}
        if node.op in {"feature", "lag"}:
            if node.name == "__target__":
                shape["role"] = "predicted_state" if replaying and node.op == "lag" else "observed_target"
            else:
                shape["role"] = names.get(node.name)
        if node.lag:
            shape["lag"] = node.lag
        if node.children:
            shape["children"] = [walk(child, replaying or node.op == "state_replay")
                                 for child in node.children]
        return shape

    structure = walk(program)
    families = ["raw"]
    if operators.intersection({"lag", "available", "window_sum", "event_delta", "accumulate", "state_replay"}):
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


def _semantic_coverage(data):
    """Key set-valued coverage by meaning; do not reorder temporal sequences."""
    result = deepcopy(data)
    if isinstance(result.get('coverage'), list):
        result['coverage'] = {str(item['role']): {key: value for key, value in item.items() if key != 'role'}
            for item in result['coverage']}
    if isinstance(result.get('conditional_coverage'), list):
        result['conditional_coverage'] = {item['condition']: {
            str(value['role']): {key: child for key, child in value.items() if key != 'role'}
            for value in item.get('coverage', [])} for item in result['conditional_coverage']}
    for values in [result.get('coverage') or {}, *(result.get('conditional_coverage') or {}).values()]:
        for item in values.values():
            if isinstance(item.get('bins'), list):
                item['bins'] = sorted(set(item['bins']))
    return result


def _condition(profile):
    """Keep the finite evidence needed to compare attempts, without nesting traces."""
    representation = profile.get("representation", {})
    data = profile.get("data", {})
    return {"objective": deepcopy(profile.get("objective")),
            "representation": {key: deepcopy(representation.get(key))
                               for key in ("kind", "families", "history_depth", "structure")},
            "data": _semantic_coverage({key: deepcopy(data.get(key)) for key in (
                "feature_dimensions", "control_dimensions", "coverage", "max_history",
                "independent_trajectories", "linear_data_support", "object_groups", "object_fields",
                "response_levels", "conditional_coverage")}),
            "residual": deepcopy(profile.get("residual", {})),
            "boundary": deepcopy(profile.get("boundary", {})),
            "source_revision": revision_condition(profile)}


def paid_transition_evidence(records, rows, predictions, *, guard_depth=None,
                             squared_error_threshold=1e-10, source_refs=()):
    """Locate first closed-loop errors in complete paid reset trajectories.

    Later errors do not independently refute a mature transition: its predicted
    state may already be wrong. Missing records or predictions remain unknown.
    """
    from dataclasses import asdict

    rows = json.loads(json.dumps([asdict(row) if hasattr(row, 'features') else row for row in rows]))
    records = json.loads(json.dumps(records))
    result = {'status': 'unknown', 'source_refs': deepcopy(list(source_refs))}
    if not rows or len(predictions) != len(rows):
        return {**result, 'reason': 'saved_prediction_coverage_unavailable'}
    indices = [index for record in records for index in record.get('row_indices', [])]
    if sorted(indices) != list(range(len(rows))):
        return {**result, 'reason': 'complete_reset_records_unavailable'}
    for record in records:
        selected = [rows[index] for index in record['row_indices']]
        if (record.get('rows') != selected or not selected
                or any(len(row.get('history', [])) != position for position, row in enumerate(selected))):
            return {**result, 'reason': 'reset_records_do_not_match_paid_rows'}
    known_guard = isinstance(guard_depth, int) and guard_depth > 0
    counts = {'startup_first_divergences': 0 if known_guard else None,
              'mature_first_divergences': 0 if known_guard else None,
              'correct_prefix_first_divergences': 0, 'later_errors_unattributed': 0,
              'error_rows': 0, 'correct_trajectories': 0, 'trajectory_count': len(records)}
    for record in records:
        errors = [position for position, index in enumerate(record['row_indices'])
                  if (predictions[index] - rows[index]['target']) ** 2 > squared_error_threshold]
        counts['error_rows'] += len(errors)
        if not errors:
            counts['correct_trajectories'] += 1
            continue
        counts['later_errors_unattributed'] += len(errors) - 1
        counts['correct_prefix_first_divergences'] += errors[0] > 0
        if known_guard:
            key = 'startup_first_divergences' if errors[0] < guard_depth else 'mature_first_divergences'
            counts[key] += 1
    return {**result, **counts, 'status': 'observed', 'guard_depth': guard_depth,
            'squared_error_threshold': squared_error_threshold}


def method_profile(spec, observations, incumbent=None, trace=(), *, residual_loss=None,
                   baseline_loss=None, reference_loss=None, minimum_improvement=.002, boundary_evidence=None,
                   research_state=None, transition_evidence=None):
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
    if transition_evidence is not None and transition_evidence.get('status') == 'observed':
        residual['transition'] = {key: deepcopy(value) for key, value in transition_evidence.items()
                                  if key != 'source_refs'}
        profile['transition_evidence_refs'] = deepcopy(transition_evidence.get('source_refs', []))
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
    if research_state is not None:
        profile['research'] = deepcopy(research_state)
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
    data = _semantic_coverage(target_profile.get("data", {}))
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
    source_data = _semantic_coverage(source_profile.get("data", {}))
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
    # Coverage lists are sets of named scientific conditions/roles. Other
    # lists (in particular ordered histories) retain their order.
    selected['data'] = _semantic_coverage(selected['data'])
    research = profile.get('research') or {}
    selected['research'] = {'stage': research.get('stage')}
    for key, names in {
        'last_unfinished_request': ('operator', 'status', 'stage'),
        'evidence': ('unconsumed_answer', 'new_distinction', 'distinguishing_experiment_available'),
        'budget': ('complete_unit_payable', 'minimum_compute', 'remaining_compute', 'remaining_seconds'),
        'comparison': ('stale', 'paid_rows', 'models_requested', 'models_completed'),
        'candidate_pool': ('status', 'models_requested', 'known_executions',
            'unknown_prediction_models', 'guarded_executions', 'unknown_guard_executions',
            'min_mature_first_divergences', 'startup_only_first_error_executions',
            'correct_executions'),
    }.items():
        value = research.get(key) or {}
        selected['research'][key] = {name: deepcopy(value[name]) for name in names if name in value}
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
        if all(type(value) in (int, float, bool) for value in values) and len(values) == len(views):
            coordinates.append({"path": list(path), "kind": "number"})
        elif len(values) == len(views) and not all(type(value) in (int, float, bool) for value in values):
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
            (float(value == coordinate["value"]) if path in view else None) if kind == "equals" else
            float(value) if type(value) in (int, float, bool) and isfinite(value) else None)
    return features


def method_condition_features(profile, coordinate_paths):
    """Project a learned guard's public coordinates; missing numbers are None.

    The ordinary consumer must fall back to its original policy when any
    required numeric coordinate is unavailable, instead of treating it as 0.
    Availability coordinates themselves remain known booleans encoded as 0/1.
    """
    return _method_features(_method_observables(profile), coordinate_paths)


def method_condition_support(profile, research):
    """Check the existing public measured range, never infer a world domain."""
    view = _method_observables(profile)
    features = _method_features(view, research.get('coordinate_paths', []))
    outside = [name for name, (low, high) in research.get('measured_coverage', {}).items()
        if features.get(name) is None or not low <= features[name] <= high]
    outside.extend(str(item['path']) for item in research.get('public_value_coverage', [])
        if view.get(tuple(item['path'])) not in item['values'])
    outside.extend(str(item['path']) for item in research.get('public_presence_coverage', [])
        if (tuple(item['path']) in view) not in item['values'])
    return {'can_apply': bool(research.get('measured_coverage')) and not outside,
        'outside_coverage': outside}


def _method_conflict_basis(method_ref, measured, unresolved, public_experiments):
    return {"method_ref": deepcopy(method_ref),
        "condition_search": "complete_public_coordinate_and_affine_pass",
        "returns": [
        {**{key: deepcopy(item[key]) for key in (
            "feedback_ref", "status", "return_basis", "before_loss", "after_loss", "process_return")},
         "public_profile": [{"path": list(path), "value": value}
             for path, value in _method_observables(item["profile"]).items()]}
        for item in measured], "unresolved_feedback_refs": deepcopy(unresolved),
        "public_experiments": deepcopy(list(public_experiments))}


def research_method_conflict(method_ref, outcomes, budget, *, target_profile=None,
                             public_experiments=(), previous=None, paired_returns=None,
                             max_conditions=None):
    """Execute finite diagnoses and learn use conditions from method-level returns.

    A sample is a completed invocation, not a prediction row. The existing
    observable-condition learner supplies finite halfspaces; competing fits
    remain separate and must agree before they guide another visit. No method
    name is passed to a world-prediction experiment selector. All support here
    is exposed development evidence, including successful separation.
    """
    from .condition_learning import learn_condition
    from .models import Observation

    measured, unresolved = (_completed_method_returns(method_ref, outcomes)
        if paired_returns is None else paired_returns)
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
        proposals = [*names, None] if len(names) > 1 else names
        # The bound limits executable conditions, not failed coordinate tests.
        # The complete coordinate and affine pass is already reserved above.
        for name in proposals:
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
            if max_conditions is not None and len(result['candidates']) >= max_conditions:
                break
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
    if any(value is None for value in features.values()):
        return []
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
    outside = method_condition_support(profile, research)['outside_coverage']
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


def method_delivery_outcome(outcome):
    """Separate method delivery reliability from world evidence and censorship."""
    status = outcome.get('status')
    if outcome.get('administrative_censoring') or outcome.get('censored') or status in {
            'administrative_pause', 'missing_source', 'administratively_censored'}:
        return 'administratively_censored'
    if status in {'resource_exhausted', 'budget_exhausted', 'time_exhausted',
            'window_exhausted_paid_work_preserved', 'compute_budget_exhausted',
            'python_execution_budget_exhausted', 'insufficient_complete_step_budget',
            'method_step_limit', 'observation_budget_exhausted', 'confirmation_budget_exhausted'}:
        return 'resource_non_delivery'
    if outcome.get('execution_error') or status in {'execution_error', 'inexecutable', 'action_failed', 'method_error'}:
        return 'execution_failure'
    return 'delivered' if outcome.get('execution_complete') is True else 'unresolved'


def paired_method_identification(pair):
    """State what this actual pair can identify, before deriving a label.

    The common ``data`` basis is the input at the decision point. Experiments
    chosen inside a method are part of that strategy, not a fixed data factor.
    A single pair cannot identify a separate data effect or an interaction.
    """
    def queries(content):
        entries = content if isinstance(content, list) else [content]
        return [deepcopy(item.get('arguments', {}).get('actions', []))
                for item in entries if isinstance(item, dict) and item.get('action') == 'query']

    contents = [pair.get(f'{side}_action_content') for side in ('baseline', 'challenger')]
    acquisitions = [queries(content) for content in contents]
    sides = [pair.get(side) or {} for side in ('baseline', 'challenger')]
    observation_deltas = [side.get('costs', {}).get('observation_cost') for side in sides]
    # A query can consume an already paid answer with no new material. Conversely
    # internal acquisition can add rows without a top-level query in this view.
    fixed_material = all(type(delta) in (int, float) and delta == 0 for delta in observation_deltas)
    effect = 'fixed_data_method_effect' if fixed_material else 'research_strategy_effect'
    evaluation_use = ('exposed_paid_working_comparison'
        if pair.get('evaluation_role') == 'paid_working_comparison'
        else 'exposed_development_label_after_outputs_freeze')
    return {'identified_effect': effect, 'acquisition_actions_differ': acquisitions[0] != acquisitions[1],
        'new_observation_costs': observation_deltas,
        'final_data_refs': [deepcopy(side.get('continuation_data_ref')) for side in sides],
        'data_and_method_separately_identified': False,
        'separate_data_effect_requires': 'fixed_D1_crossed_methods_with_matching_complete_cells',
        'evaluation_use': evaluation_use,
        'independent_confirmation': False}


def paired_method_preference(pair):
    """Compare actually executed alternatives at the same declared start.

    ``basis`` contains the resolved start/data/split/quality/resource/evaluation
    contracts. A caller's same_start boolean cannot supply those contracts.
    """
    sides = [pair.get(name) or {} for name in ('baseline', 'challenger')]
    result = {'preference': 'UNKNOWN', 'reason': 'matched_executed_pair_required',
        'feedback_ref': deepcopy(pair.get('feedback_ref')),
        'deliveries': [method_delivery_outcome(side) for side in sides],
        'world_support_updated': False, **paired_method_identification(pair)}
    if pair.get('comparison_kind') == 'additional_research_vs_retained_parent':
        return {**result, 'reason': 'retained_parent_contrast_is_not_a_method_preference',
            'identified_effect': 'local_incremental_research_value'}
    # Fixed-model supplement receipts explicitly retain the failed original
    # pair. They cannot become method labels by adding a loss to that record.
    if any(side.get('original_pair_completed') is False for side in sides):
        return {**result, 'reason': 'fixed_model_readout_is_not_an_executed_method_pair'}
    keys = ('start_state', 'data', 'split', 'quality_contract', 'resource_contract', 'evaluation')
    basis = [side.get('basis', {}) for side in sides]
    if any(key not in item or item[key] is None for key in keys for item in basis):
        return {**result, 'reason': 'complete_matching_basis_required'}
    if any(basis[0][key] != basis[1][key] for key in keys):
        return {**result, 'reason': 'start_data_or_evaluation_contract_differs'}
    for side in sides:
        limits = side.get('registered_time_limits', side.get('basis', {}).get(
            'resource_contract', {}).get('time_limits', {}))
        cap = limits.get('active_wall_seconds')
        if type(cap) in (int, float):
            elapsed = side.get('timing', {}).get('activity_wall_seconds')
            if type(elapsed) not in (int, float):
                return {**result, 'reason': 'actual_activity_duration_required'}
            if elapsed > cap:
                return {**result, 'reason': 'registered_activity_limit_exceeded'}
    lifecycle = pair.get('evaluation_lifecycle')
    if lifecycle is not None and (lifecycle.get('all_outputs_frozen_before_labels') is not True
            or lifecycle.get('labels_revealed') is not True
            or lifecycle.get('role') != 'development_after_freeze'):
        return {**result, 'reason': 'evaluation_labels_require_all_outputs_frozen_first'}
    target = pair.get('identification_target')
    if target in {'data_effect', 'method_data_interaction'}:
        return {**result, 'reason': 'fixed_D1_crossed_complete_cells_required'}
    if target == 'fixed_data_method_effect' and result['identified_effect'] != target:
        return {**result, 'reason': 'own_acquisition_identifies_research_strategy_only'}
    if _method_observables(sides[0].get('profile', {})) != _method_observables(sides[1].get('profile', {})):
        return {**result, 'reason': 'public_pre_use_profiles_differ'}
    if any(delivery != 'delivered' for delivery in result['deliveries']):
        return {**result, 'reason': 'non_delivery_or_censoring_is_not_a_prediction_label'}
    quality = [side.get('quality') or {'loss': side.get('loss'),
        'threshold': (side.get('basis', {}).get('quality_contract') or {}).get('threshold')}
        for side in sides]
    numeric = lambda value: type(value) in (int, float) and isfinite(value) and value >= 0
    if (any(not numeric(q.get('loss')) or not numeric(q.get('threshold')) for q in quality)
            or quality[0]['threshold'] != quality[1]['threshold']
            or any(type(side.get('completed')) is not bool or type(side.get('coverage')) is not int
                   or side['coverage'] <= 0 for side in sides)):
        return {**result, 'reason': 'actual_completion_quality_and_coverage_required'}
    passed = [q['loss'] <= q['threshold'] for q in quality]
    dimensions = [[int(side['completed']), int(passed[i]), side['coverage']]
        for i, side in enumerate(sides)]
    if not all(passed):
        for row, q in zip(dimensions, quality):
            row.append(-q['loss'])
    changes = [b - a for a, b in zip(*dimensions)]
    if any(change > 0 for change in changes) and any(change < 0 for change in changes):
        return {**result, 'preference': 'incomparable', 'reason': 'quality_coverage_tradeoff'}
    if any(changes):
        return {**result, 'preference': 'challenger' if any(c > 0 for c in changes) else 'baseline',
            'reason': 'quality_or_coverage_dominance', 'losses': [q['loss'] for q in quality]}
    resources = set().union(*(side.get(field, {}) for side in sides
        for field in ('costs', 'cost_intervals')))

    def interval(side, key):
        if key in side.get('cost_intervals', {}):
            bounds = side['cost_intervals'][key]
            return bounds if (isinstance(bounds, (list, tuple)) and len(bounds) == 2
                and all(numeric(value) for value in bounds) and bounds[0] <= bounds[1]) else None
        value = side.get('costs', {}).get(key)
        return [value, value] if numeric(value) else None

    bounds = [(interval(sides[0], key), interval(sides[1], key)) for key in sorted(resources)]
    if not {'observation_cost', 'compute_cost'} <= resources or any(a is None or b is None for a, b in bounds):
        return {**result, 'reason': 'complete_resource_vector_required'}
    deltas = [[b[0] - a[1], b[1] - a[0]] for a, b in bounds]
    preference = ('challenger' if all(high <= 0 for low, high in deltas)
            and any(high < 0 for low, high in deltas)
        else 'baseline' if all(low >= 0 for low, high in deltas) and any(low > 0 for low, high in deltas)
        else 'tie' if all(delta == [0, 0] for delta in deltas)
        else 'incomparable' if any(low > 0 for low, high in deltas) and any(high < 0 for low, high in deltas)
        else 'UNKNOWN')
    return {**result, 'preference': preference, 'reason': 'resource_tradeoff_without_weights'
        if preference == 'incomparable' else 'resource_order_not_identified' if preference == 'UNKNOWN'
        else 'equal_quality_resource_order',
        'losses': [q['loss'] for q in quality]}


def research_paired_method_conditions(method_ref, pairs, budget, *, max_conditions=8,
                                      target_profile=None):
    """Learn only from matched action contrasts; retain untested/negative work."""
    if not 1 <= max_conditions <= 8:
        raise ValueError('paired method research permits one to eight conditions')
    unique = []
    for pair in pairs:
        origin = pair.get('origin_ref', pair.get('feedback_ref'))
        previous = next((i for i, value in enumerate(unique)
            if value.get('origin_ref', value.get('feedback_ref')) == origin), None)
        if previous is None:
            unique.append(pair)
        else:
            unique[previous] = pair
    preferences, measured, validation, unresolved = [], [], [], []
    for pair in unique:
        preference = paired_method_preference(pair)
        preferences.append(preference)
        if preference['preference'] not in {'baseline', 'challenger'}:
            unresolved.append(deepcopy(pair.get('feedback_ref')))
            continue
        side = pair['baseline']
        item = {'feedback_ref': deepcopy(pair['feedback_ref']),
            'status': 'local_improvement' if preference['preference'] == 'challenger' else 'complete_unhelpful',
            'return_basis': {'kind': 'matched_action_pair', 'pair_ref': deepcopy(pair['feedback_ref']),
                             'identified_effect': preference['identified_effect']},
            'before_loss': preference['losses'][0], 'after_loss': preference['losses'][1],
            'process_return': preference, 'profile': deepcopy(side['profile']),
            'physical_source_ref': deepcopy(pair.get('physical_source_ref'))}
        (validation if pair.get('data_role') == 'condition_validation' else measured).append(item)
    local = [pair['feedback_ref'] for pair, preference in zip(unique, preferences)
        if pair.get('data_role') == 'development' and preference['preference'] == 'challenger'
        and pair.get('baseline_action_content') is not None
        and pair.get('challenger_action_content') is not None
        and pair['baseline_action_content'] != pair['challenger_action_content']]
    result = {'status': 'insufficient_paired_preferences', 'preferences': preferences,
        'condition': None, 'candidates': [], 'coordinate_paths': [], 'new_version_supported': False,
        'local_program_supported': bool(local), 'local_support_refs': deepcopy(local),
        'general_guard_supported': False,
        'compute_cost': 0, 'world_support_updated': False, 'scope': 'exposed_development',
        'unresolved_feedback_refs': unresolved, 'paired_invocations': len(unique),
        'identified_effects': sorted({preference['identified_effect'] for preference in preferences
            if preference['preference'] in {'baseline', 'challenger'}}),
        'evaluation_use': 'exposed_development_labels_only', 'independent_confirmation': False}
    if (len(measured) < 3 or {item['status'] for item in measured} != {
            'local_improvement', 'complete_unhelpful'}):
        return result
    # Static grouping is performed by the producer; without actual ordinary
    # action witnesses no guard is published, even when losses happen to differ.
    if not any(pair.get('baseline_action_content') is not None and
            pair.get('challenger_action_content') is not None and
            pair['baseline_action_content'] != pair['challenger_action_content']
            for pair in unique if pair.get('data_role') == 'development'):
        return {**result, 'status': 'no_distinct_executable_action'}
    learned = research_method_conflict(method_ref, [], budget,
        paired_returns=(measured, unresolved), max_conditions=max_conditions, target_profile=target_profile)
    result.update(learned, preferences=preferences, paired_invocations=len(unique),
        new_version_supported=False, local_program_supported=bool(local),
        local_support_refs=deepcopy(local), general_guard_supported=False)
    if learned.get('status') != 'development_conditions_found':
        return result
    groups = [item['physical_source_ref'] for item in measured]
    checked = [item for item in validation if item['physical_source_ref'] is not None
        and item['physical_source_ref'] not in groups]
    accepted = [candidate for candidate in learned['candidates'] if checked and all(
        (prediction := _conflict_predictions({**learned, 'candidates': [candidate]}, item['profile']))
        and prediction[0]['helpful'] == (item['status'] == 'local_improvement') for item in checked)]
    result['group_validation_refs'] = [item['feedback_ref'] for item in checked]
    if not accepted:
        result.update(status='development_condition_without_group_support', condition=None)
        return result
    result.update(candidates=accepted, condition=deepcopy(accepted[0]['condition']),
        new_version_supported=True, general_guard_supported=True,
        status='paired_condition_supported_on_declared_groups')
    return result
