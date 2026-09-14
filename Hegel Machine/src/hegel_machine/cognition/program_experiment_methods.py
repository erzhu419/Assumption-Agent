"""Reuse measured experimental procedures against current executable laws.

A method keeps a predicate role pattern and a successful way of exciting and
solving it. Its thresholds, calibration and final predictions come from the
current task. Only the caller's purchased experiment can support the method.
"""

from copy import deepcopy
from dataclasses import replace

from .language import Program
from .models import Probe
from .scale_bindings import bind_measurement_arguments, control_history_depth


def _predicate_patterns(programs):
    names = {}

    def visit(node):
        name = node.name
        if node.op in {"feature", "lag"}:
            name = names.setdefault(name, f"role_{len(names)}")
        return replace(node, name=name, children=tuple(visit(child) for child in node.children))

    return [visit(program).to_dict() for program in programs]


def _predicate_pattern(program):
    return _predicate_patterns([program])[0]


def _interval_shape(constraint):
    thresholds = constraint["thresholds"]
    return {side: thresholds.index(constraint[side]) if constraint[side] is not None else None
            for side in ("lower", "upper")}


def _control_roles(programs, binding, bounds):
    """Order public controls by their occurrence in measured program roles."""
    result = []

    def visit(node):
        if node.op in {"feature", "lag"} and node.name in bounds and node.name not in result:
            result.append(node.name)
        for child in node.children:
            visit(child)

    for program in programs:
        visit(bind_measurement_arguments(program.expanded(), binding or {}))
    return result


def _programs(plan):
    return [Program.from_dict(item["program"]) for item in plan.get("hypotheses", [])]


def _current_condition(programs, binding, bounds, task_id):
    required = {name for program in programs for name, _ in program.observed_variables()} - set(bounds)
    return {"task_id": task_id, "programs": [program.to_dict() for program in programs],
            "binding": {name: program.to_dict() for name, program in binding.items() if name in required},
            "bounds": {name: list(interval) for name, interval in bounds.items()}}


def _same_failed_field(field, binding, bounds):
    return (field["name"] in binding and binding[field["name"]].to_dict() == field["binding"] and
            all(name in bounds and list(bounds[name]) == interval for name, interval in field["bounds"].items()))


def _failure_scope(plan, assessment, binding, command_spec, task_id, tolerance):
    if assessment.get("calibration_status") != "counterexample":
        return {"reason": "unresolved_under_current_condition", "condition":
            _current_condition(_programs(plan), binding, command_spec.feature_bounds, task_id)}
    fields = []
    for name, loss in assessment.get("field_losses", {}).items():
        if loss <= tolerance or name not in binding:
            continue
        expression = binding[name]
        controls = {field for field, _ in expression.observed_variables()}
        fields.append({"name": name, "binding": expression.to_dict(),
            "bounds": {control: list(command_spec.feature_bounds[control]) for control in controls}})
    return {"reason": "calibration_counterexample", "task_id": task_id, "fields": fields}


class ProgramExperimentMethods:
    def __init__(self, records=None):
        self.records = deepcopy(records or [])
        self.last_proposal = {}

    def to_dict(self):
        return deepcopy(self.records)

    def propose(self, command_spec, programs, budget, *, binding=None, serial=0,
                length=4, tolerance=1e-7, task_id=None, contexts=None):
        """Execute a matching learned procedure before searching for a new one."""
        from .program_disagreement_probes import (
            _experiment_sequence, _forecast, predicate_boundary_groups, solve_joint_boundary,
        )
        programs = list(programs)
        self.last_proposal = {"status": "no_match", "blocked_scopes": []}
        supported = [method for method in self.records if method["status"] == "supported"]
        if len(programs) < 2 or not supported or not command_spec.feature_bounds:
            return []
        start = budget.spent
        allowance = min(4000, budget.remaining)

        def spend(units):
            return budget.spent - start + units <= allowance and budget.spend(units)

        bounds = command_spec.feature_bounds
        required = {name for program in programs for name, _ in program.observed_variables()
                    if name != "__target__"} - set(bounds)
        binding = {name: program for name, program in (binding or {}).items()
                   if {field for field, _ in program.observed_variables()} <= set(bounds)}
        if required - set(binding):
            return []
        for method in self.records:
            for scope in method.get("failure_scopes", []):
                if scope["reason"] != "calibration_counterexample":
                    continue
                failed = [field["name"] for field in scope["fields"] if field["name"] in required
                          and _same_failed_field(field, binding, bounds)]
                if failed:
                    self.last_proposal["blocked_scopes"].append({"method_id": method["method_id"],
                        "reason": "calibration_counterexample", "fields": failed})
        if self.last_proposal["blocked_scopes"]:
            self.last_proposal["status"] = "blocked_calibration"
            return []
        executable = [bind_measurement_arguments(program.expanded(), binding) for program in programs]
        dynamic = any(control_history_depth(program) for name, program in binding.items() if name in required)
        length = min(command_spec.max_probe_length, max(2, length,
                     max(program.history_depth for program in executable) + 2))
        roles = _control_roles(programs, binding, bounds)
        groups = [] if dynamic else predicate_boundary_groups(programs, binding=binding, spend=spend)
        condition = _current_condition(programs, binding, bounds, task_id)
        regions = None

        def propose_solved(method, context, solved, boundary):
            if solved is None:
                return []
            sequence = _experiment_sequence(context, solved, length, method["sequence_mode"],
                history_depth=max(program.history_depth for program in executable),
                max_length=command_spec.max_probe_length)
            if sequence is None:
                return []
            predictions = [_forecast(program, sequence, spend) for program in executable]
            if any(prediction is None for prediction in predictions):
                return []
            common = [row for row in zip(*predictions) if all(value is not None for value in row)]
            separation = sum((max(row) - min(row)) ** 2 for row in common) / len(common) if common else 0.
            if separation <= tolerance:
                return []
            self.last_proposal["status"] = "proposed"
            return [{"probe": Probe(f"program_method:{method['method_id']}:{serial}", sequence),
                "reason": "learned_program_experiment_method", "method_id": method["method_id"],
                "source_task_id": method["source_task_id"], "source_probe": method["source_probe"],
                "sequence_mode": method["sequence_mode"], "predictions": predictions,
                "predicted_separation": separation, "compute_cost": budget.spent - start,
                "boundary": {**boundary, "context": context}}]

        failed_preparations = []
        # Recent real success selects the procedure; retrieval alone never
        # changes its support count or establishes current applicability.
        for method in reversed(supported):
            if any(scope.get("condition") == condition for scope in method.get("failure_scopes", [])):
                self.last_proposal["blocked_scopes"].append({"method_id": method["method_id"],
                    "reason": "unresolved_under_current_condition", "fields": []})
                self.last_proposal["status"] = "blocked_condition"
                continue
            if method["solver_strategy"] == "temporal_prefix":
                from .temporal_experiment_methods import propose_temporal_method
                proposed = propose_temporal_method(command_spec, programs, executable, binding,
                    method, spend, serial=serial, length=length, tolerance=tolerance,
                    failed_preparations=failed_preparations)
                if proposed:
                    self.last_proposal["status"] = "proposed"
                    proposed[0]["compute_cost"] = budget.spent - start
                    return proposed
                continue
            if dynamic:
                continue
            if len(roles) != len(method["context_fractions"]):
                continue
            context = {name: (low + high) / 2 for name, (low, high) in bounds.items()}
            for name, fraction in zip(roles, method["context_fractions"]):
                low, high = bounds[name]
                context[name] = low + (high - low) * fraction
            if contexts and method["method_id"] in contexts:
                background = contexts[method["method_id"]]
                if not spend(len(background)):
                    return []
                context.update(background)
            if method["solver_strategy"] == "compound_conditions":
                from .program_condition_sets import compound_predicate_regions, solve_compound_region
                if regions is None:
                    regions = compound_predicate_regions(programs, binding=binding, spend=spend)
                for region in regions:
                    constraints = region["constraints"]
                    patterns = _predicate_patterns([item["measured_expression"] for item in constraints])
                    if patterns != method["condition_patterns"] or [
                            _interval_shape(item) for item in constraints] != method["condition_intervals"]:
                        continue
                    solved, details = solve_compound_region(region, context, bounds, spend)
                    proposal = propose_solved(method, context, solved, {**details,
                        "solver_strategy": "compound_conditions", "constraints": [
                            {**item, "expression": item["expression"].to_dict(),
                             "measured_expression": item["measured_expression"].to_dict()} for item in constraints]})
                    if proposal:
                        return proposal
                continue
            for group in groups:
                if _predicate_pattern(group["measured_expression"]) != method["predicate_pattern"]:
                    continue
                thresholds = sorted(group["thresholds"])
                if method["target_strategy"] == "between_adjacent":
                    targets = [(left + right) / 2 for left, right in zip(thresholds, thresholds[1:])]
                else:
                    direction = {"below": -1, "at": 0, "above": 1}[method["target_strategy"]]
                    targets = [threshold + direction * max(1., abs(threshold)) * 1e-5
                               for threshold in thresholds]
                control = roles[method["coordinate_role"]] if method["coordinate_role"] is not None else None
                if control is not None and control not in {
                        name for name, _ in group["expression"].observed_variables()}:
                    continue
                for target in targets[:6]:
                    solved, details = solve_joint_boundary(group["expression"], target,
                        context, bounds, spend, strategy=method["solver_strategy"], control=control)
                    proposal = propose_solved(method, context, solved, {
                            "expression": group["expression"].to_dict(),
                            "measured_expression": group["measured_expression"].to_dict(),
                            "thresholds": thresholds, "target": target, **details})
                    if proposal:
                        return proposal
        # Every supported version gets its unchanged opportunity first. Only
        # then revise a failed preparation using its already compiled problem.
        from .temporal_experiment_methods import revise_temporal_preparation
        for attempt in failed_preparations:
            proposed = revise_temporal_preparation(attempt, spend, serial=serial, length=length,
                maximum=command_spec.max_probe_length, tolerance=tolerance)
            if not proposed:
                continue
            revision = proposed[0]
            failed_variant = next((method for method in self.records
                if method.get("parent_method_id") == revision["parent_method_id"]
                and method.get("temporal_recipe") == revision["boundary"]["temporal_recipe"]
                and any(scope.get("condition") == condition for scope in method.get("failure_scopes", []))), None)
            if failed_variant is not None:
                self.last_proposal["status"] = "blocked_condition"
                self.last_proposal["blocked_scopes"].append({"method_id": failed_variant["method_id"],
                    "parent_method_id": revision["parent_method_id"],
                    "reason": "unresolved_under_current_condition", "fields": []})
                continue
            self.last_proposal["status"] = "proposed"
            revision["compute_cost"] = budget.spent - start
            return proposed
        return []

    def observe(self, plan, assessment, *, task_id, command_spec, binding=None, tolerance=1e-7):
        """Learn or revise applicability only after the proposed probe was paid."""
        if not assessment.get("observation_rows"):
            return None
        proposal = plan.get("proposal", {})
        method_id = proposal.get("method_id")
        method = next((item for item in self.records if item["method_id"] == method_id), None)
        probe = plan.get("probe")
        probe_id = probe.probe_id if isinstance(probe, Probe) else (probe or {}).get("probe_id")
        evidence = {"task_id": task_id, "probe_id": probe_id,
            "status": assessment["status"], "calibration_status": assessment.get("calibration_status"),
            "observation_rows": assessment["observation_rows"],
            "hypotheses": deepcopy(assessment.get("hypotheses", [])),
            "field_losses": deepcopy(assessment.get("field_losses", {}))}
        success = (assessment["status"] in {"distinguished", "all_refuted"} and
                   assessment.get("calibration_status") != "counterexample")
        if method is not None:
            if success:
                method["support"].append(evidence)
                return {"action": "use_program_experiment_method", "method_id": method_id,
                        "source_task_id": method["source_task_id"], "task_id": task_id,
                        "cross_task": task_id != method["source_task_id"],
                        "outcome": assessment["status"]}
            measured = assessment.get("hypotheses", [])
            if assessment.get("calibration_status") == "counterexample" or (
                    measured and all(item.get("rows", 0) for item in measured)):
                scope = _failure_scope(plan, assessment, binding or {}, command_spec, task_id, tolerance)
                method.setdefault("failure_scopes", []).append(scope)
                method["counterexamples"].append(evidence)
                return {"action": "revise_program_experiment_method", "method_id": method_id,
                        "source_task_id": method["source_task_id"],
                        "task_id": task_id, "reason": scope["reason"], "status": "condition_restricted",
                        "scope": deepcopy(scope)}
            return None
        boundary = proposal.get("boundary", {})
        measured = assessment.get("hypotheses", [])
        measured_failure = (assessment.get("calibration_status") == "counterexample" or
                            (measured and all(item.get("rows", 0) for item in measured)))
        if (boundary.get("solver_strategy") == "temporal_prefix" and
                (success or (proposal.get("parent_method_id") and measured_failure))):
            recipe = boundary.get("temporal_recipe")
            if recipe is None:
                return None
            method_id = f"program_experiment_{len(self.records) + 1}"
            method = {"method_id": method_id, "status": "supported" if success else "counterexample",
                "source_task_id": task_id, "source_probe": probe_id,
                "solver_strategy": "temporal_prefix", "coordinate_role": None,
                "sequence_mode": proposal["sequence_mode"],
                "context_fractions": [values[-1] for values in recipe["context_by_role"]],
                "source_control_roles": list(recipe["source_control_roles"]),
                "temporal_recipe": deepcopy(recipe), "source_boundary": deepcopy(boundary),
                "steps": ["match_measured_predicate_roles", "rebind_current_time_dependencies",
                          "prepare_supported_history", "execute_supported_prefix_solver",
                          "forecast_current_programs", "purchase_and_compare"],
                "support": [evidence] if success else [], "counterexamples": [], "failure_scopes": []}
            if proposal.get("parent_method_id"):
                method.update(parent_method_id=proposal["parent_method_id"],
                              preparation_revision=deepcopy(proposal["preparation_revision"]))
                method["steps"].insert(2, "use_revised_preparation_history")
            if not success:
                scope = _failure_scope(plan, assessment, binding or {}, command_spec, task_id, tolerance)
                method["failure_scopes"].append(scope)
                method["counterexamples"].append(evidence)
            self.records.append(method)
            if not success:
                return {"action": "revise_program_experiment_method", "method_id": method_id,
                    "parent_method_id": proposal["parent_method_id"], "task_id": task_id,
                    "reason": scope["reason"], "status": "condition_restricted", "scope": deepcopy(scope)}
            return {"action": "learn_program_experiment_method", "method_id": method_id,
                    "task_id": task_id, "solver_strategy": "temporal_prefix",
                    **({"parent_method_id": proposal["parent_method_id"],
                        "preparation_revision": deepcopy(proposal["preparation_revision"])}
                       if proposal.get("parent_method_id") else {})}
        if not success or "context" not in boundary:
            return None
        compound = boundary.get("solver_strategy") == "compound_conditions"
        if compound and not boundary.get("constraints"):
            return None
        if not compound and ("measured_expression" not in boundary or not boundary.get("thresholds")):
            return None
        roles = _control_roles(_programs(plan), binding or {}, command_spec.feature_bounds)
        fractions = []
        for name in roles:
            low, high = command_spec.feature_bounds[name]
            fractions.append((boundary["context"][name] - low) / (high - low) if high > low else .5)
        strategy = boundary.get("solver_strategy", "coordinate")
        control = boundary.get("control")
        if strategy == "coordinate" and control not in roles:
            controls = boundary.get("controls", [])
            control = controls[0] if len(controls) == 1 else None
            if control not in roles:
                return None
        method_id = f"program_experiment_{len(self.records) + 1}"
        method = {"method_id": method_id, "status": "supported",
            "source_task_id": task_id, "source_probe": probe_id,
            "solver_strategy": strategy, "coordinate_role": roles.index(control) if strategy == "coordinate" else None,
            "sequence_mode": proposal.get("sequence_mode", "maintain"), "context_fractions": fractions,
            "steps": ["locate_current_predicate_conditions", "rebind_instrument_roles",
                      "excite_supported_context", strategy, "forecast_current_programs",
                      "purchase_and_compare"],
            "source_boundary": deepcopy(boundary), "source_control_roles": roles,
            "support": [evidence], "counterexamples": [], "failure_scopes": []}
        if compound:
            method["condition_patterns"] = _predicate_patterns([
                Program.from_dict(item["measured_expression"]) for item in boundary["constraints"]])
            method["condition_intervals"] = [_interval_shape(item) for item in boundary["constraints"]]
        else:
            threshold = boundary["thresholds"][0]
            method["predicate_pattern"] = _predicate_pattern(Program.from_dict(boundary["measured_expression"]))
            method["target_strategy"] = ("between_adjacent" if len(boundary["thresholds"]) > 1 else
                "below" if boundary["target"] < threshold else "above" if boundary["target"] > threshold else "at")
        self.records.append(method)
        return {"action": "learn_program_experiment_method", "method_id": method_id,
                "task_id": task_id, "solver_strategy": strategy}
