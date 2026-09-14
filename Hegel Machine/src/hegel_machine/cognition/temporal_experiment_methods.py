"""Recall a measured prefix-solving operation under current instrument laws."""

from dataclasses import replace
from copy import deepcopy

from .language import MissingHistory, Program
from .models import Probe
from .scale_bindings import bind_measurement_arguments, control_history_depth


def _names(program):
    result = []

    def visit(node):
        if node.op in {"feature", "lag"} and node.name not in result:
            result.append(node.name)
        for child in node.children:
            visit(child)

    visit(program)
    return result


def temporal_descriptors(prefix, programs, binding, spend):
    """Relate solved time coordinates to the original measured predicates."""
    from .program_disagreement_probes import predicate_boundary_groups
    from .temporal_control_probes import _BudgetEnded

    descriptions = {}
    for group in predicate_boundary_groups(programs, spend=spend):
        measured = group["measured_expression"]
        executable = bind_measurement_arguments(measured, binding)
        for offset in range(prefix.depth + 1):
            try:
                symbolic = prefix.program(executable, prefix.depth - offset)
            except (MissingHistory, KeyError):
                continue
            except _BudgetEnded:
                return descriptions
            descriptions.setdefault(symbolic, {"measured_expression": measured, "offset": offset})
    return descriptions


def temporal_recipe(prefix, selected, descriptions, context, details):
    """Save role/time preparation and the successful solve, not its answer."""
    from .program_experiment_methods import _interval_shape, _predicate_patterns

    constraints = selected.get("constraints")
    groups = constraints or [selected]
    descriptors = [descriptions.get(group["expression"]) for group in groups]
    if not all(descriptors):
        return None
    inverse = {alias: (name, step) for (name, step), alias in prefix.coordinates.items()}
    aliases = list(dict.fromkeys(alias for group in groups for alias in _names(group["expression"])))
    roles = list(dict.fromkeys(inverse[alias][0] for alias in aliases))
    dependencies = [[{"control_role": roles.index(inverse[alias][0]),
                      "lag": prefix.depth - inverse[alias][1]}
                     for alias in _names(group["expression"])] for group in groups]
    context_by_role = []
    for name in roles:
        fractions = []
        for step in range(prefix.depth + 1):
            alias = prefix.coordinates[name, step]
            low, high = prefix.bounds[alias]
            fractions.append((context[alias] - low) / (high - low) if high > low else .5)
        context_by_role.append(fractions)
    recipe = {"predicate_patterns": _predicate_patterns([item["measured_expression"] for item in descriptors]),
        "predicate_offsets": [item["offset"] for item in descriptors],
        "solver_strategy": details["solver_strategy"], "source_control_roles": roles,
        "source_history_depth": prefix.depth, "control_time_dependencies": dependencies,
        "context_by_role": context_by_role, "coordinate_dependency": None}
    if constraints:
        recipe["condition_intervals"] = [_interval_shape(item) for item in constraints]
    else:
        thresholds = selected["thresholds"]
        recipe["target_strategy"] = ("between_adjacent" if len(thresholds) > 1 else
            "below" if details["target"] < thresholds[0] else
            "above" if details["target"] > thresholds[0] else "at")
        if details["solver_strategy"] == "coordinate":
            recipe["coordinate_dependency"] = _names(selected["expression"]).index(details["control"])
    return recipe


def _sequence(prefix, solved, length, mode, maximum):
    commands = prefix.sequence(solved)
    if mode == "prefix_cycle":
        if len(commands) + 2 * (prefix.depth + 1) > maximum:
            return None
        return (*commands, *(dict(commands[-1]) for _ in range(prefix.depth + 1)),
                *(dict(commands[0]) for _ in range(prefix.depth + 1)))
    if mode != "prefix":
        return None
    count = min(maximum, max(length, len(commands)))
    return (*commands, *(dict(commands[-1]) for _ in range(count - len(commands))))


def _measured_time_regions(programs, depth, spend):
    """Keep the same measured field at different times as different axes."""
    from .program_condition_sets import compound_predicate_regions
    from .temporal_control_probes import _BudgetEnded, _Prefix

    names = list(dict.fromkeys(name for program in programs for name in _names(program.expanded())))
    measured_prefix = _Prefix(dict.fromkeys(names, (0., 1.)), depth, spend)
    try:
        symbolic = [measured_prefix.program(program) for program in programs]
    except (MissingHistory, KeyError, _BudgetEnded):
        return []
    inverse = {alias: (name, step) for (name, step), alias in measured_prefix.coordinates.items()}
    regions = []
    for region in compound_predicate_regions(symbolic, spend=spend):
        constraints = []
        for item in region["constraints"]:
            expression = item["measured_expression"]
            final_step = max(inverse[name][1] for name in _names(expression))

            def restore(node):
                if node.op == "feature":
                    name, step = inverse[node.name]
                    lag = final_step - step
                    return Program("lag", name=name, lag=lag) if lag else Program("feature", name=name)
                return replace(node, children=tuple(restore(child) for child in node.children))

            constraints.append({**item, "measured_expression": restore(expression),
                                "predicate_offset": depth - final_step})
        regions.append({**region, "constraints": constraints})
    return regions


def propose_temporal_method(command_spec, programs, executable, binding, method, spend,
                            *, serial=0, length=4, tolerance=1e-7, failed_preparations=None):
    """Execute only this saved recipe; no context, solver or mode search."""
    from .program_disagreement_probes import _forecast, predicate_boundary_groups, solve_joint_boundary
    from .program_condition_sets import solve_compound_region
    from .program_experiment_methods import _interval_shape, _predicate_patterns
    from .temporal_control_probes import _BudgetEnded, _Prefix

    recipe = method["temporal_recipe"]
    depth = max(control_history_depth(program) for program in executable)
    if depth + 1 > command_spec.max_probe_length:
        return []
    prefix = _Prefix(command_spec.feature_bounds, depth, spend)
    compound = recipe["solver_strategy"] == "compound_conditions"
    choices = (_measured_time_regions(programs, depth, spend) if compound else
               predicate_boundary_groups(programs, spend=spend))
    for choice in choices:
        measured_groups = choice["constraints"] if compound else [choice]
        if _predicate_patterns([item["measured_expression"] for item in measured_groups]) != recipe["predicate_patterns"]:
            continue
        if compound and [_interval_shape(item) for item in measured_groups] != recipe["condition_intervals"]:
            continue
        if compound and [item["predicate_offset"] for item in measured_groups] != recipe["predicate_offsets"]:
            continue
        symbolic = []
        try:
            for group, offset in zip(measured_groups, recipe["predicate_offsets"]):
                if offset > depth:
                    return []
                expression = prefix.program(bind_measurement_arguments(group["measured_expression"], binding), depth - offset)
                symbolic.append({**group, "expression": expression})
        except (MissingHistory, KeyError, _BudgetEnded):
            return []
        inverse = {alias: (name, step) for (name, step), alias in prefix.coordinates.items()}
        roles = list(dict.fromkeys(inverse[alias][0] for group in symbolic for alias in _names(group["expression"])))
        if len(roles) != len(recipe["context_by_role"]):
            continue
        context = {alias: (low + high) / 2 for alias, (low, high) in prefix.bounds.items()}
        for name, fractions in zip(roles, recipe["context_by_role"]):
            for step in range(depth + 1):
                # Preserve the supported recent preparation; extend a longer
                # delay with its earliest prepared value before solving it.
                source_step = max(0, len(fractions) - 1 - (depth - step))
                alias = prefix.coordinates[name, step]
                low, high = prefix.bounds[alias]
                context[alias] = low + (high - low) * fractions[source_step]
        if compound:
            solved, details = solve_compound_region({**choice, "constraints": symbolic}, context, prefix.bounds, spend)
            attempts = [(solved, details)]
        else:
            group = symbolic[0]
            thresholds = group["thresholds"]
            strategy = recipe["target_strategy"]
            if strategy == "between_adjacent":
                targets = [(left + right) / 2 for left, right in zip(thresholds, thresholds[1:])]
            else:
                direction = {"below": -1, "at": 0, "above": 1}[strategy]
                targets = [value + direction * max(1., abs(value)) * 1e-5 for value in thresholds]
            dependencies = _names(group["expression"])
            coordinate = recipe["coordinate_dependency"]
            if coordinate is not None and coordinate >= len(dependencies):
                continue
            control = dependencies[coordinate] if coordinate is not None else None
            def solve_targets():
                for target in targets[:6]:
                    solved, details = solve_joint_boundary(group["expression"], target, context, prefix.bounds,
                        spend, strategy=recipe["solver_strategy"], control=control)
                    yield solved, {**details, "target": target}

            attempts = solve_targets()
        for solved, details in attempts:
            def retain_preparation_failure():
                if failed_preparations is not None and recipe["solver_strategy"] == "coordinate":
                    failed_preparations.append({"method": method, "prefix": prefix,
                        "group": symbolic[0], "context": context, "control": control,
                        "target": details["target"], "roles": roles, "executable": executable})

            if solved is None:
                retain_preparation_failure()
                continue
            sequence = _sequence(prefix, solved, length, method["sequence_mode"], command_spec.max_probe_length)
            if sequence is None:
                continue
            predictions = [_forecast(program, sequence, spend) for program in executable]
            if any(prediction is None for prediction in predictions):
                return []
            common = [row for row in zip(*predictions) if all(value is not None for value in row)]
            separation = sum((max(row) - min(row)) ** 2 for row in common) / len(common) if common else 0.
            if separation <= tolerance:
                retain_preparation_failure()
                continue
            dependencies = [[{"control_role": roles.index(inverse[alias][0]), "lag": depth - inverse[alias][1]}
                             for alias in _names(group["expression"])] for group in symbolic]
            return [{"probe": Probe(f"temporal_method:{method['method_id']}:{serial}", sequence),
                "reason": "learned_temporal_experiment_method", "method_id": method["method_id"],
                "source_task_id": method["source_task_id"], "source_probe": method["source_probe"],
                "sequence_mode": method["sequence_mode"], "predictions": predictions,
                "predicted_separation": separation,
                "boundary": {"solver_strategy": "temporal_prefix", "history_depth": depth,
                    "control_time_dependencies": dependencies, "solve": details,
                    "prefix_coordinates": [{"control": name, "step": step, "value": solved[alias]}
                                           for (name, step), alias in prefix.coordinates.items()],
                    "temporal_recipe": recipe}}]
    return []


def revise_temporal_preparation(attempt, spend, *, serial=0, length=4, maximum=12, tolerance=1e-7):
    """Modify failed preparation dependencies in the already compiled prefix."""
    from .program_disagreement_probes import _forecast, solve_joint_boundary

    method, prefix = attempt["method"], attempt["prefix"]
    context, group, roles = attempt["context"], attempt["group"], attempt["roles"]
    inverse = {alias: (name, step) for (name, step), alias in prefix.coordinates.items()}
    preparation = [alias for alias in _names(group["expression"])
                   if alias != attempt["control"] and inverse[alias][1] < prefix.depth]
    if not preparation:
        return []
    adjustable = {*preparation, attempt["control"]}
    bounds = {alias: interval if alias in adjustable else (context[alias], context[alias])
              for alias, interval in prefix.bounds.items()}
    solved, details = solve_joint_boundary(group["expression"], attempt["target"], context,
        bounds, spend, strategy="joint_segment")
    if solved is None:
        return []
    changed = [alias for alias in preparation if solved[alias] != context[alias]]
    if not changed:
        return []
    # The joint solve proposes preparation. Execute the actual retained
    # coordinate operation on it before forecasting or buying the experiment.
    # Nonlinear coordinates can otherwise select a different root on replay.
    prepared = {**context, **{alias: solved[alias] for alias in changed}}
    solved, details = solve_joint_boundary(group["expression"], attempt["target"], prepared,
        prefix.bounds, spend, strategy="coordinate", control=attempt["control"])
    if solved is None:
        return []
    sequence = _sequence(prefix, solved, length, method["sequence_mode"], maximum)
    if sequence is None:
        return []
    predictions = [_forecast(program, sequence, spend) for program in attempt["executable"]]
    if any(prediction is None for prediction in predictions):
        return []
    common = [row for row in zip(*predictions) if all(value is not None for value in row)]
    separation = sum((max(row) - min(row)) ** 2 for row in common) / len(common) if common else 0.
    if separation <= tolerance:
        return []
    # Keep the parent's coordinate-solving operation. Learn a changed history
    # preparation only if the caller subsequently measures its actual value.
    recipe = deepcopy(method["temporal_recipe"])
    fractions = []
    for name in roles:
        values = []
        for step in range(prefix.depth + 1):
            alias = prefix.coordinates[name, step]
            low, high = prefix.bounds[alias]
            values.append((prepared[alias] - low) / (high - low) if high > low else .5)
        fractions.append(values)
    dependencies = [{"control_role": roles.index(inverse[alias][0]), "lag": prefix.depth - inverse[alias][1]}
                    for alias in _names(group["expression"])]
    recipe.update(source_control_roles=roles, source_history_depth=prefix.depth,
                  context_by_role=fractions, control_time_dependencies=[dependencies])
    revision = {"operation": "joint_segment", "target": attempt["target"],
        "changed_steps": [{"control_role": roles.index(inverse[alias][0]),
            "lag": prefix.depth - inverse[alias][1], "previous": context[alias], "value": solved[alias]}
            for alias in changed]}
    return [{"probe": Probe(f"temporal_preparation:{method['method_id']}:{serial}", sequence),
        "reason": "revised_temporal_preparation", "parent_method_id": method["method_id"],
        "source_task_id": method["source_task_id"], "source_probe": method["source_probe"],
        "sequence_mode": method["sequence_mode"], "predictions": predictions,
        "predicted_separation": separation, "preparation_revision": revision,
        "boundary": {"solver_strategy": "temporal_prefix", "history_depth": prefix.depth,
            "control_time_dependencies": [dependencies], "solve": {**details, "target": attempt["target"]},
            "prefix_coordinates": [{"control": name, "step": step, "value": solved[alias]}
                                   for (name, step), alias in prefix.coordinates.items()],
            "temporal_recipe": recipe}}]
