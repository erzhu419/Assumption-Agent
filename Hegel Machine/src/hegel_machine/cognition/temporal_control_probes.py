"""Solve measured-history and recurrent predicates over actual command prefixes."""

from dataclasses import replace
from random import Random

from .language import MissingHistory, Program
from .models import Probe
from .scale_bindings import control_history_depth


class _Prefix:
    def __init__(self, bounds, depth, spend):
        self.depth, self.spend = depth, spend
        self.coordinates = {(name, step): f"prefix_control_{index}"
            for index, (name, step) in enumerate((name, step)
                for step in range(depth + 1) for name in bounds)}
        self.bounds = {alias: bounds[name] for (name, _), alias in self.coordinates.items()}

    def program(self, program, step=None):
        step = self.depth if step is None else step
        if not self.spend(1):
            raise _BudgetEnded
        if program.op == "call":
            return self.program(program.expanded(), step)
        if program.op == "feature":
            return Program("feature", name=self.coordinates[program.name, step])
        if program.op == "lag":
            if step < program.lag:
                raise MissingHistory("prefix is shorter than the requested instrument history")
            return Program("feature", name=self.coordinates[program.name, step - program.lag])
        if program.op == "available":
            return Program("const", value=float(step >= program.lag))
        if program.op == "if":
            condition = self.program(program.children[0], step).simplified()
            if condition.op == "const":
                return self.program(program.children[1 if condition.value else 2], step)
            return replace(program, children=(condition, self.program(program.children[1], step),
                                               self.program(program.children[2], step)))
        if program.op == "event_delta":
            if step < program.lag:
                return Program("const", value=0.)
            return Program("sub", children=(self.program(program.children[0], step),
                self.program(program.children[0], step - program.lag)))
        if program.op == "window_sum":
            parts = [self.program(program.children[0], prior)
                     for prior in range(max(0, step - program.lag + 1), step + 1)]
            result = parts[0]
            for part in parts[1:]:
                result = Program("add", children=(result, part))
            return result
        if program.op == "accumulate":
            state = self.program(program.children[1], step)
            for prior in range(step + 1):
                if not self.spend(2):
                    raise _BudgetEnded
                state = Program("add", children=(Program("mul", children=(
                    Program("const", value=program.value), state)),
                    self.program(program.children[0], prior)))
            return state
        return replace(program, children=tuple(self.program(child, step) for child in program.children))

    def sequence(self, solution):
        return tuple({name: solution[alias] for (name, step), alias in self.coordinates.items()
                      if step == index} for index in range(self.depth + 1))


class _BudgetEnded(Exception):
    pass


def propose_temporal_control_probes(command_spec, executable, spend, *, serial=0, length=4,
                                   programs=None, binding=None, control_levels=None):
    """Reuse the existing numerical solvers on time-indexed real controls.

    Only the solve sees flattened prefix coordinates. Every returned proposal
    forecasts the original programs on a reset trajectory with recorded history.
    ``spend`` belongs to the caller's existing 6000-unit proposal allowance.
    """
    from .program_disagreement_probes import (_forecast, _legal_sequence, _public_control_levels,
        predicate_boundary_groups, solve_joint_boundary)
    from .program_condition_sets import compound_predicate_regions, solve_compound_region

    def uses_state_replay(program):
        return program.op == 'state_replay' or any(uses_state_replay(child)
                                                  for child in program.children)

    if any(uses_state_replay(program.expanded()) for program in executable):
        # Endogenous predictions are not public prefix-control coordinates.
        # The numerical forecast path supports replay; this symbolic flattener
        # does not yet unroll a learned transition's internal state.
        return []

    depth = max(control_history_depth(program) for program in executable)
    def recurrent(program):
        return program.op == "accumulate" or any(recurrent(child) for child in program.children)
    if any(recurrent(program.expanded()) for program in executable):
        # A recurrent model's history_depth marks dependence, not a memory
        # cutoff. Solve over the requested real experiment, including buildup.
        depth = max(depth, min(length, command_spec.max_probe_length) - 1)
    if depth + 1 > command_spec.max_probe_length:
        return []
    prefix = _Prefix(command_spec.feature_bounds, depth, spend)
    levels = _public_control_levels(command_spec.feature_bounds, control_levels)
    try:
        symbolic = [prefix.program(program) for program in executable]
    except (MissingHistory, KeyError, _BudgetEnded):
        return []
    groups = predicate_boundary_groups(symbolic, spend=spend)
    regions = compound_predicate_regions(symbolic, spend=spend) if len(groups) > 1 else []
    from .temporal_experiment_methods import temporal_descriptors, temporal_recipe
    descriptions = temporal_descriptors(prefix, programs, binding or {}, spend) if programs is not None else {}
    rng = Random(524287 + 1009 * serial)
    contexts = [{name: rng.uniform(*interval) for name, interval in prefix.bounds.items()} for _ in range(3)]
    proposals, seen = [], set()

    def append(solution, details, selected, context):
        if solution is None:
            return True
        commands = prefix.sequence(solution)
        count = min(command_spec.max_probe_length, max(length, len(commands)))
        sequences = [("prefix", (*commands, *(dict(commands[-1]) for _ in range(count - len(commands)))))]
        if len(commands) + 2 * (depth + 1) <= command_spec.max_probe_length:
            sequences.append(("prefix_cycle", (*commands,
                *(dict(commands[-1]) for _ in range(depth + 1)),
                *(dict(commands[0]) for _ in range(depth + 1)))))
        for mode, sequence in sequences:
            sequence = _legal_sequence(sequence, levels, spend)
            if sequence is None:
                return False
            key = tuple(tuple(sorted(point.items())) for point in sequence)
            if key in seen:
                continue
            seen.add(key)
            predictions = [_forecast(program, sequence, spend) for program in executable]
            if any(prediction is None for prediction in predictions):
                return False
            common = [row for row in zip(*predictions) if all(value is not None for value in row)]
            separation = sum((max(row) - min(row)) ** 2 for row in common) / len(common) if common else 0.
            if separation <= 0.:
                continue
            recipe = temporal_recipe(prefix, selected, descriptions, context, details)
            proposals.append({"probe": Probe(f"temporal_predicate:{serial}:{len(proposals)}", sequence),
                "reason": "temporal_program_predicate", "predicted_separation": separation,
                "predictions": predictions, "sequence_mode": mode,
                "boundary": {"solver_strategy": "temporal_prefix", "history_depth": depth,
                    "prefix_coordinates": [{"control": name, "step": step, "value": solution[alias]}
                                           for (name, step), alias in prefix.coordinates.items()],
                    **({"public_control_levels": {name: list(values) for name, values in levels.items()},
                        "executed_prefix": list(sequence[:len(commands)])} if levels is not None else {}),
                    "solve": details, **({"temporal_recipe": recipe} if recipe is not None else {})}})
        return len(proposals) < 8

    # Narrow conditions on different delayed sensors must be satisfied in the
    # same physical prefix; solving their individual roots is insufficient.
    for region in regions:
        for context in contexts:
            solved, details = solve_compound_region(region, context, prefix.bounds, spend)
            if not append(solved, details, region, context):
                return sorted(proposals, key=lambda item: -item["predicted_separation"])
        if proposals:
            return sorted(proposals, key=lambda item: -item["predicted_separation"])
    for group in groups:
        thresholds = group["thresholds"]
        targets = [(left + right) / 2 for left, right in zip(thresholds, thresholds[1:])]
        if not targets:
            threshold = thresholds[0]
            delta = max(1., abs(threshold)) * 1e-5
            targets = [threshold - delta, threshold, threshold + delta]
        axes = sorted({name for name, _ in group["expression"].observed_variables()})
        for target in targets[:6]:
            for context in contexts:
                for control in axes:
                    solved, details = solve_joint_boundary(group["expression"], target, context,
                        prefix.bounds, spend, strategy="coordinate", control=control)
                    if not append(solved, {**details, "target": target, "control": control}, group, context):
                        return sorted(proposals, key=lambda item: -item["predicted_separation"])
                if not proposals and len(axes) > 1:
                    solved, details = solve_joint_boundary(group["expression"], target, context,
                        prefix.bounds, spend, strategy="joint_segment")
                    if not append(solved, {**details, "target": target}, group, context):
                        return sorted(proposals, key=lambda item: -item["predicted_separation"])
            if proposals:
                return sorted(proposals, key=lambda item: -item["predicted_separation"])
    return sorted(proposals, key=lambda item: -item["predicted_separation"])
