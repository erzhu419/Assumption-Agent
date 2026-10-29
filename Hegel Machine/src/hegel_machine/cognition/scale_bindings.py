"""Bind an aggregate measurement role to paid measurements at another scale."""

from dataclasses import replace

from .language import Program
from .structure_lab import ObjectSchema


def control_history_depth(program):
    """Prefix length needed to expose finite controls and their reset guards."""
    expanded = program.expanded()
    return max(expanded.history_depth, expanded.lag if expanded.op == "available" else 0,
               *(control_history_depth(child) for child in expanded.children))


def aggregate_measurement_binding(source_schema, current_schema):
    """Express whole-apparatus source fields using the current object view.

    Quantity, impulse and focus membership are additive. Group position is an
    average over its members, so different group sizes require weighted means.
    The argument still denotes the whole apparatus if its object count changes.
    """
    if source_schema is None or current_schema is None:
        return None
    source = source_schema.to_dict() if isinstance(source_schema, ObjectSchema) else source_schema
    current = current_schema.to_dict() if isinstance(current_schema, ObjectSchema) else current_schema
    if len(source["objects"]) != 1:
        return None
    source_group = source["objects"][0]
    if set(source["members"][source_group]) != set(source["controls"]):
        return None
    members = [member for group in current["objects"] for member in current["members"][group]]
    if not members or len(members) != len(set(members)) or set(members) != set(current["controls"]):
        return None
    binding = {}
    for role, name in source["fields"][source_group].items():
        fields = tuple(Program("feature", name=current["fields"][group][role])
                       for group in current["objects"])
        if role == "x":
            fields = tuple(Program("mul", children=(
                Program("const", value=len(current["members"][group]) / len(members)), field))
                for group, field in zip(current["objects"], fields))
        binding[name] = Program("object_sum", children=fields)
    return binding


def aggregate_role_binding(template, source_schema, current_schema):
    """Bind learned template roles to expressions for their aggregate fields."""
    if set(template.object_roles) != set(template.roles) or any(
            index != 0 or role not in {"q", "x", "u", "f"}
            for index, role in template.object_roles.values()):
        return None
    measurements = aggregate_measurement_binding(source_schema, current_schema)
    if measurements is None:
        return None
    source = source_schema.to_dict() if isinstance(source_schema, ObjectSchema) else source_schema
    fields = source["fields"][source["objects"][0]]
    return {name: measurements[fields[role]] for name, (_, role) in template.object_roles.items()}


def _shift_time(program, lag):
    if not lag:
        return program
    if program.op == "feature":
        return Program("lag", name=program.name, lag=lag)
    if program.op in {"lag", "available"}:
        return replace(program, lag=program.lag + lag)
    if program.op == "call":
        return _shift_time(program.expanded(), lag)
    if program.op in {"accumulate", "state_replay"}:
        # Shifting the transition's inputs changes its reset and update rule.
        # Evaluate the whole state on its actual earlier prefix instead.
        return Program("sub", children=(program,
            Program("event_delta", lag=lag, children=(program,))))
    if program.op == "event_delta":
        return Program("if", children=(Program("available", lag=program.lag + lag),
            Program("sub", children=(_shift_time(program.children[0], lag),
                _shift_time(program.children[0], program.lag + lag))), Program("const", value=0.)))
    if program.op == "window_sum":
        # A window at an earlier time has that earlier reset boundary too.
        result = _shift_time(program.children[0], lag)
        for offset in range(1, program.lag):
            term = Program("if", children=(Program("available", lag=lag + offset),
                _shift_time(program.children[0], lag + offset), Program("const", value=0.)))
            result = Program("add", children=(result, term))
        return result
    if not program.children:
        return program
    return replace(program, children=tuple(_shift_time(child, lag) for child in program.children))


def _at_lag(program, lag):
    shifted = _shift_time(program, lag)
    if shifted.minimum_history_depth < lag:
        fields = sorted({name for name, _ in program.observed_variables()})
        if fields:
            # The mapped instrument can have a guarded initial value, while
            # requesting its value before the experiment still lacks history.
            missing = Program("lag", name=fields[0], lag=lag)
            return Program("if", children=(Program("available", lag=lag), shifted, missing))
    return shifted


def bind_measurement_arguments(argument, mapping):
    """Substitute observed arguments without changing a learned function body."""
    if argument.op in {"feature", "lag"} and argument.name in mapping:
        replacement = mapping[argument.name]
        if isinstance(replacement, str):
            return replace(argument, name=replacement)
        return _at_lag(replacement, argument.lag) if argument.op == "lag" else replacement
    if not argument.children:
        return argument
    return replace(argument, children=tuple(bind_measurement_arguments(child, mapping)
                                           for child in argument.children))
