"""Separate learned measurement computations from their instrument calibration.

Command-native definitions are opened with lexical argument substitution. Calls
whose source was learned on measured object roles keep their frozen definitions;
their actual calibrated arguments are recovered using exact stored expressions.
No response fitting or approximate numerical equivalence enters this conversion.
"""

from dataclasses import replace
import json

from .instrument_models import rebind_instrument_model
from .language import Program
from .scale_bindings import bind_measurement_arguments
from .structure_lab import ObjectSchema


def _schema(spec):
    value = spec.context.get("object_schema")
    return ObjectSchema.from_dict(json.loads(value) if isinstance(value, str) else value) if value else None


def _field_projections(source, current):
    """Represent source groups only when current measurements cover their members."""
    result = {}
    for source_group in source.objects:
        wanted = set(source.members[source_group])
        for role, name in source.fields[source_group].items():
            covered, parts = set(), []
            for group in current.objects:
                members = set(current.members[group])
                if not members <= wanted or members & covered or role not in current.fields[group]:
                    continue
                part = Program("feature", name=current.fields[group][role])
                if role == "x" and members != wanted:
                    part = Program("mul", children=(
                        Program("const", value=len(members) / len(wanted)), part))
                parts.append(part)
                covered.update(members)
            if covered == wanted and parts:
                result[name] = parts[0] if len(parts) == 1 else Program("object_sum", children=tuple(parts))
    return result


def _substitute(program, bindings):
    """Substitute this call's lexical arguments without opening nested calls."""
    if program.op == "arg":
        return bindings.get(program.name, program)
    if not program.children:
        return program
    if program.op == "relation_sum":
        from .relation_language import EDGE_ARGUMENTS
        outside = {name: value for name, value in bindings.items() if name not in EDGE_ARGUMENTS}
        return replace(program, children=tuple(_substitute(child, outside if index == 1 else bindings)
                                              for index, child in enumerate(program.children)))
    return replace(program, children=tuple(_substitute(child, bindings) for child in program.children))


def _lagged(program, lag):
    if lag == 0:
        return program
    return bind_measurement_arguments(Program("lag", name="measurement", lag=lag),
                                      {"measurement": program})


def abstract_measurements(program, measured_spec, library):
    """Recover a measured-coordinate program from actual callable provenance.

    The supplied public schema determines the target measurement representation.
    Historical instrument models identify stale arguments as well as current
    ones; recognizing an old expression does not endorse that calibration. Any
    command dependency outside recoverable measured calls remains explicit.
    """
    current = _schema(measured_spec)
    if current is None:
        raise ValueError("measurement abstraction requires a public object schema")
    priors = {prior["source_concept_id"]: prior for prior in library.meta_priors}
    templates = {template.template_id: template for template in library.abstracts}
    models = list(library.instrument_models)
    current_bindings = [rebind_instrument_model(model, measured_spec) for model in models]
    pools, restored_calls, expanded_sources, dependencies, unresolved_arguments = {}, [], [], [], []

    def pool(source_id):
        if source_id in pools:
            return pools[source_id]
        prior = priors[source_id]
        source = ObjectSchema.from_dict(prior.get("object_schema") or
                                        prior["input_program"]["source_schema"])
        projected = _field_projections(source, current)
        source_spec = replace(measured_spec, feature_names=source.feature_names,
                             context={"object_schema": json.dumps(source.to_dict())})
        source_bindings = [rebind_instrument_model(model, source_spec) for model in models]
        entries = {}

        def add(role, actual, measured):
            pair = (actual.simplified(), measured)
            if pair not in entries.setdefault(role, []):
                entries[role].append(pair)

        for group in current.objects:
            for role, name in current.fields[group].items():
                measured = Program("feature", name=name)
                add(role, measured, measured)
                for binding in current_bindings:
                    if name in binding:
                        add(role, binding[name], measured)
        for group in source.objects:
            for role, name in source.fields[group].items():
                measured = projected.get(name)
                if measured is None:
                    continue
                add(role, Program("feature", name=name), measured)
                for binding in source_bindings:
                    if name in binding:
                        add(role, binding[name], measured)
                # A coarse callable can have received the sum of separately
                # calibrated fine instruments rather than one coarse formula.
                for binding in current_bindings:
                    if measured.required_features() <= set(binding):
                        add(role, bind_measurement_arguments(measured, binding), measured)
        pools[source_id] = entries
        return entries

    def measured_call(call, source_id, input_program=None):
        if input_program is None:
            binding = library.definition_bindings.get(call.definition.definition_id)
            if binding is None:
                # A local mechanism replacement keeps its measured source's
                # input slots while receiving a new definition identity.
                binding = next((record for definition_id, record in library.definition_bindings.items()
                    if record["template_id"] == f"{source_id}:abstract:0"
                    and library.definitions[definition_id].arguments == call.definition.arguments), None)
            template = templates.get(binding["template_id"]) if binding else None
        else:
            from .abstractions import AbstractTemplate
            binding = {"arguments": tuple(Program.from_dict(item) for item in input_program["arguments"])}
            template = AbstractTemplate.from_dict(input_program["template"])
        if binding is None or template is None or len(binding["arguments"]) != len(call.children):
            unresolved_arguments.append({"definition_id": call.definition.definition_id,
                                         "reason": "source_argument_roles_unavailable"})
            return call
        source = ObjectSchema.from_dict(input_program["source_schema"] if input_program else
                                        priors[source_id]["object_schema"])
        roles = {name: role for group in source.objects for role, name in source.fields[group].items()}
        original = dict(zip(template.roles, template.source_features))
        recovered, failures = [], []
        for index, (leaf, actual) in enumerate(zip(binding["arguments"], call.children)):
            if leaf.op == "available" and actual == leaf:
                recovered.append(actual)
                continue
            name = original.get(leaf.name)
            role = roles.get(name)
            candidates = []
            if role is not None and leaf.op in {"feature", "lag"}:
                for expression, measurement in pool(source_id).get(role, []):
                    if _lagged(expression, leaf.lag).simplified() == actual.simplified():
                        candidate = _lagged(measurement, leaf.lag)
                        if candidate not in candidates:
                            candidates.append(candidate)
            if len(candidates) == 1:
                recovered.append(candidates[0])
            else:
                recovered.append(actual)
                failures.append({"definition_id": call.definition.definition_id,
                    "argument": index, "role": role,
                    "reason": "ambiguous_calibration" if candidates else "calibration_not_recoverable"})
        unresolved_arguments.extend(failures)
        if not failures:
            restored_calls.append({"source_id": source_id, "definition_id": call.definition.definition_id,
                                   "required_fields": sorted(set().union(
                                       *(child.required_features() for child in recovered)))})
        return replace(call, children=tuple(recovered))

    def visit(node, bindings, local_names=()):
        if node.op == "arg":
            if node.name in bindings:
                return visit(bindings[node.name], {}, local_names)
            if node.name in local_names:
                return node
            unresolved_arguments.append({"argument": node.name, "reason": "unbound_lexical_argument"})
            return node
        if node.op == "call":
            arguments = tuple(_substitute(child, bindings) for child in node.children)
            call = replace(node, children=arguments)
            source = node.definition.source_concept_id
            if source and source not in dependencies:
                dependencies.append(source)
            if priors.get(source, {}).get("object_schema"):
                return measured_call(call, source)
            inputs = priors.get(source, {}).get("input_program")
            if inputs and tuple(inputs["definition"]["arguments"]) == node.definition.arguments:
                return measured_call(call, source, inputs)
            if source and source not in expanded_sources:
                expanded_sources.append(source)
            return visit(node.definition.body, dict(zip(node.definition.arguments, arguments)), local_names)
        if not node.children:
            return node
        if node.op == "relation_sum":
            from .relation_language import EDGE_ARGUMENTS
            outside = {name: value for name, value in bindings.items() if name not in EDGE_ARGUMENTS}
            return replace(node, children=tuple(visit(child, outside if index == 1 else bindings,
                (*local_names, *EDGE_ARGUMENTS) if index == 1 else local_names)
                for index, child in enumerate(node.children)))
        return replace(node, children=tuple(visit(child, bindings, local_names) for child in node.children))

    abstract = visit(program, {})
    variables = {name for name, _ in abstract.observed_variables()}
    fields = set(measured_spec.feature_names)
    unresolved_commands = sorted(variables - fields - {"__target__"})
    complete = not unresolved_commands and not unresolved_arguments
    return {"kind": "relocatable_measurement_program", "program": abstract.to_dict(),
            "source_schema": current.to_dict(), "source_dependencies": dependencies,
            "required_fields": sorted(variables & fields), "complete": complete,
            "unresolved_commands": unresolved_commands, "unresolved_arguments": unresolved_arguments,
            "restored_calls": restored_calls, "expanded_sources": expanded_sources}


def instantiate_measurements(payload, binding):
    """Instantiate a complete measured program using a newly learned calibration."""
    if not payload["complete"]:
        raise ValueError("cannot instantiate an incomplete measurement abstraction")
    missing = set(payload["required_fields"]) - set(binding)
    if missing:
        raise ValueError(f"instrument binding lacks required fields: {sorted(missing)}")
    return bind_measurement_arguments(Program.from_dict(payload["program"]), binding)
