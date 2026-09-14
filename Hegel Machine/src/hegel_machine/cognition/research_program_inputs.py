"""Keep a learned research procedure independent of its instrument calibration."""

from copy import deepcopy
from dataclasses import replace

from .abstractions import AbstractTemplate
from .language import Program
from .relocatable_programs import abstract_measurements


def abstract_research_inputs(payload, measured_spec, library):
    """Recover every saved computation in one public measurement representation.

    The original procedure remains available on its original inputs. This second
    procedure preserves its operators and question dependencies; only actual
    learned call arguments are recovered. An unrecoverable computation leaves
    no partly converted procedure to replay.
    """
    result = deepcopy(payload)
    recovered, fields, dependencies = {}, set(), []
    source_schema = None

    def convert(value, graph_arguments=()):
        nonlocal source_schema
        if value is None:
            return None
        program = Program.from_dict(value)
        if program not in recovered:
            recovered[program] = abstract_measurements(program, measured_spec, library)
        converted = recovered[program]
        unresolved = [item for item in converted["unresolved_arguments"]
                      if not (item["reason"] == "unbound_lexical_argument" and
                              item["argument"] in graph_arguments)]
        if converted["unresolved_commands"] or unresolved:
            raise ValueError("research computation has unrecovered measurement inputs")
        source_schema = converted["source_schema"]
        fields.update(converted["required_fields"])
        dependencies.extend(converted["source_dependencies"])
        return converted["program"]

    try:
        for node in result["graph"]["nodes"].values():
            for name in ("expression", "answer", "residual_base"):
                node[name] = convert(node.get(name), node["depends_on"] if name == "expression" else ())
            node["alternatives"] = [convert(program) for program in node["alternatives"]]
            programs = [Program.from_dict(node[name]) for name in ("expression", "answer", "residual_base")
                        if node.get(name) is not None]
            node["required_measurements"] = sorted({name for program in programs
                for name, _ in program.observed_variables() if name != "__target__"})
        for step in result["selection_steps"]:
            original = Program.from_dict(step["program"])
            step["program"] = convert(step["program"])
            if step["operation"] == "resolve_relation_intervals":
                symbols = tuple(item["symbol"] for item in step["parameters"])

                def convert_parameter(value):
                    try:
                        return convert(value, symbols)
                    except ValueError:
                        from .relation_revisions import _replacement_call
                        from .relocatable_programs import _substitute

                        def calls(node):
                            if node.op == "call":
                                yield node
                            else:
                                for child in node.children:
                                    yield from calls(child)

                        anchor = next(calls(original), None)
                        template = Program.from_dict(value)
                        scoped = _replacement_call(original, template, "input_template")
                        if (anchor is None or scoped.children != anchor.children or
                                scoped.definition.arguments != anchor.definition.arguments):
                            raise
                        # Recover an opened computation using this step's actual
                        # calibrated call slots, then reopen the temporary call.
                        contextual = replace(anchor, definition=replace(anchor.definition,
                            body=scoped.definition.body))
                        measured = Program.from_dict(convert(contextual.to_dict(), symbols))
                        if measured.op == "call":
                            measured = _substitute(measured.definition.body,
                                dict(zip(measured.definition.arguments, measured.children)))
                        return measured.to_dict()

                step["parameter_template"] = convert_parameter(step["parameter_template"])
                if "parameter_terms" in step:
                    step["parameter_terms"] = [convert_parameter(term) for term in step["parameter_terms"]]
    except ValueError:
        return None
    if source_schema is None:
        return None

    # A selected answer may drop a term whose coefficient was zero. The saved
    # procedure still retests that term, so bind its full input set rather than
    # only the final answer's template.
    names = tuple(sorted(fields))
    roles = tuple(f"input_{index}" for index in range(len(names)))
    object_fields = {name: (index, role) for index, obj in enumerate(source_schema["objects"])
                     for role, name in source_schema["fields"][obj].items()}
    template = AbstractTemplate("research_inputs", "research_procedure",
        Program("object_sum", children=tuple(Program("feature", name=role) for role in roles)),
        roles, names, (), object_roles={role: object_fields[name]
                                      for role, name in zip(roles, names)})
    result.update(input_template=template.to_dict(), source_schema=source_schema,
                  input_features=list(names), input_source_ids=list(dict.fromkeys(dependencies)))
    return result
