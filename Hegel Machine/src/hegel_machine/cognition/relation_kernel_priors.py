"""Recall measured relation kernels independently of their response wrappers."""

from copy import deepcopy

from .language import Program
from .scale_bindings import bind_measurement_arguments


def parameter_relations(program, symbol):
    own = [program] if (program.op in {"neighbor_sum", "relation_sum"} and
        program.children[0] == Program("arg", name=symbol)) else []
    return [*own, *(item for child in program.children
                   for item in parameter_relations(child, symbol))]


def kernel_key(relation):
    """Keep input computations and roles exact; object summation order is free."""
    offset = 2 if relation.op == "relation_sum" else 1
    triples = tuple(tuple(relation.children[index:index + 3])
                    for index in range(offset, len(relation.children), 3))
    return (relation.op, relation.name or "symmetric",
            relation.children[1] if offset == 2 else None, frozenset(triples), len(triples))


def rebind_relation_inputs(relation, input_bindings, available_features=None):
    """Rewrite a relation with an explicitly learned command/readout binding.

    ``input_bindings`` is evidence supplied by the instrument learner: each
    source measured field is mapped to an executable program over the current
    input rows.  This function deliberately does no role or name inference.
    A partial binding, an invalid serialized program, or a program requiring a
    feature outside the current task is rejected.  Relation radii are left in
    the source measured coordinate system; only measured leaves are replaced.
    """
    if not input_bindings:
        return relation if available_features is None or relation.required_features() <= set(available_features) else None
    required = relation.required_features()
    if not required <= set(input_bindings):
        return None
    mapping = {}
    for name in required:
        value = input_bindings[name]
        if isinstance(value, dict):
            try:
                value = Program.from_dict(value)
            except (KeyError, TypeError, ValueError):
                return None
        if not isinstance(value, (Program, str)):
            return None
        mapping[name] = value
    rebound = bind_measurement_arguments(relation, mapping).simplified()
    if available_features is not None and not rebound.required_features() <= set(available_features):
        return None
    return rebound


def recall_relation_kernels(template, parameters, priors, budget):
    """Return interval hypotheses in the same measured coordinates.

    Radius and outer response are deliberately absent from a kernel key. The
    f/x/q computations, predicate and direction are retained, so a new readout
    can use an old relation without copying its coefficients or target rows.
    Different instruments need an independently learned coordinate binding;
    opaque field names or object indices alone cannot supply one.
    """
    current = {item["symbol"]: {kernel_key(node) for node in
        parameter_relations(template, item["symbol"])} for item in parameters
        if item.get("source_concept_id") is None and not item.get("kernel_priors")}
    result = {symbol: [] for symbol in current}
    if not current:
        return result
    for prior in priors:
        if prior.get("status") != "supported":
            continue
        steps = (prior.get("research_program") or {}).get("selection_steps", [])
        saved = next((step for step in reversed(steps)
                      if step["operation"] == "resolve_relation_intervals"), None)
        if saved is None:
            continue
        source = Program.from_dict(saved["parameter_template"])
        if not budget.spend(source.size):
            break
        for parameter in saved["parameters"]:
            keys = {kernel_key(node) for node in parameter_relations(source, parameter["symbol"])}
            if not keys or not parameter["radius_intervals"]:
                continue
            for symbol, required in current.items():
                if required and required <= keys:
                    record = {"source_concept_id": prior["source_concept_id"],
                              "radius_intervals": deepcopy(parameter["radius_intervals"])}
                    if record not in result[symbol]:
                        result[symbol].append(record)
    return result


def interval_union(intervals):
    result = []
    for low, high in sorted(intervals):
        if result and low <= result[-1][1]:
            result[-1][1] = max(high, result[-1][1])
        else:
            result.append([low, high])
    return result
