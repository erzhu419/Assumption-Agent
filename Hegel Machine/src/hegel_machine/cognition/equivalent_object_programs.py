"""Execute structural explanations shared by unresolved object assignments.

The alternatives are hypotheses about object membership. A shared expression
is executable without choosing one of those alternatives as the true schema.
Expression identity uses the algebra of the actual language, never equality of
predictions on a finite collection of observations.
"""

from dataclasses import replace
import json

from .language import Program
from .models import Candidate
from .scale_bindings import bind_measurement_arguments
from .unified_structure import propose_structures


def _order(program):
    return json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":"))


def canonical_object_program(program):
    """Normalize commutative reductions and symmetric interaction distances.

    A neighborhood is expanded into its directed edge contributions. This
    identifies a two-location exchange under either location assignment while
    retaining the quantity/selector pairing and every distance predicate. It
    does not make arbitrary object assignments equivalent.
    """
    if program.op == "relation_sum":
        from .relation_language import expand_relation
        return canonical_object_program(expand_relation(program).simplified())
    children = tuple(canonical_object_program(child) for child in program.children)
    if program.op == "neighbor_sum":
        radius = children[0]
        objects = [children[index:index + 3] for index in range(1, len(children), 3)]
        edges = []
        for index, (selector, position, quantity) in enumerate(objects):
            for other, (other_selector, other_position, other_quantity) in enumerate(objects):
                if index == other:
                    continue
                if program.name in {"right", "left"}:
                    left, right = (position, other_position) if program.name == "right" else (other_position, position)
                    distance = Program("sub", children=(right, left))
                    predicate = Program("and", children=(
                        Program("lt", children=(Program("const", value=0.), distance)),
                        Program("lt", children=(distance, radius))))
                    transfer = Program("mul", children=(Program("sub", children=(other_selector, selector)), quantity))
                    edges.append(Program("if", children=(predicate, transfer, Program("const", value=0.))))
                    continue
                distance = Program("abs", children=(Program("sub", children=(position, other_position)),))
                edge = Program("if", children=(Program("lt", children=(distance, radius)),
                    Program("mul", children=(selector, Program("sub", children=(other_quantity, quantity)))),
                    Program("const", value=0.)))
                edges.append(edge)
        return canonical_object_program(Program("object_sum", children=tuple(edges)))
    if program.op in {"add", "object_sum"}:
        terms = []
        for child in children:
            terms.extend(child.children if child.op == "object_sum" else (child,))
        return Program("object_sum", children=tuple(sorted(terms, key=_order)))
    if program.op == "mul":
        factors = []

        def collect(child):
            if child.op == "mul":
                for item in child.children:
                    collect(item)
            else:
                factors.append(child)

        for child in children:
            collect(child)
        factors.sort(key=_order)
        result = factors[0]
        for factor in factors[1:]:
            result = Program("mul", children=(result, factor))
        return result
    if program.op == "abs" and children[0].op == "sub":
        difference = Program("sub", children=tuple(sorted(children[0].children, key=_order)))
        return Program("abs", children=(difference,))
    if program.op in {"min", "max", "and", "or"}:
        children = tuple(sorted(children, key=_order))
    return replace(program, children=children)


def _schema_parent(parent, schema):
    """Reuse a search interval only for an alternative it already covered."""
    equivalence = (parent or {}).get("object_equivalence", {})
    if schema not in equivalence.get("schemas", ()) or not parent.get("structure"):
        return None
    return {**parent, "structure": {**parent["structure"], "schema": schema}}


def propose_equivalent_object_programs(spec, training, budget, *, parent=None,
                                      tolerance=1e-7, transitions=(), max_candidates=8):
    """Fit a structural law and check its identity under every permitted schema.

    A truncated assignment search cannot support equivalence over all remaining
    assignments. Each returned structure is one operational representative for
    choosing an experiment; ``object_equivalence.schemas`` retains its full
    alternatives for a later refit. An expression that is identical under every
    role assignment has one set of fit columns, so fitting it again for each
    assignment adds no evidence. Nothing updates the supplied object model.
    """
    from .object_alternatives import enumerate_object_schemas

    model = spec.context.get("object_model")
    if not model or not training:
        return []
    model = json.loads(model) if isinstance(model, str) else model
    start = budget.spent
    enumeration = enumerate_object_schemas(model, budget)
    schemas = enumeration["schemas"]
    if not enumeration["exhaustive"] or len(schemas) < 2:
        return []
    representative = schemas[0]
    local_spec = replace(spec, context={**spec.context, "object_schema": representative})
    entries = propose_structures(local_spec, training, budget,
        parent=_schema_parent(parent, representative), tolerance=tolerance, transitions=transitions)
    common = {}
    for entry in entries:
        program = canonical_object_program(entry["candidate"].program)
        shared = True
        for schema in schemas:
            mapping = {field: schema["fields"][group][role]
                       for group, fields in representative["fields"].items()
                       for role, field in fields.items()}
            rebound = bind_measurement_arguments(entry["candidate"].program, mapping)
            if not budget.spend(program.size):
                shared = False
                break
            if canonical_object_program(rebound) != program:
                shared = False
                break
        if shared:
            previous = common.get(program)
            if previous is None or entry["training_loss"] < previous["training_loss"]:
                common[program] = entry
        if not budget.remaining:
            break
    ranked = sorted(common, key=lambda program: (
        0. if common[program]["training_loss"] <= tolerance else common[program]["training_loss"],
        common[program]["candidate"].program.size, common[program]["training_loss"], _order(program)))
    results = []
    for index, program in enumerate(ranked[:max_candidates]):
        entry = common[program]
        source = entry["candidate"]
        candidate = Candidate(f"equivalent_objects:{index}", source.program,
            "representation", "Execute a law shared by all remaining object assignments", source.parents)
        structure = {**entry["structure"], "candidates": [{
            "candidate_id": candidate.candidate_id, "program": candidate.program.to_dict(),
            "operation": candidate.operation, "rationale": candidate.rationale}],
            "fits": [{**fit, "candidate_id": candidate.candidate_id}
                     for fit in entry["structure"]["fits"]],
            "compute_cost": budget.spent - start}
        results.append({**entry, "candidate": candidate, "kind": "equivalent_objects",
            "structure": structure,
            "object_equivalence": {
                "source_model_id": enumeration.get("source_model_id"),
                "status": "algebraically_shared", "exhaustive": True,
                "schemas": schemas, "unresolved": enumeration["unresolved"],
                "canonical_program": program.to_dict(),
                "member_families": [entry["structure"]["fits"][0]["family"]] * len(schemas),
                "fit_evaluations": 1, "equivalence_checks": len(schemas),
                "compute_cost": budget.spent - start}})
    return results
