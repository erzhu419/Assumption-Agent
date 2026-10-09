"""Rebind a paid experiment's response branch to its current predicate.

The saved branch connects its activating fields to the fields being solved.
Current instrument programs, rather than source command names or object labels,
translate that measured background into commands on the next task.
"""

from .language import Program
from .models import Observation


def _branches(programs, expression):
    from .equivalent_object_programs import canonical_object_program

    expression = canonical_object_program(expression)

    def contains(node):
        return node == expression or any(contains(child) for child in node.children)

    def visit(node):
        if node.op == "if" and contains(node.children[0]):
            yield node
        for child in node.children:
            yield from visit(child)

    seen = set()
    for program in programs:
        for branch in visit(canonical_object_program(program.expanded())):
            if branch not in seen:
                seen.add(branch)
                yield branch


def learn_response_context(programs, expression, binding, context):
    """Keep a source branch whose contribution the paid setup actually excites."""
    row = Observation(context, 0.)
    fields = {name: program.predict(row) for name, program in binding.items()
              if not program.history_depth}
    fields.update(context)
    measured = Observation(fields, 0.)
    for branch in _branches(programs, expression):
        names = {name for name, _ in branch.observed_variables()}
        if not names <= fields.keys() or branch.history_depth:
            continue
        if abs(branch.children[1].predict(measured) - branch.children[2].predict(measured)) <= 1e-12:
            continue
        # Conditions carry activation across changes of calibrated units.
        # Numeric amplitudes need only remain discriminating; copying their
        # exact source values can demand an unreachable value on a new sensor.
        gates = {name for name in names if name in binding and
                 binding[name].op in {"lt", "ge", "and", "or", "not"}}
        if not gates:
            continue
        return {"branch": branch.to_dict(), "fields": {name: fields[name] for name in sorted(gates)}}
    return None


def _match(source, current, mapping):
    if source.op != current.op or source.lag != current.lag or len(source.children) != len(current.children):
        return None
    if source.op in {"feature", "lag"}:
        if source.name in mapping:
            return mapping if mapping[source.name] == current.name else None
        if current.name in mapping.values():
            return None
        return {**mapping, source.name: current.name}
    if source.op == "const":
        return mapping
    if source.name != current.name:
        return None
    orders = [tuple(range(len(current.children)))]
    if source.op in {"mul", "add", "min", "max", "and", "or"} and len(source.children) == 2:
        orders.append((1, 0))
    if source.op == "abs" and source.children[0].op == current.children[0].op == "sub":
        direct = _match(source.children[0], current.children[0], mapping)
        if direct is not None:
            return direct
        swapped = Program("sub", children=tuple(reversed(current.children[0].children)))
        return _match(source.children[0], swapped, mapping)
    for order in orders:
        result = mapping
        # The response establishes quantity/selector roles before an unordered
        # distance expression constrains the two endpoint names.
        positions = (1, 2, 0) if source.op == "if" else range(len(source.children))
        for index in positions:
            result = _match(source.children[index], current.children[order[index]], result)
            if result is None:
                break
        if result is not None:
            return result
    return None


def rebind_response_context(method, expression, programs, binding, context, bounds, spend):
    """Yield measured-role backgrounds for the current matching branches."""
    from .program_disagreement_probes import solve_joint_boundary

    saved = method.get("response_context")
    if saved is None:
        return
    source = Program.from_dict(saved["branch"])
    predicate_fields = {name for name, _ in expression.observed_variables()}
    for branch in _branches(programs, expression):
        if not spend(source.size + branch.size):
            return
        mapping = _match(source, branch, {})
        if mapping is None:
            continue
        point, rebound = dict(context), {}
        for name, value in saved["fields"].items():
            current_name = mapping.get(name)
            if current_name is None or current_name in predicate_fields:
                continue
            instrument = binding.get(current_name)
            if instrument is None and current_name in bounds:
                instrument = Program("feature", name=current_name)
            if instrument is None or instrument.history_depth:
                break
            if instrument.op not in {"lt", "ge", "and", "or", "not"}:
                continue
            controls = {field for field, _ in instrument.observed_variables()}
            if not controls <= bounds.keys() or not controls:
                break
            point, _ = solve_joint_boundary(instrument, value, point, bounds, spend,
                strategy="coordinate" if len(controls) == 1 else "joint_segment",
                control=next(iter(controls)) if len(controls) == 1 else None)
            if point is None:
                break
            rebound[current_name] = value
        else:
            yield point, {"reason": "rebind_predicate_response_branch", "measured_background": rebound}
