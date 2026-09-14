"""Ask for endpoint readings at unresolved cuts in an existing instrument map."""

from .models import Probe


def unmeasured_instrument_conditions(binding, spec, paired_batches, budget):
    """Find learned cuts whose actual input readings have not been bought."""
    cuts, visited = {}, set()

    def visit(program, field):
        key = (program, field)
        if key in visited:
            return True
        if not budget.spend():
            return False
        visited.add(key)
        if program.op in {"lt", "ge"}:
            left, right = program.children
            if left.op == "const":
                left, right = right, left
            if left.op == "feature" and right.op == "const" and left.name in spec.feature_bounds:
                low, high = spec.feature_bounds[left.name]
                if low <= right.value <= high:
                    cuts.setdefault((left.name, right.value), set()).add(field)
        return all(visit(child, field) for child in program.children)

    for field, program in binding.items():
        if not visit(program, field):
            return None
    questions = []
    for (control, value), fields in sorted(cuts.items()):
        observed = False
        for batch in paired_batches:
            for commands, measured in batch:
                if not budget.spend():
                    return None
                if commands.features.get(control) == value and fields <= set(measured.features):
                    observed = True
                    break
            if observed:
                break
        if not observed:
            questions.append({"control": control, "value": value, "fields": sorted(fields)})
    return questions


def instrument_condition_probe(binding, spec, paired_batches, budget, *, reference, serial):
    """Use learned comparisons and paid input rows, independent of response labels."""
    questions = unmeasured_instrument_conditions(binding, spec, paired_batches, budget)
    if not questions:
        return None, []
    questions = questions[:spec.max_probe_length]
    points = tuple({**reference, item["control"]: item["value"]} for item in questions)
    return Probe(f"instrument_conditions:{serial}", points), questions
