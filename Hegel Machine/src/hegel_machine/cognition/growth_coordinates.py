"""Test reusable programs across public coordinate ranges and unknown role order.

Public bounds propose affine coordinate hypotheses; they do not establish that
two instruments measure the same quantity. Paid response comparisons choose the
role permutation. The caller still validates the selected hypothesis on new data.
"""

from itertools import chain, islice, permutations
from statistics import mean

from .language import Program
from .memory import _called_definitions
from .scale_bindings import bind_measurement_arguments


def normalized_role_bindings(program, source_bounds, target_bounds, rows, budget):
    """Return up to 96 paid role/range hypotheses, ordered by current MSE.

    Each source coordinate is represented by
    source_low + (target - target_low) * source_span / target_span.
    Charges follow candidate-row units, as in ordinary empirical role binding.
    Lexical definitions remain unchanged; their caller supplies new coordinates.
    """
    rows = list(rows)
    if not rows:
        return []
    observed = set.intersection(*(set(row.features) for row in rows))
    roles = sorted({name for name, _ in program.observed_variables() if name != "__target__"})
    names = sorted(name for name, bounds in target_bounds.items()
        if name in observed and bounds[1] > bounds[0])
    if not roles or not set(roles) <= set(source_bounds) or len(roles) > len(names):
        return []
    # A proper lexical call exposes all observed coordinates through arguments.
    # Expand expressions with free features in definitions before substituting.
    lexical = all(not definition.body.observed_variables() for definition in _called_definitions(program))
    template = program if lexical else program.expanded()
    choices = permutations(names, len(roles))
    if set(roles) <= set(names):
        choices = chain((tuple(roles),), (choice for choice in choices if choice != tuple(roles)))
    results = []
    for choice in islice(choices, 96):
        if not budget.spend(len(rows)):
            break
        mapping, substitutions, coordinates = dict(zip(roles, choice)), {}, {}
        for source, target in mapping.items():
            source_low, source_high = source_bounds[source]
            target_low, target_high = target_bounds[target]
            scale = (source_high - source_low) / (target_high - target_low)
            offset = source_low - scale * target_low
            substitutions[source] = Program("add", children=(Program("mul", children=(
                Program("const", value=scale), Program("feature", name=target))),
                Program("const", value=offset))).simplified()
            coordinates[source] = {"target": target, "scale": scale, "offset": offset}
        rebound = bind_measurement_arguments(template, substitutions)
        measured = mean((rebound.predict(row) - row.target) ** 2 for row in rows)
        results.append({"program": rebound, "mapping": mapping, "loss": measured,
            "affine_coordinates": coordinates})
    return sorted(results, key=lambda item: (item["loss"], tuple(sorted(item["mapping"].items()))))
