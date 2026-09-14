"""Compose edge conditions from roles and the ordinary Boolean language."""

from itertools import combinations, permutations

from .language import Program


TRUE = Program("const", value=1.)


def atomic_conditions(role):
    """Compare either endpoint or zero within one measured quantity type."""
    terms = (Program("arg", name=f"edge_source_{role}"),
             Program("arg", name=f"edge_target_{role}"), Program("const", value=0.))
    return [Program("lt", children=pair) for pair in permutations(terms, 2)]


def compound_conditions(atoms):
    for pair in combinations(dict.fromkeys(atoms), 2):
        for operation in ("and", "or"):
            yield Program(operation, children=pair)
