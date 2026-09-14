"""Lexical edge predicates and their ordinary-expression interpretation."""

from .language import Program


EDGE_ARGUMENTS = ("edge_source_position", "edge_source_quantity",
                  "edge_target_position", "edge_target_quantity")


def relation_objects(program):
    start = 2 if program.op == "relation_sum" else 1
    return tuple(program.children[index:index + 3] for index in range(start, len(program.children), 3))


def edge_bindings(source, target):
    return dict(zip(EDGE_ARGUMENTS, (*source[1:], *target[1:])))


def instantiate_predicate(predicate, source, target):
    """Bind this edge; nested relation predicates retain their own edge scope."""
    return predicate._expand(edge_bindings(source, target))


def expand_relation(program):
    """Expose geometry and the learned predicate to ordinary program planners."""
    radius, predicate = program.children[:2]
    objects, edges = relation_objects(program), []
    zero = Program("const", value=0.)
    for index, source in enumerate(objects):
        selector, position, quantity = source
        for other, target in enumerate(objects):
            if index == other:
                continue
            other_selector, other_position, other_quantity = target
            if program.name in {"right", "left"}:
                left, right = ((position, other_position) if program.name == "right"
                               else (other_position, position))
                distance = Program("sub", children=(right, left))
                geometry = Program("and", children=(Program("lt", children=(zero, distance)),
                                                     Program("lt", children=(distance, radius))))
                contribution = Program("mul", children=(
                    Program("sub", children=(other_selector, selector)), quantity))
            else:
                distance = Program("abs", children=(Program("sub", children=(position, other_position)),))
                geometry = Program("lt", children=(distance, radius))
                contribution = Program("mul", children=(selector,
                    Program("sub", children=(other_quantity, quantity))))
            condition = instantiate_predicate(predicate, source, target)
            # Keep predicate evaluation inside the geometric edge. Its history
            # requirements matter only when that edge actually exists.
            conditional = Program("if", children=(condition, contribution, zero))
            edges.append(Program("if", children=(geometry, conditional, zero)))
    return Program("object_sum", children=tuple(edges))
