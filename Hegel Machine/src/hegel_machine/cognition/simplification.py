"""Local expression identities for finite computations on task trajectories.

This is deliberately narrower than symbolic algebra: summands are not reordered
and fitted coefficients are not rounded or merged. Rewriting must keep required
history and the language's lazy conditional/call behavior.
"""

from math import isfinite

from .language import Program


def _constant(program, value):
    return program.op == "const" and program.value == value


def _current_expression(program):
    # Zero decay discards every previous update, including for a fitted
    # combination of current inputs. Keep its arithmetic order and parameters;
    # only the unnecessary replay is removed. The task's observed controls and
    # numerical computations are finite. Temporal reads remain explicit rather
    # than being mistaken for current coordinates because decay is zero.
    if program.op == "call":
        return _current_expression(program.expanded())
    if program.op in {"const", "feature", "available"}:
        return True
    if program.op not in {"add", "sub", "mul", "abs", "min", "max", "ge", "lt",
                          "and", "or", "not", "if", "object_sum", "neighbor_sum"}:
        return False
    return all(_current_expression(child) for child in program.children)


def simplify(program: Program) -> Program:
    children = tuple(child.simplified() for child in program.children)
    if any(left is not right for left, right in zip(children, program.children)):
        program = Program(program.op, program.value, program.name, program.lag,
                          children, program.definition)
    if not children or program.op == "call":
        # Mutating a shared definition's body under its existing identifier can
        # make two calls disagree about that identifier. Normalize arguments,
        # retain the original callable, and expand explicitly for body analysis.
        return program
    if program.op == "accumulate":
        drive, initial = children
        if program.value == 0 and initial.op == "const" and _current_expression(drive):
            return drive
        return program
    if program.op == "if" and children[0].op == "const":
        return children[1 if children[0].value else 2]
    if program.op == "window_sum" and program.lag == 1:
        return children[0]
    if program.op == "abs" and children[0].op == "abs":
        return children[0]
    if program.op == "add":
        if _constant(children[0], 0):
            return children[1]
        if _constant(children[1], 0):
            return children[0]
    if program.op == "sub" and _constant(children[1], 0):
        return children[0]
    if program.op == "mul":
        if _constant(children[0], 1):
            return children[1]
        if _constant(children[1], 1):
            return children[0]
    if program.op in {"min", "max"} and children[0] == children[1]:
        return children[0]
    # Only evaluate constant operators that do not inspect trajectory length.
    # A constant drive still has meaningful startup behavior in event_delta,
    # window_sum and every nonzero-decay accumulator.
    if program.op in {"add", "sub", "mul", "abs", "not", "min", "max",
                      "ge", "lt", "and", "or", "object_sum"} and all(
                          child.op == "const" for child in children):
        from .models import Observation
        value = program.predict(Observation({}, 0.0))
        if isfinite(value):
            return Program("const", value=value)
    return program
