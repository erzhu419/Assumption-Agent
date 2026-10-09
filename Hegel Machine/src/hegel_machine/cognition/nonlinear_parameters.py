"""Fit coefficients inside an acquired nonlinear response on paid observations.

The nonlinear consumer and its observed computations stay fixed. Outer linear
coefficients are projected out with the ordinary response fitter at every trial.
Finite differences propose local parameter moves; measured training loss chooses
them. These proposals still need an independent experiment and confirmation.
"""

from math import isfinite

from .continuous_state import _linear_fit
from .language import Program
from .relation_uncertainty import _bind
from .research_programs import _Fitter


def _coefficient(term):
    coefficient = 1.
    while term.op == "mul" and any(child.op == "const" for child in term.children):
        index = next(index for index, child in enumerate(term.children) if child.op == "const")
        coefficient *= term.children[index].value
        term = term.children[1 - index]
    if term.op == "const":
        return coefficient * term.value, Program("const", value=1.)
    return coefficient, term


def _context(terms, coefficients):
    # Keep zero-valued branches explicit during fitting: a subsequent local
    # move must be able to restore a temporarily zero coefficient.
    pieces = [Program("mul", children=(Program("const", value=value), term))
              for term, value in zip(terms, coefficients)]
    result = pieces[0]
    for piece in pieces[1:]:
        result = Program("add", children=(result, piece))
    return result


def _independent_columns(columns):
    # A homogeneous consumer's common input scale is already absorbed by its
    # fitted outer coefficient. Finite differences give that redundant scale
    # a tiny numerical derivative; do not mistake it for an identifiable move.
    selected, basis, centered = [], [], []
    for index, column in enumerate(columns):
        center = sum(column) / len(column)
        residual = [value - center for value in column]
        norm = sum(value * value for value in residual)
        centered.append((norm, index, residual))
    strongest = max(norm for norm, _, _ in centered)
    for norm, index, residual in sorted(centered, reverse=True):
        if norm <= max(1e-20, 1e-8 * strongest):
            continue
        for previous in basis:
            projection = sum(left * right for left, right in zip(residual, previous))
            residual = [value - projection * component for value, component in zip(residual, previous)]
        remaining = sum(value * value for value in residual)
        if remaining <= 1e-6 * norm:
            continue
        selected.append(index)
        basis.append([value / remaining ** .5 for value in residual])
    return selected


def fit_nonlinear_parameters(program, training, budget, *, values, quantile, reference_loss):
    """Return one improved local response, preserving its relation symbols.

    At most two local sums are considered, with at most four coefficients each.
    Zero/sign-reversed starting trials can cross a changed response orientation;
    all magnitudes and subsequent moves derive from the source and observations.
    """
    from .mechanism_revision import _additive_terms, _local_sums, _replace_shared, _signed_terms

    fitter = _Fitter(training, budget, quantile)
    count = len(training)
    best = None
    for context in list(dict.fromkeys(_local_sums(program)))[:2]:
        parsed = [_coefficient(term) for term in _signed_terms(context)]
        if len(parsed) > 4:
            continue
        initial, terms = map(list, zip(*parsed))
        cache = {}

        def evaluate(coefficients):
            key = tuple(coefficients)
            if key in cache:
                return cache[key]
            replacement = _context(terms, coefficients)
            revised, occurrences = _replace_shared(program, context, replacement)
            symbolic_terms = tuple(dict.fromkeys(term for term in _additive_terms(revised)
                                                 if term.op != "const"))
            numerical = tuple(_bind(term, values) for term in symbolic_terms)
            symbolic = dict(zip(numerical, symbolic_terms))
            if len(symbolic) != len(set(symbolic_terms)):
                return None
            # Projection needs its normal fitted columns and comparison, plus
            # the predicted response used by the following finite difference.
            missing = sum(term not in fitter.columns for term in numerical)
            if (missing + 3) * count > budget.remaining:
                return None
            fit = fitter.fit(numerical)
            if fit is None or not budget.spend(count):
                return None
            columns = [fitter.columns[term] for term in fit["fit_terms"]]
            predictions = [fit["coefficients"][0] + sum(coefficient * column[index]
                for coefficient, column in zip(fit["coefficients"][1:], columns))
                for index in range(count)]
            result = {"fit": fit, "symbolic": symbolic, "predictions": predictions,
                "coefficients": list(coefficients), "replacement": replacement,
                "shared_occurrences": occurrences}
            cache[key] = result
            return result

        current = evaluate(initial)
        if current is None:
            break
        starts = [initial]
        for index in range(len(initial)):
            for value in (0., -initial[index]):
                if value == initial[index]:
                    continue
                trial = list(initial)
                trial[index] = value
                starts.append(trial)
        for coefficients in starts[1:]:
            trial = evaluate(coefficients)
            if trial is None:
                break
            if trial["fit"]["loss"] < current["fit"]["loss"]:
                current = trial

        iterations = 0
        for _ in range(12):
            if current["fit"]["loss"] < 1e-12:
                break
            coefficients = current["coefficients"]
            derivatives = []
            for index, value in enumerate(coefficients):
                step = 1e-4 * max(1., abs(value))
                changed = list(coefficients)
                changed[index] += step
                perturbed = evaluate(changed)
                if perturbed is None or not budget.spend(count):
                    break
                derivatives.append([(new - old) / step for new, old in
                                    zip(perturbed["predictions"], current["predictions"])])
            if len(derivatives) != len(coefficients) or not budget.spend(count):
                break
            residuals = [row.target - prediction for row, prediction in
                         zip(training, current["predictions"])]
            independent = _independent_columns(derivatives)
            if not independent:
                break
            fitted_direction = _linear_fit([derivatives[index] for index in independent], residuals)[1:]
            direction = [0.] * len(coefficients)
            for index, value in zip(independent, fitted_direction):
                direction[index] = value
            if not all(isfinite(value) for value in direction):
                break
            scale = max(1., max(abs(delta) / (2. * max(1., abs(value)))
                                for value, delta in zip(coefficients, direction)))
            improved = None
            for fraction in (1., .5, .25, .125):
                changed = [value + fraction * delta / scale
                           for value, delta in zip(coefficients, direction)]
                trial = evaluate(changed)
                if trial is None:
                    break
                if trial["fit"]["loss"] < current["fit"]["loss"]:
                    improved = trial
                    break
            if improved is None:
                break
            current = improved
            iterations += 1

        if current["fit"]["loss"] < reference_loss and (
                best is None or current["fit"]["loss"] < best["fit"]["loss"]):
            best = {**current, "parameter_revisions": [{
                "edit_context": context, "replacement_context": current["replacement"],
                "coefficients_before": initial, "coefficients_after": current["coefficients"],
                "shared_occurrences": current["shared_occurrences"]}],
                "iterations": iterations, "projection_trials": len(cache)}
        if best is not None and best["fit"]["loss"] < 1e-12:
            break
    return best
