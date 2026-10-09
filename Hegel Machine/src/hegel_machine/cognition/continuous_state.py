"""Discover continuous memory from observed trajectories and executable filters.

Recorded outputs can identify candidate poles through an autoregression. The
resulting hypotheses replay only observed input features, including on a new
trajectory with no past outputs. Nonlinear readouts are fitted from filtered
coordinates and their products; their decay is refined on the supplied data.
"""

from itertools import combinations_with_replacement
from math import sqrt

from .language import Program
from .models import Candidate
from .synthesis import _binary, _constant, _solve


def _linear_fit(columns, targets):
    count = len(targets)
    centers = [sum(column) / count for column in columns]
    scales = [max(abs(value - center) for value in column) or 1.0
              for column, center in zip(columns, centers)]
    design = [[1.0, *((column[index] - center) / scale
                     for column, center, scale in zip(columns, centers, scales))]
              for index in range(count)]
    width = len(columns) + 1
    gram = [[sum(row[i] * row[j] for row in design) for j in range(width)]
            for i in range(width)]
    rhs = [sum(row[i] * target for row, target in zip(design, targets)) for i in range(width)]
    for index in range(1, width):
        gram[index][index] += 1e-9
    fitted = _solve(gram, rhs)
    slopes = [value / scale for value, scale in zip(fitted[1:], scales)]
    return [fitted[0] - sum(value * center for value, center in zip(slopes, centers)), *slopes]


def _sum_terms(terms, coefficients, intercept=0.0):
    parts = [_constant(intercept)] if abs(intercept) >= 1e-8 else []
    for term, value in zip(terms, coefficients):
        if abs(value) < 1e-8:
            continue
        part = term if abs(value - 1.0) < 1e-8 else _binary("mul", _constant(value), term)
        parts.append(part)
    # Object views can produce hundreds of terms. Keep their executable sum
    # shallow enough to simplify, serialize and reuse as a callable concept.
    while len(parts) > 1:
        parts = [_binary("add", parts[index], parts[index + 1])
                 if index + 1 < len(parts) else parts[index]
                 for index in range(0, len(parts), 2)]
    return parts[0] if parts else _constant(intercept)


def _filter(drive, decay, initial=0.0):
    return Program("accumulate", value=float(decay), children=(drive, _constant(initial)))


def _poles(program):
    program = program.expanded()
    result = [program.value] if program.op == "accumulate" else []
    return result + [value for child in program.children for value in _poles(child)]


def _linear_program(raw, poles, coefficients):
    """A weighted sum of equal-decay filters is one learned driving direction."""
    count = len(raw)
    result = _sum_terms(raw, coefficients[1:count + 1], coefficients[0])
    offset = count + 1
    for decay in poles:
        slopes = coefficients[offset:offset + count]
        initial = coefficients[offset + count]
        offset += count + 1
        if max([abs(initial), *map(abs, slopes)]) < 1e-8:
            continue
        state = _filter(_sum_terms(raw, slopes), decay, initial)
        result = state if result.op == "const" and abs(result.value) < 1e-8 else _binary("add", result, state)
    return result


def _quadratic_terms(raw, filters, coefficients, pairs, max_states):
    """Propose states; their readout must be fitted after compression.

    A trial decay can give the homogeneous initial-state column a huge
    compensating coefficient. That is not evidence for a useful driving
    direction. Infer the direction from the driven block and retain its fitted
    initial value, then measure the resulting small representation itself.
    """
    count = len(raw)
    linear = coefficients[1 + count:1 + count + len(filters)]
    quadratic = coefficients[1 + count + len(filters):]
    matrix = [[0.0] * len(filters) for _ in filters]
    for (left, right), value in zip(pairs, quadratic):
        matrix[left][right] = matrix[right][left] = value if left == right else value / 2
    pivot = max(range(count), key=lambda index: abs(matrix[index][index]))
    scale = matrix[pivot][pivot]
    terms = [*raw, *filters, *[_binary("mul", filters[i], filters[j]) for i, j in pairs]]
    unrestricted = _sum_terms(terms, coefficients[1:], coefficients[0])
    if abs(scale) > 1e-8:
        direction = [matrix[pivot][index] / scale for index in range(len(filters))]
        error = max(abs(matrix[i][j] - scale * direction[i] * direction[j])
                    for i in range(len(filters)) for j in range(len(filters)))
        magnitude = max(abs(value) for row in matrix for value in row)
        if error > 1e-5 * max(magnitude, 1e-8) and len(_states(unrestricted)) <= max_states:
            return terms
        state = _filter(_sum_terms(raw, direction[:count]), filters[0].value, direction[-1])
        result = [*raw, state, _binary("mul", state, state)]
        remainder = [value - linear[pivot] * weight for value, weight in zip(linear, direction)]
        if max_states > 1 and max(map(abs, remainder)) > 1e-8:
            result.append(_filter(_sum_terms(raw, remainder[:count]), filters[0].value, remainder[-1]))
        return result
    if len(_states(unrestricted)) <= max_states:
        return terms
    return [*raw, _filter(_sum_terms(raw, linear[:count]), filters[0].value, linear[-1])]


def _states(program):
    return ({program} if program.op == "accumulate" else set()).union(
        *(_states(child) for child in program.children))


def state_identification_cost(observations):
    """Fund first- and second-order identification and their executable fits.

    Each order includes a cold fit, so shared columns can only reduce its bill.
    The caller separately funds scoring at most six returned candidates.
    """
    rows = list(observations)
    if len(rows) < 6 or all(row.target in (0., 1.) for row in rows):
        return 0
    names = set.intersection(*(set(row.features) for row in rows)) - {"__target__"}
    names = {name for name in names if all(name in past for row in rows for past in row.history)}
    if not names or max((len(row.history) for row in rows), default=0) < 2:
        return 0
    total = 0
    for order in (1, 2):
        applicable = sum(len(row.history) >= order
            and all("__target__" in past for past in row.history[-order:]) for row in rows)
        width = order + len(names) * (order + 1)
        if applicable >= max(6, width + 1):
            total += (width + 2) * (applicable + len(rows))
    return total


def continuous_state_candidates(observations, *, budget, max_states=2, seed_programs=()):
    """Fit one or two continuous states without an environment or task label.

    Each new column costs one candidate-row pass, as do each complete fit and
    executable prediction check. Columns are cached and the entire applicable
    evidence is retained. Before measuring a column, its complete fit and check
    must be affordable. Output hypotheses are selected later on new trajectories
    by the ordinary controller.
    """
    if max_states not in (1, 2):
        raise ValueError("continuous state search supports one or two states")
    rows = list(observations)
    if len(rows) < 6 or max(map(lambda row: len(row.history), rows), default=0) < 2:
        return []
    if all(row.target in (0.0, 1.0) for row in rows):
        return []
    names = sorted(set.intersection(*(set(row.features) for row in rows)) - {"__target__"})
    names = [name for name in names if all(name in past for row in rows for past in row.history)]
    if not names:
        return []
    raw = [Program("feature", name=name) for name in names]
    count, columns, tested, results = len(rows), {}, {}, {}
    targets = [row.target for row in rows]

    def fit(poles, quadratic=False):
        poles = tuple(round(float(value), 12) for value in poles)
        key = (poles, quadratic)
        if key in tested:
            return tested[key]
        filters = [_filter(drive, decay, initial) for decay in poles
                   for drive, initial in [*((term, 0.0) for term in raw), (_constant(0.0), 1.0)]]
        pairs = list(combinations_with_replacement(range(len(filters)), 2)) if quadratic else []
        terms = [*raw, *filters, *[_binary("mul", filters[i], filters[j]) for i, j in pairs]]
        missing = [term for term in dict.fromkeys(terms) if term not in columns]
        # Compression can introduce at most three columns, another fit, and
        # its executable check. Reserve these before fitting the broad model.
        if (len(missing) + (6 if quadratic else 2)) * count > budget.remaining:
            return None
        for term in missing:
            budget.spend(count)
            columns[term] = [term.predict(row) for row in rows]
        budget.spend(count)
        coefficients = _linear_fit([columns[term] for term in terms], targets)
        if quadratic:
            compact = _quadratic_terms(raw, filters, coefficients, pairs, max_states)
            if compact != terms:
                for term in dict.fromkeys(compact):
                    if term not in columns:
                        budget.spend(count)
                        columns[term] = [term.predict(row) for row in rows]
                budget.spend(count)
                coefficients = _linear_fit([columns[term] for term in compact], targets)
            program = _sum_terms(compact, coefficients[1:], coefficients[0])
        else:
            program = _linear_program(raw, poles, coefficients)
        budget.spend(count)
        error = sum((program.predict(row) - row.target) ** 2 for row in rows) / count
        if 0 < len(_states(program)) <= max_states:
            previous = results.get(program)
            if previous is None or error < previous:
                results[program] = error
        else:
            # A discarded intermediate cannot satisfy the stopping criterion.
            error = None
        tested[key] = error
        return error

    def identify(order):
        applicable = [row for row in rows if len(row.history) >= order
                      and all("__target__" in past for past in row.history[-order:])]
        width = order + len(raw) * (order + 1)
        reserve = (2 * len(raw) + 3) * count
        if len(applicable) < max(6, width + 1) or (width + 2) * len(applicable) + reserve > budget.remaining:
            return []
        values = [[row.history[-lag]["__target__"] for row in applicable] for lag in range(1, order + 1)]
        for lag in range(order + 1):
            values.extend([[row.features[name] if lag == 0 else row.history[-lag][name]
                            for row in applicable] for name in names])
        budget.spend((width + 1) * len(applicable))
        coefficients = _linear_fit(values, [row.target for row in applicable])
        # Evaluate the fitted identifying equation on those same observed rows.
        budget.spend(len(applicable))
        residual = [row.target - coefficients[0] - sum(coef * column[index]
                    for coef, column in zip(coefficients[1:], values)) for index, row in enumerate(applicable)]
        center = sum(row.target for row in applicable) / len(applicable)
        if sum(value * value for value in residual) >= sum((row.target - center) ** 2 for row in applicable):
            return []
        if order == 1:
            proposed = coefficients[1:2]
        else:
            discriminant = coefficients[1] ** 2 + 4 * coefficients[2]
            proposed = [] if discriminant < 0 else [(coefficients[1] + sign * sqrt(discriminant)) / 2
                                                    for sign in (1, -1)]
        return [value for value in proposed if -0.995 <= value <= 0.9995]

    def finish():
        ordered = sorted(results, key=lambda program: (results[program], program.size))[:6]
        return [Candidate(f"continuous_state:{index}:{budget.spent}", program, "representation",
                          "Infer continuous memory and its readout from observed trajectories")
                for index, program in enumerate(ordered)]

    # Existing learned decays are concrete reuse candidates; their input/output
    # coefficients are fitted again to the current observations.
    seeds = list(dict.fromkeys(value for program in seed_programs for value in _poles(program)))
    for value in seeds:
        error = fit((value,))
        if error is not None and error < 1e-12:
            return finish()
    estimated = identify(1)
    for value in estimated:
        error = fit((value,))
        if error is not None and error < 1e-12:
            return finish()
    if max_states == 2:
        pair = identify(2)
        if len(pair) == 2 and abs(pair[0] - pair[1]) > 1e-5:
            error = fit(tuple(pair))
            if error is not None and error < 1e-12:
                return finish()
        estimated.extend(pair)

    scored, families = {}, {}
    for value in dict.fromkeys([*seeds, *estimated, 0.0, 0.25, 0.5, 0.7, 0.85, 0.95, 0.985, -0.5]):
        linear = fit((value,))
        quadratic = fit((value,), True)
        available = [error for error in (linear, quadratic) if error is not None]
        if available:
            scored[value] = min(available)
            families[value] = quadratic is not None and (linear is None or quadratic < linear)
        if available and min(available) < 1e-12:
            return finish()
    if not scored:
        return finish()
    ordered = sorted(scored)
    best = min(scored, key=scored.get)
    index = ordered.index(best)
    left = ordered[index - 1] if index else -0.995
    right = ordered[index + 1] if index + 1 < len(ordered) else 0.9995
    # Refine the measured winning readout family. Parabolic interpolation uses
    # the already paid neighboring trials; golden steps keep the search inside
    # its bracket when that interpolation is not informative.
    quadratic = families[best]
    neighbors = sorted((value for value in scored if value != best and families[value] == quadratic),
                       key=lambda value: abs(value - best))[:2]
    x, w, v = best, *(neighbors + [best, best])[:2]
    fx, fw, fv = scored[x], scored[w], scored[v]
    previous_step, step = right - left, 0.0
    for _ in range(32):
        if fx < 1e-12 or right - left < 1e-8:
            break
        midpoint = (left + right) / 2
        tolerance = 1e-8 * max(1.0, abs(x))
        numerator = (x - v) ** 2 * (fx - fw) - (x - w) ** 2 * (fx - fv)
        denominator = 2 * ((x - v) * (fx - fw) - (x - w) * (fx - fv))
        proposed = x - numerator / denominator if denominator else midpoint
        if (denominator and left + tolerance < proposed < right - tolerance
                and abs(proposed - x) < abs(previous_step) / 2):
            previous_step, step = step, proposed - x
        else:
            previous_step = right - x if x < midpoint else left - x
            step = (3 - sqrt(5.0)) / 2 * previous_step
        trial = x + (step if abs(step) >= tolerance else tolerance if step >= 0 else -tolerance)
        error = fit((trial,), quadratic)
        if error is None:
            break
        if error <= fx:
            if trial < x:
                right = x
            else:
                left = x
            v, fv, w, fw, x, fx = w, fw, x, fx, trial, error
        else:
            if trial < x:
                left = trial
            else:
                right = trial
            if error <= fw or w == x:
                v, fv, w, fw = w, fw, trial, error
            elif error <= fv or v in (x, w):
                v, fv = trial, error
    return finish()
