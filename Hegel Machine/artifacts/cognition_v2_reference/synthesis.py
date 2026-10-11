"""Budgeted, data-driven search over the common executable language.

One search unit is one candidate evaluated on one observed row, including its
fit. Operations share a round-robin budget; labels are used only from the
observations explicitly supplied by the caller. There is no environment-type
dispatch, formal admission gate, or domain-wide behavioral deduplication here.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from itertools import combinations
from math import isfinite

from .language import Program
from .models import Candidate, Observation, SearchBudget

OPERATIONS = (
    "refit", "representation", "relation", "composition",
    "robustification", "idealization",
)


def _constant(value: float) -> Program:
    return Program("const", value=float(value))


def _binary(op: str, left: Program, right: Program) -> Program:
    if op in {"add", "mul", "and", "or", "min", "max"}:
        left, right = sorted((left, right), key=lambda item: item.describe())
    return Program(op, children=(left, right))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (index - low) * (ordered[high] - ordered[low])


def _loss(program: Program, rows: Sequence[Observation], quantile: float) -> float:
    residuals = [row.target - program.predict(row) for row in rows]
    if quantile == 0.5:
        return sum(value * value for value in residuals) / len(rows)
    return sum(max(quantile * value, (quantile - 1) * value) for value in residuals) / len(rows)


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Small dense least squares systems with feature-scaled regularization."""
    width = len(rhs)
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for index in range(width):
        pivot = max(range(index, width), key=lambda row: abs(augmented[row][index]))
        augmented[index], augmented[pivot] = augmented[pivot], augmented[index]
        divisor = augmented[index][index]
        if abs(divisor) < 1e-12:
            continue
        augmented[index] = [value / divisor for value in augmented[index]]
        for row in range(width):
            if row == index:
                continue
            factor = augmented[row][index]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[index])
            ]
    return [row[-1] for row in augmented]


def _fit_linear(terms: Sequence[Program], rows: Sequence[Observation]) -> Program:
    if not terms:
        return _constant(sum(row.target for row in rows) / len(rows))
    columns = [[term.predict(row) for row in rows] for term in terms]
    means = [sum(column) / len(column) for column in columns]
    scales = [max(abs(value - mean) for value in column) or 1.0
              for column, mean in zip(columns, means)]
    design = [
        [1.0] + [(column[index] - mean) / scale
                 for column, mean, scale in zip(columns, means, scales)]
        for index in range(len(rows))
    ]
    width = len(terms) + 1
    gram = [[sum(row[left] * row[right] for row in design)
             for right in range(width)] for left in range(width)]
    rhs = [sum(row[index] * observation.target for row, observation in zip(design, rows))
           for index in range(width)]
    for index in range(1, width):
        gram[index][index] += 1e-9
    coefficients = _solve(gram, rhs)
    slopes = [coefficient / scale for coefficient, scale in zip(coefficients[1:], scales)]
    intercept = coefficients[0] - sum(slope * mean for slope, mean in zip(slopes, means))
    program = _constant(intercept)
    for term, slope in zip(terms, slopes):
        if abs(slope) < 1e-8:
            continue
        contribution = term if abs(slope - 1) < 1e-8 else _binary("mul", _constant(slope), term)
        program = contribution if abs(intercept) < 1e-8 and program.op == "const" else _binary("add", program, contribution)
    return program


def _thresholds(term: Program, rows: Sequence[Observation]) -> list[float]:
    values = sorted({term.predict(row) for row in rows})
    midpoints = [(left + right) / 2 for left, right in zip(values, values[1:])]
    if len(midpoints) <= 7:
        return midpoints
    return [midpoints[round(index * (len(midpoints) - 1) / 6)] for index in range(7)]


def _terms(rows: Sequence[Observation], max_history: int = 4) -> tuple[list[Program], list[Program]]:
    current_names = sorted(set.intersection(*(set(row.features) for row in rows)))
    current = [Program("feature", name=name) for name in current_names]
    historical: list[Program] = []
    for lag in range(1, min(max_history, min(len(row.history) for row in rows)) + 1):
        names = sorted(set.intersection(*(set(row.history[-lag]) for row in rows)))
        historical.extend(Program("lag", name=name, lag=lag) for name in names)
    return current, historical


Proposal = tuple[Callable[[], Program], str]


def synthesize(
    observations: Sequence[Observation], *, operations: Sequence[str],
    budget: SearchBudget, incumbent: Program | None = None,
    risk_quantile: float = 0.5,
    max_history: int = 4,
    seed_programs: Sequence[Program] = (),
) -> list[Candidate]:
    if not observations:
        return []
    if not 0 < risk_quantile < 1:
        raise ValueError("risk quantile must be between zero and one")
    unknown = set(operations) - set(OPERATIONS)
    if unknown:
        raise ValueError(f"unknown cognitive operations: {sorted(unknown)}")
    rows = tuple(observations)
    current, historical = _terms(rows, max_history)
    intermediate_terms = [program for program in seed_programs
                          if program.op != "const" and program not in current + historical
                          and program.history_depth <= min(len(row.history) for row in rows)]
    all_terms = list(dict.fromkeys([*intermediate_terms, *current, *historical]))
    binary_targets = all(row.target in (0.0, 1.0) for row in rows)
    thresholds_by_term = [
        sorted(_thresholds(term, rows), key=lambda value: abs(value - _quantile(
            [term.predict(row) for row in rows], 0.5)))
        for term in all_terms
    ]
    # Interleave coordinates so the first raw feature cannot consume every
    # threshold trial before the search ever reaches another feature or lag.
    atoms = [
        _binary(direction, term, _constant(thresholds[index]))
        for index in range(max((len(values) for values in thresholds_by_term), default=0))
        for term, thresholds in zip(all_terms, thresholds_by_term)
        if index < len(thresholds)
        for direction in ("ge", "lt")
    ]
    scored: list[tuple[float, Program]] = []
    for seed in seed_programs:
        if seed.history_depth <= min(len(row.history) for row in rows) and budget.spend(len(rows)):
            scored.append((_loss(seed, rows, risk_quantile), seed))

    def linear(terms: Sequence[Program], reason: str) -> Proposal:
        frozen = tuple(terms)
        return lambda: _fit_linear(frozen, rows), reason

    def refits() -> Iterator[Proposal]:
        yield linear((), "fit a constant to observed targets")
        yield linear(current, "fit the observed current features jointly")
        for term in current:
            yield linear((term,), "refit an observed feature")
        if incumbent is not None and incumbent.history_depth <= min(len(row.history) for row in rows):
            yield linear((incumbent,), "calibrate the incumbent prediction from residual evidence")
        if len(current) > 2:
            for pair in combinations(current, 2):
                yield linear(pair, "fit a pair of current features")

    def representations() -> Iterator[Proposal]:
        for term in historical:
            yield linear(current + [term], "test a historical observable as an additional coordinate")
        if historical:
            yield linear(all_terms, "fit the available historical coordinates jointly")
        for term in historical:
            yield linear((term,), "test a historical coordinate on its own")
        for term in current:
            yield linear((Program("abs", children=(term,)),), "test an absolute-value coordinate")

    def relations() -> Iterator[Proposal]:
        for atom in atoms:
            if binary_targets:
                yield (lambda atom=atom: atom), "test an observed threshold relation"
            else:
                def fit_stump(atom: Program = atom) -> Program:
                    groups = [[row.target for row in rows if bool(atom.predict(row)) == side]
                              for side in (True, False)]
                    predictions = [sum(group) / len(group) for group in groups]
                    return Program("if", children=(atom, _constant(predictions[0]), _constant(predictions[1])))
                yield fit_stump, "fit a piecewise relation at an observed threshold"
        for left, right in combinations(all_terms, 2):
            if binary_targets:
                yield lambda left=left, right=right: _binary("ge", left, right), "compare two observed coordinates"

    def compositions() -> Iterator[Proposal]:
        # Consume the previous plan stage before spending its entire budget on
        # another independent search over the original coordinates.
        if not binary_targets:
            for seed in intermediate_terms:
                yield linear((seed,), "calibrate an intermediate representation")
                yield linear(current + [_binary("mul", seed, seed)], "compose a nonlinear intermediate representation")
                for term in current + historical:
                    yield linear(current + [_binary("mul", seed, term)], "combine an intermediate representation with an observable")
        if binary_targets:
            # Scores are filled by the shared scheduler. Keep all distinct
            # programs: equal predictions so far need not remain equal later.
            attempted: set[Program] = set()
            seed_index = 0
            composition_count = 0
            while True:
                if seed_index < len(atoms) and composition_count % 3 == 0:
                    atom = atoms[seed_index]
                    seed_index += 1
                    composition_count += 1
                    yield lambda atom=atom: atom, "seed composition with an observed relation"
                    continue
                by_size: dict[int, dict[Program, float]] = {}
                for loss, program in scored:
                    if program.op in {"ge", "lt", "and", "or", "not"}:
                        by_size.setdefault(program.size, {})[program] = loss
                # Preserve a small beam at each expression size. A useful
                # intermediate relation can tie a shorter model's current loss
                # and only become better after another composition.
                pool = sorted(
                    (item for size, programs in by_size.items() if size <= 11
                     for item in sorted(
                         ((loss, program) for program, loss in programs.items()),
                         key=lambda item: (item[0], item[1].describe()),
                     )[:4]),
                    key=lambda item: (item[0], item[1].size, item[1].describe()),
                )
                proposal = None
                for _, parent in pool:
                    if parent.size > 11:
                        continue
                    for atom in atoms:
                        if atom == parent:
                            continue
                        for op in ("and", "or"):
                            program = _binary(op, parent, atom)
                            if program not in attempted:
                                attempted.add(program)
                                proposal = program
                                break
                        if proposal is not None:
                            break
                    if proposal is not None:
                        break
                if proposal is None:
                    break
                composition_count += 1
                yield lambda proposal=proposal: proposal, "compose observed Boolean relations"
        for left, right in combinations(all_terms, 2):
            interaction = _binary("mul", left, right)
            yield linear(current + [interaction], "fit an interaction of observed coordinates")
        for term in all_terms:
            yield linear(current + [_binary("mul", term, term)], "test a nonlinear coordinate composed from an observable")

    def robustifications() -> Iterator[Proposal]:
        def calibrated(condition: Program | None = None) -> Program:
            base = incumbent
            if base is None or base.history_depth > min(len(row.history) for row in rows):
                base = _fit_linear(current, rows)
            residuals = [(row, row.target - base.predict(row)) for row in rows]
            if condition is None:
                offset = _constant(_quantile([value for _, value in residuals], risk_quantile))
            else:
                groups = [[value for row, value in residuals if bool(condition.predict(row)) == side]
                          for side in (True, False)]
                offset = Program("if", children=(
                    condition, _constant(_quantile(groups[0], risk_quantile)),
                    _constant(_quantile(groups[1], risk_quantile)),
                ))
            return _binary("add", base, offset)
        yield lambda: calibrated(), "calibrate risk from the empirical residual quantile"
        for atom in atoms:
            if atom.children[0].history_depth == 0:
                yield lambda atom=atom: calibrated(atom), "fit conditional residual quantiles across an observed boundary"

    def idealizations() -> Iterator[Proposal]:
        required = incumbent.required_features() if incumbent else {term.name for term in current}
        terms = [term for term in current if term.name in required]
        for removed in terms:
            yield linear([term for term in terms if term != removed], "remove a feature and refit the reduced model")
        if incumbent is not None and incumbent.history_depth:
            yield linear(current, "test whether current evidence still requires historical state")
        yield linear((), "test whether the remaining structure can be replaced by a constant")

    generators = {
        "refit": refits, "representation": representations, "relation": relations,
        "composition": compositions, "robustification": robustifications,
        "idealization": idealizations,
    }
    queue = deque((operation, generators[operation]()) for operation in dict.fromkeys(operations))
    candidates: list[Candidate] = []
    seen: set[tuple[str, Program]] = set()
    while queue and budget.remaining >= len(rows):
        operation, proposals = queue.popleft()
        try:
            factory, rationale = next(proposals)
        except StopIteration:
            continue
        if not budget.spend(len(rows)):
            break
        program = factory()
        loss = _loss(program, rows, risk_quantile)
        queue.append((operation, proposals))
        if not isfinite(loss):
            continue
        scored.append((loss, program))
        key = (operation, program)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(Candidate(
            candidate_id=f"{operation}:{budget.spent}:{len(candidates)}",
            program=program, operation=operation, rationale=rationale,
        ))
    measured_losses = {program: measured for measured, program in scored}
    return sorted(candidates, key=lambda candidate: (measured_losses[candidate.program], candidate.program.size))
