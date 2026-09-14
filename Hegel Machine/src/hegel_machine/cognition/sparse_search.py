"""Budgeted joint refitting and revisable sparse supports across history depths."""

from dataclasses import dataclass
from itertools import combinations
from statistics import mean

from .language import Program
from .models import Candidate
from .synthesis import _binary, _constant, _fit_linear, _solve, _terms


def _greedy_candidates(observations, *, budget, max_history, max_terms):
    if not observations or all(row.target in (0.0, 1.0) for row in observations):
        return []
    depth = max((k for k in range(1, max_history + 1)
                 if sum(len(row.history) >= k for row in observations) >= 6), default=0)
    rows = [row for row in observations if len(row.history) >= depth]
    current, history = _terms(rows, max_history)
    terms = [*current,
             *[Program("mul", children=(term, term)) for term in current],
             *[Program("mul", children=(a, b)) for a, b in combinations(current, 2)],
             *history]
    columns = {}
    for term in terms:
        if not budget.spend(len(rows)):
            break
        values = [term.predict(row) for row in rows]
        center = mean(values)
        centered = [value - center for value in values]
        if sum(value * value for value in centered) > 1e-10:
            columns[term] = centered
    residuals = [row.target - mean(r.target for r in rows) for row in rows]
    selected, results = [], []
    for index in range(min(max_terms, len(columns))):
        scores = {term: sum(x * r for x, r in zip(column, residuals)) ** 2 / sum(x * x for x in column)
                  for term, column in columns.items() if term not in selected}
        if not scores or max(scores.values()) < 1e-10 or not budget.spend(len(rows)):
            break
        selected.append(max(scores, key=scores.get))
        program = _fit_linear(selected, rows)
        residuals = [row.target - program.predict(row) for row in rows]
        results.append(Candidate(f"sparse:{index}:{budget.spent}", program, "composition",
                                 "Compose coordinates selected by unexplained residual variation"))
    return results


@dataclass
class _Fit:
    terms: tuple
    program: Program
    residuals: list
    loss: float
    importance: dict


def _fit_columns(terms, indices, columns, targets):
    """Joint least squares using already measured columns, including refits."""
    means, scales, values = [], [], []
    for term in terms:
        column = [columns[term][index] for index in indices]
        center = mean(column)
        scale = max(abs(value - center) for value in column) or 1.0
        values.append(column)
        means.append(center)
        scales.append(scale)
    design = [[1.0] + [(column[index] - center) / scale
                       for column, center, scale in zip(values, means, scales)]
              for index in range(len(indices))]
    width = len(terms) + 1
    gram = [[sum(row[left] * row[right] for row in design) for right in range(width)]
            for left in range(width)]
    rhs = [sum(row[index] * targets[source] for row, source in zip(design, indices))
           for index in range(width)]
    for index in range(1, width):
        gram[index][index] += 1e-9
    coefficients = _solve(gram, rhs)
    slopes = [value / scale for value, scale in zip(coefficients[1:], scales)]
    intercept = coefficients[0] - sum(slope * center for slope, center in zip(slopes, means))
    program = _constant(intercept)
    retained = []
    for term, slope in zip(terms, slopes):
        if abs(slope) < 1e-8:
            continue
        retained.append(term)
        contribution = term if abs(slope - 1.0) < 1e-8 else _binary("mul", _constant(slope), term)
        program = contribution if program.op == "const" and abs(intercept) < 1e-8 else _binary("add", program, contribution)
    residuals = [targets[source] - sum(value * coefficient for value, coefficient in zip(row, coefficients))
                 for row, source in zip(design, indices)]
    return _Fit(tuple(retained), program, residuals, mean(value * value for value in residuals),
                {term: abs(value) for term, value in zip(terms, coefficients[1:])})


def _concept_candidates(observations, additional_terms, budget):
    """Fit learned computations and their compositions as search coordinates."""
    results = []
    for atom in dict.fromkeys(additional_terms):
        rows = [row for row in observations
                if len(row.history) >= atom.minimum_history_depth
                and atom.required_features().issubset(row.features)]
        if len(rows) < 6:
            continue
        current, _ = _terms(rows, 0)
        terms = list(dict.fromkeys([atom, Program("mul", children=(atom, atom)), *current]))
        # Every measured column and the complete fit use all applicable rows.
        if not budget.spend((len(terms) + 1) * len(rows)):
            continue
        program = _fit_linear(terms, rows)
        results.append(Candidate(f"concept_composition:{len(results)}:{budget.spent}",
                                 program, "composition", "Compose a learned executable definition"))
    return results


def sparse_candidates(observations, *, budget, max_history=4, max_terms=4, strategy="beam",
                      additional_terms=(), select_additional_terms=None):
    """Compare history supports and recover from misleading early coordinates.

    Broad current-coordinate fits and one-lag extensions initialize two supports
    per depth. Joint coefficient pruning and bounded add/swap passes can remove
    an earlier choice. Depth zero uses every row; each other depth keeps all rows
    where that history exists. The controller compares their complete startup
    and mature programs on fresh evidence.

    Every column evaluation and joint fit spends candidate-row units. Columns
    are measured only for an affordable fit, leaving enough work to finish it.
    Cached values also serve later fits and residual comparisons.
    ``greedy`` preserves the previous longest-history forward-only search.
    """
    if strategy not in {"beam", "greedy"}:
        raise ValueError("sparse search strategy must be beam or greedy")
    if not observations or all(row.target in (0.0, 1.0) for row in observations):
        return []
    from .models import SearchBudget
    concept_limit = budget.remaining // 3
    concepts = []

    def compose_calls(baseline_program=None):
        concepts_budget = SearchBudget(min(concept_limit, budget.remaining))
        atoms = list(additional_terms)
        if select_additional_terms is not None:
            # Selection must leave enough work to turn a useful call into an
            # executable fit. Reuse the raw fit as its comparison coordinate.
            fit_reserve = (len(observations[0].features) + 3) * len(observations)
            selection_budget = SearchBudget(max(0, concepts_budget.remaining - fit_reserve))
            atoms.extend(select_additional_terms(selection_budget, baseline_program))
            concepts_budget.spend(selection_budget.spent)
        result = _concept_candidates(observations, atoms, concepts_budget)
        budget.spend(concepts_budget.spent)
        return result

    if strategy == "greedy":
        results = _greedy_candidates(observations, budget=budget, max_history=max_history, max_terms=max_terms)
        concepts = compose_calls()
        return concepts + results
    rows = list(observations)
    targets = [row.target for row in rows]
    current, _ = _terms(rows, 0)
    current = [*current, *[Program("mul", children=(term, term)) for term in current],
               *[Program("mul", children=(a, b)) for a, b in combinations(current, 2)]]
    depths = [0] + [depth for depth in range(1, max_history + 1)
                    if sum(len(row.history) >= depth for row in rows) >= 6]
    historical = []
    if len(depths) > 1:
        _, historical = _terms([row for row in rows if len(row.history) >= depths[-1]], depths[-1])
    columns = {}
    states = []
    for depth in depths:
        indices = [index for index, row in enumerate(rows) if len(row.history) >= depth]
        available = [*current, *[term for term in historical if term.history_depth <= depth]]
        states.append({"depth": depth, "indices": indices, "available": available,
                       "fits": {}, "beam": []})

    def measure_columns(state, terms, *, reserve):
        missing = {term: [index for index in state["indices"] if index not in columns.get(term, {})]
                   for term in terms}
        if sum(map(len, missing.values())) + reserve > budget.remaining:
            return False
        for term, indices in missing.items():
            budget.spend(len(indices))
            column = columns.setdefault(term, {})
            column.update((index, term.predict(rows[index])) for index in indices)
        return True

    def fit(state, terms, *, reserve=0):
        key = tuple(sorted(set(terms), key=lambda term: term.describe()))
        if key not in state["fits"]:
            if not measure_columns(state, key, reserve=len(state["indices"]) + reserve):
                return None
            if not budget.spend(len(state["indices"])):
                return None
            state["fits"][key] = _fit_columns(key, state["indices"], columns, targets)
        return state["fits"][key]

    def retain(state, fitted):
        if fitted is None or len(fitted.terms) > max_terms:
            return
        pool = {item.program: item for item in [*state["beam"], fitted]}
        state["beam"] = sorted(pool.values(), key=lambda item: (item.loss, item.program.size))[:2]

    # Fit the existing polynomial dictionary jointly before committing to a
    # residual correlation ordering. The history extension uses its own depth's
    # rows, instead of throwing away data for every shorter-history model.
    extensions = [[None] if state["depth"] == 0 else [
        term for term in historical if term.history_depth == state["depth"]] for state in states]
    # Give each history depth a complete first attempt before a shallower
    # depth consumes its second and third coordinate extensions.
    for position in range(max(map(len, extensions), default=0)):
        for state, alternatives in zip(states, extensions):
            if position >= len(alternatives):
                continue
            extension = alternatives[position]
            terms = [*current, *([extension] if extension is not None else [])]
            # A broad fit may need pruning to the supported expression size.
            # Reserve that fit too, so a completed wide regression can become
            # an executable candidate instead of exhausting its last units.
            fitted = fit(state, terms, reserve=len(state["indices"]) if len(terms) > max_terms else 0)
            if fitted is None and state["depth"] == 0:
                # With less work than the polynomial dictionary needs, fit a
                # smaller current-coordinate model using all acquired rows.
                for width in range(min(max_terms, len(current)), -1, -1):
                    fitted = fit(state, current[:width])
                    if fitted is not None:
                        break
            if fitted is None:
                continue
            if len(fitted.terms) > max_terms:
                reduced = sorted(fitted.terms, key=lambda term: fitted.importance[term], reverse=True)[:max_terms]
                fitted = fit(state, reduced)
            retain(state, fitted)

    # Finish the ordinary coordinate and history hypotheses first. Learned
    # concepts add expressive power when those complete fits leave residuals;
    # they need not take work from an explanation that already fits the data.
    if not any(fitted.loss < 1e-12 for state in states for fitted in state["beam"]):
        fitted = min((item for state in states for item in state["beam"]),
                     key=lambda item: (item.loss, item.program.size), default=None)
        concepts = compose_calls(fitted.program if fitted is not None else None)

    # Two passes let a replacement become the parent of another replacement.
    # Every depth gets a turn; no unbounded subset enumeration is performed.
    for _ in range(2):
        for state in states:
            if not state["beam"] or state["beam"][0].loss < 1e-12:
                continue
            trials = 0
            for parent in tuple(state["beam"]):
                removals = [None] if len(parent.terms) < max_terms else sorted(
                    parent.terms, key=lambda term: parent.importance[term])
                for removed in removals:
                    reduced_terms = [term for term in parent.terms if term != removed]
                    reduced = parent if removed is None else fit(state, reduced_terms)
                    if reduced is None:
                        break
                    retain(state, reduced)
                    scores = []
                    for term in state["available"]:
                        if term in reduced_terms or term == removed:
                            continue
                        if not measure_columns(state, (term,), reserve=len(state["indices"])):
                            continue
                        values = [columns[term][index] for index in state["indices"]]
                        center = mean(values)
                        centered = [value - center for value in values]
                        scale = sum(value * value for value in centered)
                        if scale > 1e-10:
                            score = sum(value * residual for value, residual in zip(centered, reduced.residuals)) ** 2 / scale
                            scores.append((score, term))
                    for _, term in sorted(scores, key=lambda item: item[0], reverse=True)[:2]:
                        retain(state, fit(state, [*reduced_terms, term]))
                        trials += 1
                    if trials >= 8:
                        break
                if trials >= 8:
                    break
    results, seen = [], set()
    by_depth = {state["depth"]: state for state in states}
    for state in states:
        for fitted in state["beam"]:
            support = frozenset(fitted.terms)
            if support in seen:
                continue
            # A support discovered in a longer-history slice can still require
            # less history. Refit on all rows supporting the actual expression.
            required = fitted.program.history_depth
            if required < state["depth"]:
                fitted = fit(by_depth[required], fitted.terms) or fitted
            seen.add(support)
            results.append(Candidate(f"sparse:depth{state['depth']}:{len(results)}:{budget.spent}",
                                     fitted.program, "composition",
                                     f"Jointly refit and revise coordinate support using history depth {state['depth']}"))
    return concepts + results
