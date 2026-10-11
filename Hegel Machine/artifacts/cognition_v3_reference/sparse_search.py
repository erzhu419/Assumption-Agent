"""Residual-guided composition of observed coordinates and historical terms."""

from itertools import combinations
from statistics import mean

from .language import Program
from .models import Candidate
from .synthesis import _fit_linear, _terms


def sparse_candidates(observations, *, budget, max_history=4, max_terms=4):
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
