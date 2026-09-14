"""Reuse fitting work on paid rows while keeping each executable explanation.

Equal sampled columns share numerical work only. Their original terms still
produce separate candidates, and new observations can distinguish their laws.
"""

from math import isfinite
from statistics import mean

from .continuous_state import _sum_terms
from .evaluation import initial_prediction, point_loss
from .language import EvaluationContext, MissingHistory
from .models import Candidate, Observation
from .synthesis import _solve


_MISSING = object()


class SharedTransferFitter:
    """One research session's columns, fits and checks, separated by view."""

    def __init__(self):
        self._views = {}
        self._stats = dict.fromkeys(("fit_requests", "column_evaluations", "column_reuses",
            "linear_solves", "linear_solve_rows", "linear_solve_reuses",
            "normalized_columns", "normalized_column_reuses", "normalized_rows",
            "prediction_evaluations", "prediction_reuses", "loss_row_evaluations",
            "loss_row_reuses", "primitive_steps", "call_evaluations", "call_cache_hits",
            "view_resets"), 0)

    def stats(self):
        return dict(self._stats)

    def _rows(self, view, rows):
        state = self._views.get(view)
        if state is not None:
            common = min(len(rows), len(state["rows"]))
            if any(left != right for left, right in zip(rows[:common], state["rows"][:common])):
                state = None
                self._stats["view_resets"] += 1
        if state is None:
            state = {"rows": [], "contexts": [], "columns": {}, "normalized": {},
                     "solutions": [], "losses": {}, "means": {}}
            self._views[view] = state
        for row in rows[len(state["rows"]):]:
            snapshot = Observation(dict(row.features), row.target,
                                   tuple(dict(past) for past in row.history))
            state["rows"].append(snapshot)
            state["contexts"].append(EvaluationContext(snapshot))
        return state

    def _evaluate(self, state, program, index):
        context = state["contexts"][index]
        before = context.primitive_steps, context.call_evaluations, context.cache_hits
        try:
            value = program.predict(state["rows"][index], context=context)
        except MissingHistory:
            value = _MISSING
        finally:
            for key, old, current in zip(("primitive_steps", "call_evaluations", "call_cache_hits"),
                    before, (context.primitive_steps, context.call_evaluations, context.cache_hits)):
                self._stats[key] += current - old
        return value

    def _column(self, state, program, count, budget):
        values = state["columns"].setdefault(program, [])
        self._stats["column_reuses"] += min(count, len(values))
        for index in range(len(values), count):
            if not budget.spend(1):
                return None
            self._stats["column_evaluations"] += 1
            values.append(self._evaluate(state, program, index))
        result = values[:count]
        if any(value is _MISSING for value in result):
            raise MissingHistory("transfer term requires unavailable recorded history")
        return result

    def fit(self, terms, rows, budget, kind, *, family="T02", q=.5, view=()):
        """Fit each original term list using unchanged centered ridge arithmetic.

        Charge uncached term-row evaluations, one row pass for a new linear
        solve, and the executable loss check. A reused solve reconstructs the
        caller's own coefficients; it does not identify the response programs.
        """
        self._stats["fit_requests"] += 1
        request = self._stats["fit_requests"]
        if not rows:
            return None
        terms = list(dict.fromkeys(terms))
        state = self._rows(view, rows)
        count = len(rows)
        centers, scales, normalized = [], [], []
        for term in terms:
            column = self._column(state, term, count, budget)
            if column is None:
                return None
            key = term, count
            if key not in state["normalized"]:
                center = sum(column) / count
                scale = max(abs(value - center) for value in column) or 1.0
                state["normalized"][key] = center, scale, tuple((value - center) / scale for value in column)
                self._stats["normalized_columns"] += 1
                self._stats["normalized_rows"] += count
            else:
                self._stats["normalized_column_reuses"] += 1
            center, scale, values = state["normalized"][key]
            centers.append(center)
            scales.append(scale)
            normalized.append(values)
        normalized = tuple(normalized)
        targets = tuple(row.target for row in state["rows"][:count])
        matched = next((solution for columns, observed, solution in state["solutions"]
                        if columns == normalized and observed == targets), None)
        if matched is None:
            if not budget.spend(count):
                return None
            design = [[1.0, *(column[index] for column in normalized)] for index in range(count)]
            width = len(terms) + 1
            gram = [[sum(row[i] * row[j] for row in design) for j in range(width)]
                    for i in range(width)]
            rhs = [sum(row[i] * target for row, target in zip(design, targets)) for i in range(width)]
            for index in range(1, width):
                gram[index][index] += 1e-9
            matched = _solve(gram, rhs)
            state["solutions"].append((normalized, targets, matched))
            self._stats["linear_solves"] += 1
            self._stats["linear_solve_rows"] += count
        else:
            self._stats["linear_solve_reuses"] += 1
        slopes = [value / scale for value, scale in zip(matched[1:], scales)]
        intercept = matched[0] - sum(value * center for value, center in zip(slopes, centers))
        program = _sum_terms(terms, slopes, intercept).simplified()
        measured_loss = self.loss(program, rows, budget, initial_prediction(rows, q), q, view=view)
        if measured_loss is None:
            return None
        return {"candidate": Candidate(f"shared_transfer:{kind}:{request}", program,
                                       "unified_research", kind),
                "kind": kind, "family_id": family, "training_loss": measured_loss}

    def loss(self, program, rows, budget, fallback, q, *, view=()):
        """Check an actual executable once per row, fallback and risk objective."""
        if not rows:
            raise ValueError("cannot evaluate a candidate without observations")
        state = self._rows(view, rows)
        count, key = len(rows), (program, fallback, q)
        losses = state["losses"].setdefault(key, [])
        self._stats["loss_row_reuses"] += min(count, len(losses))
        values = state["columns"].setdefault(program, [])
        for index in range(len(losses), count):
            if not budget.spend(1):
                return None
            self._stats["loss_row_evaluations"] += 1
            if index < len(values):
                value = values[index]
                self._stats["prediction_reuses"] += 1
            else:
                value = self._evaluate(state, program, index)
                values.append(value)
                self._stats["prediction_evaluations"] += 1
            prediction = fallback if value is _MISSING else float(value)
            if not isfinite(prediction):
                raise ValueError("candidate produced a non-finite prediction")
            losses.append(point_loss(prediction, state["rows"][index].target, q))
        mean_key = key, count
        if mean_key not in state["means"]:
            state["means"][mean_key] = mean(losses[:count])
        return state["means"][mean_key]
