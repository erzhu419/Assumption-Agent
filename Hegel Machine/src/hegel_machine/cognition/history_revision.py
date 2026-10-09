"""One-coordinate quadratic history revision using only paid trajectories."""

from dataclasses import dataclass
from math import ceil

import numpy as np


def _value(row, name):
    return row[name] if isinstance(row, dict) else getattr(row, name)


def _terms(lags):
    # Adding a coordinate appends terms without reordering the existing basis.
    terms = [()]
    for coordinate in range(lags + 1):
        terms.append((coordinate,))
        terms.extend((previous, coordinate) for previous in range(coordinate + 1))
    return tuple(terms)


def _basis(row, lags, terms):
    history = _value(row, "history")
    controls = [float(_value(row, "features")["u"])]
    controls.extend(float(history[-lag]["u"]) if len(history) >= lag else 0.0
                    for lag in range(1, lags + 1))
    return np.asarray([np.prod([controls[index] for index in term]) for term in terms])


def _fit(design, targets):
    coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    mse = float(np.mean((design @ coefficients - targets) ** 2))
    return coefficients, int(rank), mse


@dataclass
class HistoryRevisionProposal:
    selected_lags: int
    coefficients: tuple[float, ...]
    resultdict: dict

    def predict(self, row):
        return float(_basis(row, self.selected_lags, _terms(self.selected_lags))
                     @ np.asarray(self.coefficients))

    def to_code(self):
        """Export the selected predictor with its fitted feature order intact."""
        return f'''import numpy as np

_LAGS = {self.selected_lags!r}
_TERMS = {_terms(self.selected_lags)!r}
_COEF = np.asarray({self.coefficients!r}, dtype=float)

def _value(row, name):
    return row[name] if isinstance(row, dict) else getattr(row, name)

def _basis(row):
    history = _value(row, "history")
    controls = [float(_value(row, "features")["u"])]
    controls.extend(float(history[-lag]["u"]) if len(history) >= lag else 0.0
                    for lag in range(1, _LAGS + 1))
    return np.asarray([np.prod([controls[index] for index in term]) for term in _TERMS])

def fit(observations):
    global _COEF
    design = np.asarray([_basis(row) for row in observations])
    targets = np.asarray([float(_value(row, "target")) for row in observations])
    _COEF = np.linalg.lstsq(design, targets, rcond=None)[0]

def predict(row):
    return float(_basis(row) @ _COEF)
'''


def revise_history(rows, base_lags=2, holdout_fraction=1 / 3):
    """Compare a complete quadratic basis with one extra history coordinate.

    Input rows must preserve complete, contiguous trajectories, each starting
    with empty history. The final fraction of whole trajectories is held out
    for selection. Only the selected model is then refitted on all paid rows.
    Rank-deficient training or holdout designs do not support the extension.
    """
    rows = list(rows)
    if not rows or base_lags < 0 or not 0 < holdout_fraction < 1:
        raise ValueError("provide paid rows, nonnegative base_lags, and a holdout fraction in (0, 1)")
    starts = []
    for index, row in enumerate(rows):
        history = _value(row, "history")
        if not history:
            starts.append(index)
        if not starts or len(history) != index - starts[-1]:
            raise ValueError("rows must preserve complete trajectories from each reset")

    count = len(starts)
    held_count = min(count - 1, ceil(count * holdout_fraction))
    split = starts[count - held_count] if held_count else len(rows)
    targets = np.asarray([float(_value(row, "target")) for row in rows])
    designs, candidates = {}, {}
    for label, lags in (("base", base_lags), ("revision", base_lags + 1)):
        design = np.asarray([_basis(row, lags, _terms(lags)) for row in rows])
        designs[label] = design
        coefficients, rank, train_mse = _fit(design[:split], targets[:split])
        held_design = design[split:]
        candidates[label] = {
            "lags": lags, "feature_count": design.shape[1], "training_rank": rank,
            "holdout_rank": int(np.linalg.matrix_rank(held_design)) if held_count else 0,
            "train_mse": train_mse,
            "heldout_mse": float(np.mean((held_design @ coefficients - targets[split:]) ** 2))
            if held_count else None,
        }

    revision = candidates["revision"]
    supported = (held_count > 0 and revision["training_rank"] == revision["feature_count"]
                 and revision["holdout_rank"] == revision["feature_count"])
    selected = "base"
    if not supported:
        status = "unsupported"
        reason = "The separate training and holdout trajectories do not identify all added history terms."
    elif candidates["base"]["heldout_mse"] <= 1e-10:
        status = "base_retained"
        reason = "The refitted base already meets the prediction tolerance on held-out trajectories."
    elif revision["heldout_mse"] < candidates["base"]["heldout_mse"]:
        selected, status = "revision", "revised"
        reason = "The added history coordinate reduced error on held-out whole trajectories."
    else:
        status = "base_retained"
        reason = "The added history coordinate did not improve held-out trajectory error."

    coefficients, rank, mse = _fit(designs[selected], targets)
    selected_lags = candidates[selected]["lags"]
    result = {
        "status": status, "reason": reason, "selected_lags": selected_lags,
        "split": {"train_trajectories": list(range(count - held_count)),
                  "holdout_trajectories": list(range(count - held_count, count)),
                  "train_rows": split, "holdout_rows": len(rows) - split,
                  "holdout_start_row": split},
        **candidates,
        "selected_refit": {"rows": len(rows), "rank": rank, "train_mse": mse},
    }
    return HistoryRevisionProposal(selected_lags, tuple(float(x) for x in coefficients), result)
