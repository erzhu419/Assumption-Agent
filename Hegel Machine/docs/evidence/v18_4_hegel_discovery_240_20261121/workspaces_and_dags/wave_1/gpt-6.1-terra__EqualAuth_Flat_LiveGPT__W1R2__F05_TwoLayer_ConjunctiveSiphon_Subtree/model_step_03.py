import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_kind = None

def _value(row, p, kind):
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    q = s / (1.0 + s)
    base = p[0] * u / (p[1] + u) * (1.0 - p[2] * q)
    if kind == 1:
        base += p[3] * u / (p[1] + u) * q * q
    elif kind == 2:
        base += p[3] * u / (p[1] + u) * max(0.0, s - 1.0) / (1.0 + s)
    elif kind == 3:
        base += p[3] * max(0.0, u - 4.0) / (4.8 + u) * max(0.0, s - 1.0) / (1.0 + s)
    return base

def _fit_kind(rows, kind):
    n = 3 if kind == 0 else 4
    start = [7.5, 4.8, 0.70] + ([0.0] if n == 4 else [])
    lower = [0.0, 0.1, 0.0] + ([-10.0] if n == 4 else [])
    upper = [30.0, 30.0, 1.5] + ([10.0] if n == 4 else [])
    def residual(p):
        return np.array([_value(r, p, kind) - float(r['y']) for r in rows])
    return least_squares(residual, start, bounds=(lower, upper), max_nfev=4000).x

def fit(train_rows):
    global _params, _kind
    rows = list(train_rows)
    scores = []
    for kind in range(4):
        errors = []
        for i in range(len(rows)):
            p = _fit_kind(rows[:i] + rows[i+1:], kind)
            errors.append((_value(rows[i], p, kind) - float(rows[i]['y'])) ** 2)
        scores.append(sum(errors) / len(errors))
    _kind = int(np.argmin(scores))
    _params = _fit_kind(rows, _kind)

def predict(row) -> float:
    return float(_value(row, _params, _kind))
