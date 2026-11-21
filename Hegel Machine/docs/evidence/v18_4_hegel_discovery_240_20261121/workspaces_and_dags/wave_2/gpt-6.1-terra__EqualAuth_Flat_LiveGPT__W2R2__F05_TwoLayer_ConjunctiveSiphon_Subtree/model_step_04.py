import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(p, row, kind):
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    base = p[0] * u / (p[1] + u)
    if kind == 0:
        return base * (1.0 - p[2] * s / (1.0 + s)) + p[3] * math.sqrt(u)
    if kind == 1:
        return base * (1.0 - p[2] * s / (p[4] + s)) + p[3] * math.sqrt(u)
    return base * (1.0 - p[2] * s / (1.0 + s)) + p[3] * math.sqrt(u) - p[4] * (u / (p[1] + u)) * (s / (1.0 + s))**2

def _fit(rows, kind):
    n = 4 if kind == 0 else 5
    lower = [0.0, 0.05, 0.0, -2.0] + ([0.05] if kind == 1 else ([0.0] if kind == 2 else []))
    upper = [30.0, 30.0, 2.0, 3.0] + ([20.0] if kind == 1 else ([20.0] if kind == 2 else []))
    best = None
    for km in (1.2, 5.0):
        start = [4.0, km, 0.7, 0.1] + ([1.0] if n == 5 else [])
        result = least_squares(lambda p: np.array([_value(p, r, kind) - float(r['y']) for r in rows]), start, bounds=(lower, upper), max_nfev=3000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    scores = []
    for kind in (0, 1, 2):
        errors = []
        for i in range(len(rows)):
            p = _fit(rows[:i] + rows[i+1:], kind)
            errors.append(_value(p, rows[i], kind) - float(rows[i]['y']))
        scores.append(float(np.sqrt(np.mean(np.square(errors)))))
    minimum = min(scores)
    _model = next(k for k, score in enumerate(scores) if score <= minimum + 0.005)
    _params = _fit(rows, _model)

def predict(row):
    return float(_value(_params, row, _model))
