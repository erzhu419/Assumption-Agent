import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_model = None

def _value(p, row, model):
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    a, b, c, d, e = p
    base = a * u / (b + u)
    sink = s / (e + s)
    if model == 0:
        return base * (1.0 - d * sink) + c * math.sqrt(u)
    return (base + c * math.sqrt(u)) * (1.0 - d * sink)

def _fit(rows, model):
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(p, r, model) for r in rows]) - y
    bounds = ([0.0, 0.001, 0.0, 0.0, 0.01], [200.0, 10.0, 100.0, 1.0, 20.0])
    starts = [[20, 1.2, 10, 0.7, 1], [10, 0.15, 5, 0.5, 0.5], [50, 2, 10, 0.8, 2]]
    results = [least_squares(residual, x, bounds=bounds, max_nfev=3000) for x in starts]
    return min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def fit(train_rows):
    global _params, _model
    rows = list(train_rows)
    scores = []
    for model in (0, 1):
        errors = []
        for i, row in enumerate(rows):
            p = _fit(rows[:i] + rows[i + 1:], model)
            errors.append((_value(p, row, model) - float(row['y'])) ** 2)
        scores.append(sum(errors) / len(errors))
    _model = int(np.argmin(scores))
    _params = _fit(rows, _model)

def predict(row) -> float:
    return float(_value(_params, row, _model))
