import math
import numpy as np
from scipy.optimize import least_squares

_SELECTED = None

def _value(row, p, branch):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = p[0] * (1.0 - p[1] * s / (1.0 + s)) * (u / (1.2 + u))
    if branch == 'none':
        return base
    if branch == 'sqrt_plus':
        return base + p[2] * math.sqrt(max(u, 0.0))
    if branch == 'sqrt_minus':
        return base - p[2] * math.sqrt(max(u, 0.0))
    return base - p[2] * u

def fit(train_rows):
    global _SELECTED
    rows = list(train_rows)
    candidates = [('none', 2), ('sqrt_plus', 3), ('sqrt_minus', 3), ('leak', 3)]
    best = None
    y = np.asarray([float(r['y']) for r in rows])
    for branch, npar in candidates:
        def residual(p):
            return np.asarray([_value(r, p, branch) for r in rows]) - y
        p0 = np.array([3.0, 0.7] if npar == 2 else [3.0, 0.7, 0.1])
        lo = np.array([0.0, 0.0] if npar == 2 else [0.0, 0.0, 0.0])
        hi = np.array([20.0, 1.0] if npar == 2 else [20.0, 1.0, 5.0])
        result = least_squares(residual, p0, bounds=(lo, hi))
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, branch, result.x)
    _SELECTED = (best[1], best[2])
    return _SELECTED

def predict(row):
    if _SELECTED is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    branch, p = _SELECTED
    return float(_value(row, p, branch))