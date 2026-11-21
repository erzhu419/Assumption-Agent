import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _features(row):
    return max(float(row['u_now']), 0.0), max(float(row['s_grad']), 0.0)

def _value(p, row, family):
    u, s = _features(row)
    km, vmax, sink, local = p[:4]
    base = u / (km + u)
    g = s / (1.0 + s)
    result = vmax * base - sink * base * g + local * math.sqrt(u)
    if family == 1:
        result -= p[4] * u * g
    elif family == 2:
        result -= p[4] * base * g * g
    return result

def _fit_family(rows, family):
    n = 4 + (family != 0)
    lower = [0.001, 0.0, 0.0, -100.0] + ([-100.0] if n == 5 else [])
    upper = [10.0, 1000.0, 1000.0, 100.0] + ([1000.0] if n == 5 else [])
    def residual(p):
        return np.array([_value(p, r, family) - float(r['y']) for r in rows])
    best = None
    for km in (0.05, 0.3, 1.2, 3.0):
        p0 = [km, 30.0, 20.0, 2.0] + ([1.0] if n == 5 else [])
        res = least_squares(residual, p0, bounds=(lower, upper), max_nfev=4000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    return best

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    candidates = []
    for family in (0, 1, 2):
        score, params = _fit_family(rows, family)
        errors = []
        for i in range(len(rows)):
            _, p = _fit_family(rows[:i] + rows[i+1:], family)
            errors.append(_value(p, rows[i], family) - float(rows[i]['y']))
        cv = float(np.mean(np.square(errors)))
        candidates.append((cv, family, params, score))
    _, _model, _params, _ = min(candidates, key=lambda item: item[0])

def predict(row):
    return float(_value(_params, row, _model))
