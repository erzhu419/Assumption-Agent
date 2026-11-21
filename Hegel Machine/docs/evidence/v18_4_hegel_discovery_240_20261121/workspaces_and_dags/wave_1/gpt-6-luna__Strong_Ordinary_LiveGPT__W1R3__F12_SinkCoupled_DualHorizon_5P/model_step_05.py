import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    af, tf, ass, ts, rho = p
    h = float(row['H'])
    u = float(row['u_now'])
    s = float(row['s_grad'])
    temporal = af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts))
    sink = 1.0 - rho * s / (1.0 + s)
    return u * temporal * sink

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    x0 = np.array([1.0, 1.0, 1.0, 10.0, 0.5], dtype=float)
    lb = np.array([0.0, 1e-4, 0.0, 1e-4, 0.0])
    ub = np.array([100.0, 100.0, 100.0, 500.0, 1.0])
    def residual(p):
        return np.array([_model(r, p) - float(r['y']) for r in rows])
    best = None
    for guess in (x0, [2, .3, 2, 10, .7], [2, 2, 3, 30, .3], [5, .1, 5, 20, .9]):
        q = least_squares(residual, np.asarray(guess, dtype=float), bounds=(lb, ub), max_nfev=30000)
        if best is None or np.dot(q.fun, q.fun) < np.dot(best.fun, best.fun):
            best = q
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit must be called before predict')
    return float(_model(row, _PARAMS))