import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    af, ass, tf, ts, k = p
    h = float(row['H'])
    base = float(row['u_now']) * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts)))
    return base + k * float(row['s_grad'])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    lo = np.array([0.0, 0.0, 1e-4, 0.5001, -10.0])
    hi = np.array([20.0, 20.0, 0.5, 1000.0, 10.0])
    p0 = np.array([1.0, 1.0, 0.1, 10.0, 0.0])
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=20000)
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))