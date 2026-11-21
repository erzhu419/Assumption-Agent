import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _pred(row, E, eta):
    H = float(row['H'])
    u = float(row['u_now'])
    z = E * H / eta
    return (u / E) * (-math.expm1(-z))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(logp):
        E, eta = np.exp(logp)
        return np.array([_pred(r, E, eta) for r in rows]) - y
    result = least_squares(residual, np.log([0.45, 0.35]), max_nfev=10000)
    _PARAMS = tuple(np.exp(result.x))
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_pred(row, *_PARAMS))