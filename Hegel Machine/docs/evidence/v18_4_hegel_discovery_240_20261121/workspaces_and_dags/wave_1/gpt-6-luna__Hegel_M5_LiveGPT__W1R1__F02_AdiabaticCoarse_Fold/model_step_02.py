import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, E, eta):
    H = float(row['H'])
    u = float(row['u_now'])
    return (u / E) * (-math.expm1(-E * H / eta))

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(log_params):
        E, eta = np.exp(log_params)
        return np.asarray([_value(r, E, eta) for r in train_rows]) - ys
    result = least_squares(residual, np.log([0.48, 100.0]), max_nfev=10000)
    _params = tuple(np.exp(result.x))
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first.')
    return float(_value(row, *_params))