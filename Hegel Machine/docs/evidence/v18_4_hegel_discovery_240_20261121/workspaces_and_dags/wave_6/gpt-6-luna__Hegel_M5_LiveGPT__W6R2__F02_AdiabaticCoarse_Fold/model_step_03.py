import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, E, eta):
    u = float(row['u_now'])
    H = float(row['H'])
    return (u / E) * (1.0 - math.exp(-E * H / eta))

def fit(train_rows):
    global _PARAMS
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(log_params):
        E, eta = np.exp(log_params)
        return np.array([_value(r, E, eta) for r in train_rows]) - ys
    result = least_squares(residual, np.log([0.5, 0.5]))
    _PARAMS = tuple(np.exp(result.x))
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, *_PARAMS))