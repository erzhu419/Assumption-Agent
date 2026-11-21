import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, E, eta):
    u = float(row['u_now'])
    H = float(row['H'])
    z = E * H / eta
    return (u / E) * (-math.expm1(-z))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(log_params):
        E, eta = np.exp(log_params)
        return np.asarray([_model(r, E, eta) for r in train_rows]) - y
    result = least_squares(residual, np.log([0.6, 0.5]))
    _params = tuple(np.exp(result.x))
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, *_params))