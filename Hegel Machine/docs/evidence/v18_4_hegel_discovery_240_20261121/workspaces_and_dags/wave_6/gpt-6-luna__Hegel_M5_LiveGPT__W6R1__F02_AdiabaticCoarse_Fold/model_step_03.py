import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    H = np.asarray([float(r['H']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)

    def residual(log_params):
        E, eta = np.exp(log_params)
        pred = (u / E) * (-np.expm1(-E * H / eta))
        return pred - y

    result = least_squares(residual, np.log([0.5, 1.0]))
    _PARAMS = tuple(np.exp(result.x))
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    E, eta = _PARAMS
    u = float(row['u_now'])
    H = float(row['H'])
    return float((u / E) * (-math.expm1(-E * H / eta)))
