import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    H = np.asarray([r['H'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(logp):
        E, eta = np.exp(logp)
        pred = (u / E) * (-np.expm1(-E * H / eta))
        return pred - y
    result = least_squares(residual, np.log([0.02, 0.1]), max_nfev=20000)
    _PARAMS = tuple(np.exp(result.x))
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    E, eta = _PARAMS
    return float((row['u_now'] / E) * (-math.expm1(-E * row['H'] / eta)))