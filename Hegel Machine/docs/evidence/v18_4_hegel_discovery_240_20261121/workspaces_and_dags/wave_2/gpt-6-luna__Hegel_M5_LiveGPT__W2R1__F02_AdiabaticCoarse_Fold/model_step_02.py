import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        E, eta = p
        pred = (u / E) * (-np.expm1(-E * H / eta))
        return pred - y

    result = least_squares(residual, x0=np.array([0.55, 0.15]), bounds=([1e-8, 1e-8], [100.0, 100.0]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    E, eta = _PARAMS
    return float((row['u_now'] / E) * (-math.expm1(-E * row['H'] / eta)))