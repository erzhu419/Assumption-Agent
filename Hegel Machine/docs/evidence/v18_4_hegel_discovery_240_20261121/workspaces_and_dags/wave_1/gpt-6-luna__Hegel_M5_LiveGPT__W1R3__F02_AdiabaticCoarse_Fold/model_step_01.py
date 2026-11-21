import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(logp):
        E, eta = np.exp(logp)
        pred = (u / E) * (1.0 - np.exp(-E * H / eta))
        return pred - y
    result = least_squares(residual, np.log([0.53, 0.15]), bounds=(np.log([1e-4, 1e-5]), np.log([100.0, 1000.0])))
    _params = tuple(np.exp(result.x))
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    E, eta = _params
    return (row['u_now'] / E) * (1.0 - math.exp(-E * row['H'] / eta))