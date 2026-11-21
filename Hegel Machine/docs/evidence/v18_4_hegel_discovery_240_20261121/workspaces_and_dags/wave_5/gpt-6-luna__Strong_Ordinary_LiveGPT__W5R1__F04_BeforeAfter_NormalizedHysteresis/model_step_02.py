import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, params):
    gain, tau_relax, k_eq = params
    scale = 0.004
    u_n = float(row['u_now']) / scale
    u_p = float(row['u_past']) / scale
    h = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-h / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_predict_params(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.0]), bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_params(row, _params))