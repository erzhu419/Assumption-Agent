import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now'])
    u_p = float(row['u_past'])
    H = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.1]), bounds=([-100.0, 1e-6, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000)
    _params = result.x
    return {'gain': float(_params[0]), 'tau_relax': float(_params[1]), 'k_eq': float(_params[2])}

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_params(row, _params))