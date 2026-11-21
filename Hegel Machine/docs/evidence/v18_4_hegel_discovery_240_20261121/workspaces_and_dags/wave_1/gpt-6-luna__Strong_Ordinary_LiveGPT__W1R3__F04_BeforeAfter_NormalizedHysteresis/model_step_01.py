import math
import numpy as np
from scipy.optimize import least_squares

_tau_scale = 0.004
_params = None

def _predict_with(p, row):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / _tau_scale
    u_p = float(row['u_past']) / _tau_scale
    H = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    def residual(p):
        return np.asarray([_predict_with(p, r) - float(r['y']) for r in train_rows])
    starts = ([1.0, 0.01, 0.0], [0.5, 0.1, 0.0], [2.0, 1.0, 0.1])
    sols = [least_squares(residual, x0=s, bounds=([-100.0, 1e-8, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000) for s in starts]
    _params = min(sols, key=lambda sol: float(np.dot(sol.fun, sol.fun))).x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_with(_params, row))