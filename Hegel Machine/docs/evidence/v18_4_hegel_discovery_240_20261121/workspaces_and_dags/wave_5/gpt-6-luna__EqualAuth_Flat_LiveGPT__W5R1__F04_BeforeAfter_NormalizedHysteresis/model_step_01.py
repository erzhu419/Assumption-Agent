import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_tau_scale = 0.004

def _features(row):
    u_n = row['u_now'] / _tau_scale
    u_p = row['u_past'] / _tau_scale
    return u_n, u_p, row['H']

def _predict_params(params, row):
    gain, tau_relax, k_eq = params
    u_n, u_p, H = _features(row)
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residuals(p):
        return np.asarray([_predict_params(p, r) for r in train_rows]) - y
    starts = [np.array([1.0, 0.5, 0.0]), np.array([2.0, 1.0, 0.0]), np.array([1.0, 0.1, 0.0])]
    results = [least_squares(residuals, s, bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])) for s in starts]
    _params = min(results, key=lambda res: np.sum(res.fun ** 2)).x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_params(_params, row))