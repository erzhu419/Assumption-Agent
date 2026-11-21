import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    scale = 0.004
    u_n = row['u_now'] / scale
    u_p = row['u_past'] / scale
    h = row['H']
    return gain * (u_n - u_p) * math.exp(-h / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    best = None
    for init in ([ -1.0, 0.5, 0.4], [-1.0, 1.0, 0.0], [1.0, 0.5, 0.4]):
        res = least_squares(residual, init, bounds=([-100.0, 1e-5, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _params))