import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def model(p, r):
        gain, tau_relax, k_eq = p
        u_n = float(r['u_now'])
        u_p = float(r['u_past'])
        h = float(r['H'])
        return gain * (u_n - u_p) * math.exp(-h / max(1e-8, tau_relax)) + k_eq * u_n
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.array([model(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.0]), bounds=([-100.0, 1e-4, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    gain, tau_relax, k_eq = _PARAMS
    u_n = float(row['u_now'])
    u_p = float(row['u_past'])
    h = float(row['H'])
    return float(gain * (u_n - u_p) * math.exp(-h / max(1e-8, tau_relax)) + k_eq * u_n)
