import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def predict_with(p, row):
        gain, tau_relax, k_eq = p
        un = float(row['u_now'])
        up = float(row['u_past'])
        H = float(row['H'])
        return gain * (un - up) * math.exp(-H / max(1e-8, tau_relax)) + k_eq * un
    def residual(p):
        return [predict_with(p, r) - float(r['y']) for r in rows]
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.5]), bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    gain, tau_relax, k_eq = _PARAMS
    un = float(row['u_now'])
    up = float(row['u_past'])
    H = float(row['H'])
    return float(gain * (un - up) * math.exp(-H / max(1e-8, tau_relax)) + k_eq * un)
