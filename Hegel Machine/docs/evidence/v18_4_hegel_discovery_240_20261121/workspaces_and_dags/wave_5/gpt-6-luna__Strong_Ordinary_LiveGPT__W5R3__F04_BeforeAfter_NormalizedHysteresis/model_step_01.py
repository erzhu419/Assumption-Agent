import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / 0.02
    u_p = float(row['u_past']) / 0.02
    H = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.0]), bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _PARAMS))