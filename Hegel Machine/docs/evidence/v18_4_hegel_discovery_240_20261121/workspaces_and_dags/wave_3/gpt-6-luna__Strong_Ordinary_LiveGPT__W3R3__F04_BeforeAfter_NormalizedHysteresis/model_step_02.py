import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    un = float(row['u_now']) / 0.02
    up = float(row['u_past']) / 0.02
    h = float(row['H'])
    return gain * (un - up) * math.exp(-h / tau_relax) + k_eq * un

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([2.0, 1.0, 0.4]), bounds=([-100.0, 1e-6, -100.0], [100.0, 1000.0, 100.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _params))