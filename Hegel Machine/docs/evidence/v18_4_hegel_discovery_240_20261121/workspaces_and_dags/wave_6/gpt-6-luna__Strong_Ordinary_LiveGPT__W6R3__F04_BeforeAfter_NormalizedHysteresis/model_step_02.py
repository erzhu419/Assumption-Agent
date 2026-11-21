import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 0.02

def _value(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / TAU_SCALE
    u_p = float(row['u_past']) / TAU_SCALE
    return gain * (u_n - u_p) * math.exp(-float(row['H']) / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.2, 0.01]), bounds=([-100.0, 1e-5, -100.0], [100.0, 100.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _PARAMS))