import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    un = float(row['u_now']) / 5.0
    up = float(row['u_past']) / 5.0
    h = float(row['H'])
    return gain * (un - up) * math.exp(-h / tau_relax) + k_eq * un

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.5]), bounds=([-20.0, 0.02, -20.0], [20.0, 100.0, 20.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))