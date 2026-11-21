import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now'])
    u_p = float(row['u_past'])
    H = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    rows = list(train_rows)
    if not rows:
        raise ValueError('train_rows must not be empty')
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows], dtype=float) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.0]), bounds=([-100.0, 1e-6, -100.0], [100.0, 100.0, 100.0]))
    _params = result.x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))