import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 5.0
_params = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / _TAU_SCALE
    u_p = float(row['u_past']) / _TAU_SCALE
    h = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-h / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.0]), bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))