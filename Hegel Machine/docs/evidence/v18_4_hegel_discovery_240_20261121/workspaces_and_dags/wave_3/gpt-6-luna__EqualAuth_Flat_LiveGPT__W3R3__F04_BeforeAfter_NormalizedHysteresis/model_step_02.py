import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.02
_params = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    u_n = row['u_now'] / _TAU_SCALE
    u_p = row['u_past'] / _TAU_SCALE
    return gain * (u_n - u_p) * math.exp(-row['H'] / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    def residuals(p):
        return np.array([_value(r, p) - r['y'] for r in train_rows], dtype=float)
    result = least_squares(residuals, x0=np.array([1.0, 0.02, 0.0]), bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))