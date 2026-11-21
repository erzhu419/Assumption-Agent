import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    tau_scale = 0.004
    u_n = row['u_now'] / tau_scale
    u_p = row['u_past'] / tau_scale
    return gain * (u_n - u_p) * math.exp(-row['H'] / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(r, p) - r['y'] for r in train_rows], dtype=float)
    result = least_squares(residual, x0=np.array([-1.0, 0.5, 0.3]), bounds=([-100.0, 1e-8, -100.0], [100.0, 100.0, 100.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _params))