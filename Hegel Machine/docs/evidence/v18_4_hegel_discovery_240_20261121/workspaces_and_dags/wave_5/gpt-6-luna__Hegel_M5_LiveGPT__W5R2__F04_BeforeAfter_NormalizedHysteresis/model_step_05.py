import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    scale = 0.004
    u_n = row['u_now'] / scale
    u_p = row['u_past'] / scale
    tau_relax = max(float(tau_relax), 1e-9)
    return gain * (u_n - u_p) * math.exp(-row['H'] / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5]), bounds=([0.0, 1e-6, -100.0], [100.0, 100.0, 100.0]))
    _params = result.x
    return result.x

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_value(row, _params))