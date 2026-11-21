import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / 0.02
    u_p = float(row['u_past']) / 0.02
    return gain * (u_n - u_p) * math.exp(-float(row['H']) / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([3.0, 1.0, 0.45]), bounds=([0.0, 1e-5, -100.0], [100.0, 100.0, 100.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))