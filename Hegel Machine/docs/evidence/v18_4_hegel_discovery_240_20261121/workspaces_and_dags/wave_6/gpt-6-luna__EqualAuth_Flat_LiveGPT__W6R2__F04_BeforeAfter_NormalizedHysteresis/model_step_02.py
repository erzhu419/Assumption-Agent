import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.004
_PARAMS = None

def _predict_with(row, params):
    gain, tau_relax, k_eq = params
    u_n = float(row['u_now']) / _TAU_SCALE
    u_p = float(row['u_past']) / _TAU_SCALE
    h = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-h / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.4, 0.0]), bounds=([-100.0, 1e-6, -100.0], [100.0, 10.0, 100.0]), max_nfev=20000)
    _PARAMS = result.x
    return result.x

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_with(row, _PARAMS))