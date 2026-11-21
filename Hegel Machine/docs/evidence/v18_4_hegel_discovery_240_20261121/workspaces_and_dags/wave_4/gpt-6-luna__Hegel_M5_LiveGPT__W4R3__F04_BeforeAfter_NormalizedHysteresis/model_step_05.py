import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 0.004

def _predict_with(row, params):
    gain, tau_relax, k_eq = params
    u_n = float(row['u_now']) / TAU_SCALE
    u_p = float(row['u_past']) / TAU_SCALE
    H = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([-1.0, 1.0, 0.4]), bounds=([-100.0, 1e-6, -100.0], [100.0, 1000.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_with(row, _PARAMS))