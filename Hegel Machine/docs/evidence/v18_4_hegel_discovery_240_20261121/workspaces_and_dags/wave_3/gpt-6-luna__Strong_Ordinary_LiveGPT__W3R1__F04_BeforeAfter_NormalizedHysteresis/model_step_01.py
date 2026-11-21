import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 0.02

def _model(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / TAU_SCALE
    u_p = float(row['u_past']) / TAU_SCALE
    H = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.asarray([_model(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=[1.0, 0.5, 0.0], bounds=([-100.0, 1e-5, -100.0], [100.0, 100.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))