import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 0.004

def _features(row):
    un = row['u_now'] / TAU_SCALE
    up = row['u_past'] / TAU_SCALE
    return un, up

def _model(row, params):
    gain, tau_relax, k_eq = params
    un, up = _features(row)
    return gain * (un - up) * math.exp(-row['H'] / tau_relax) + k_eq * un

def fit(train_rows):
    global _PARAMS
    def residual(params):
        return np.asarray([_model(r, params) - r['y'] for r in train_rows], dtype=float)
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5]), bounds=([-100.0, 1e-6, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000)
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_model(row, _PARAMS))