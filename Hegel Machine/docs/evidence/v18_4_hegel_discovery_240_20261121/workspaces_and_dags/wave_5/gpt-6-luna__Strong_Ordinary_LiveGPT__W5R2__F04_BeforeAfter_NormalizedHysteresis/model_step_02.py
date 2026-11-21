import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    scale = 0.004
    un = float(row['u_now']) / scale
    up = float(row['u_past']) / scale
    return un, up, float(row['H'])

def _model(p, row):
    gain, tau_relax, k_eq = p
    un, up, h = _features(row)
    return gain * (un - up) * math.exp(-h / tau_relax) + k_eq * un

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(p, r) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([2.0, 0.55, 0.6]), bounds=([-100.0, 1e-6, -100.0], [100.0, 100.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(_PARAMS, row))