import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.02
_PARAMS = None

def _features(row):
    un = float(row['u_now']) / _TAU_SCALE
    up = float(row['u_past']) / _TAU_SCALE
    H = float(row['H'])
    return un, up, H

def _formula(row, p):
    gain, tau_relax, k_eq = p
    un, up, H = _features(row)
    return gain * (un - up) * math.exp(-H / tau_relax) + k_eq * un

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_formula(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([-0.5, 0.5, 0.5]), bounds=([-100.0, 1e-5, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000)
    _PARAMS = result.x
    return tuple(float(v) for v in _PARAMS)

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_formula(row, _PARAMS))