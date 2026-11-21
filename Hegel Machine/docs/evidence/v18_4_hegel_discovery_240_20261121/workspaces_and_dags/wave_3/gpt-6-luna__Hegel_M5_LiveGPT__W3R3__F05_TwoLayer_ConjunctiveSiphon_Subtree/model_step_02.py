import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    un = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    base = (1.0 - _PARAMS[1] * s / (1.0 + s)) * (un / (1.2 + un))
    return base, math.sqrt(max(un, 0.0))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    if not rows:
        raise ValueError('train_rows must not be empty')
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        vmax, rho, c_loc = p
        pred = []
        for r in rows:
            un = float(r['u_now']) / 4.0
            s = float(r['s_grad'])
            base = (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
            pred.append(vmax * base + c_loc * math.sqrt(max(un, 0.0)))
        return np.asarray(pred) - y
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.1]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]), max_nfev=20000)
    _PARAMS = result.x.copy()
    return tuple(float(v) for v in _PARAMS)

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('call fit(train_rows) before predict(row)')
    un = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return float(_PARAMS[0] * (1.0 - _PARAMS[1] * s / (1.0 + s)) * (un / (1.2 + un)) + _PARAMS[2] * math.sqrt(max(un, 0.0)))