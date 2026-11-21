import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u_n = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    base = (1.0 - s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return u_n, s, base

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows])
    feats = [_features(r) for r in train_rows]
    def residual(p):
        vmax, rho, c_loc = p
        pred = np.asarray([vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u) for u, s, _ in feats])
        return pred - ys
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.1]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, rho, c_loc = _PARAMS
    u_n = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * math.sqrt(u_n))