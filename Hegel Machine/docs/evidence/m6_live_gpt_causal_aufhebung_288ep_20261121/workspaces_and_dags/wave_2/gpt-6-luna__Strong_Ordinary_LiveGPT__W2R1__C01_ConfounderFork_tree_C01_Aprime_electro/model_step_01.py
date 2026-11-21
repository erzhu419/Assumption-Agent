import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    x = float(row['x_ind']) / 1.5
    q = x * x
    u = float(row['u_ctrl'])
    return q, u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    qs, us, ys = [], [], []
    for row in rows:
        q, u = _features(row)
        qs.append(q)
        us.append(u)
        ys.append(float(row['y']))
    qs = np.asarray(qs, dtype=float)
    us = np.asarray(us, dtype=float)
    ys = np.asarray(ys, dtype=float)
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * qs / (k_half + qs) - c_ctrl * us - ys
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.2]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, k_half, c_ctrl = _params
    q, u = _features(row)
    return float(vmax * q / (k_half + q) - c_ctrl * u)
