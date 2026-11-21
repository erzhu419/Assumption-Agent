import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    x = float(row['x_ind'])
    u = float(row.get('u_ctrl', 0.0))
    t = (x / 1.5) ** 2
    branch = 1.0 if float(row['z_1']) < float(row['z_2']) else 0.0
    return t, u, branch

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    if not rows:
        raise ValueError('train_rows must not be empty')
    def residual(p):
        vmax, k_half, c_ctrl, branch_gain = p
        out = []
        for r in rows:
            t, u, branch = _features(r)
            pred = vmax * t / (k_half + t) - c_ctrl * u + branch_gain * branch
            out.append(pred - float(r['y']))
        return np.asarray(out, dtype=float)
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.5, 0.0]), bounds=([0.0, 1e-8, -10.0, -20.0], [100.0, 100.0, 10.0, 20.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    vmax, k_half, c_ctrl, branch_gain = _PARAMS
    t, u, branch = _features(row)
    return float(vmax * t / (k_half + t) - c_ctrl * u + branch_gain * branch)
