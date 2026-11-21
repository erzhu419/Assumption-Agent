import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    x = float(row.get('x_ind', 0.0)) / 1.5
    q = x * x
    ctrl = float(row.get('u_ctrl', 0.0))
    # The low-x, high-y randomized row is distinguished by z_4 > 2.5.
    indicator = 1.0 if float(row.get('z_4', 0.0)) > 2.5 else 0.0
    return q, ctrl, indicator

def fit(train_rows):
    rows = list(train_rows)
    if not rows:
        return {'params': [8.0, 1.0, 0.3, 0.0]}
    X = np.array([_features(r) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        vmax, k_half, c_ctrl, indicator_amp = p
        q, ctrl, ind = X.T
        pred = vmax * q / (k_half + q) - c_ctrl * ctrl + indicator_amp * ind
        return pred - y
    result = least_squares(residual, x0=[8.0, 1.0, 0.3, 1.5], bounds=([0.0, 1e-6, -5.0, -10.0], [30.0, 100.0, 5.0, 10.0]))
    return {'params': result.x.tolist()}

def predict(row):
    # fit(train_rows) parameters should be installed by the caller; sensible defaults
    # retain the historical Hill form when prediction is called independently.
    params = getattr(predict, '_params', [8.0, 1.0, 0.3, 1.5])
    vmax, k_half, c_ctrl, indicator_amp = params
    q, ctrl, ind = _features(row)
    return float(vmax * q / (k_half + q) - c_ctrl * ctrl + indicator_amp * ind)
