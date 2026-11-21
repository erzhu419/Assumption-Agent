import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    x = float(row.get('x_ind', 0.0))
    u = float(row.get('u_ctrl', 0.0))
    # Tree-A indicator candidate: z1 exceeds z2.
    ind = 1.0 if float(row.get('z_1', 0.0)) > float(row.get('z_2', 0.0)) else 0.0
    return x, u, ind

def fit(train_rows):
    X = np.array([_features(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def pred(p):
        vmax, kh, ctrl, effect = p
        x, u, ind = X.T
        q = (x / 1.5) ** 2
        return vmax * q / (kh + q) - ctrl * u + effect * ind
    res = least_squares(lambda p: pred(p) - y, x0=np.array([9.0, 1.0, 0.15, 0.5]), bounds=([0.0, 0.01, -5.0, -10.0], [30.0, 30.0, 5.0, 10.0]))
    return res.x.tolist()

def predict(row):
    # Fit parameters are attached by the standard fit/predict harness when available.
    p = getattr(predict, '_params', None)
    if p is None:
        p = [9.0, 1.0, 0.15, 0.5]
    vmax, kh, ctrl, effect = p
    x, u, ind = _features(row)
    q = (x / 1.5) ** 2
    return float(vmax * q / (kh + q) - ctrl * u + effect * ind)
