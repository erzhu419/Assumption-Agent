import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(r):
    x = float(r['x_ind']) / 1.5
    z1, z2, z3, z4 = (float(r[k]) for k in ('z_1','z_2','z_3','z_4'))
    u = float(r['u_ctrl'])
    # Tree-A indicator branch: z1 activates the squared-input Hill response;
    # retain the measured intervention input as its substrate.
    gate = 1.0 if z1 >= 1.0 else 0.0
    return x*x*gate, u

def fit(train_rows):
    global _params
    q = np.array([_features(r)[0] for r in train_rows], dtype=float)
    u = np.array([_features(r)[1] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, kh, c = p
        return vmax*q/(kh+q) - c*u - y
    res = least_squares(residual, [8.0, 1.0, 0.5], bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    _params = tuple(float(v) for v in res.x)
    return predict

def predict(row):
    vmax, kh, c = _params
    q, u = _features(row)
    return float(vmax*q/(kh+q) - c*u)