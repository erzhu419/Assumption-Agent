import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    global _params
    keys = ('x_ind', 'z_1', 'z_2', 'z_3', 'z_4', 'u_ctrl')
    X = np.array([[float(r.get(k, 0.0)) for k in keys] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Hill saturation of an affine effective substrate; u_ctrl is an additive
    # nuisance/control effect, allowing the passive-regime subtraction to shift.
    def pred(p, row):
        v, k, b0, bx, b1, b2, b3, b4, cu = p
        s = max(0.0, b0 + bx*row.get('x_ind', 0.0) + b1*row.get('z_1', 0.0) + b2*row.get('z_2', 0.0) + b3*row.get('z_3', 0.0) + b4*row.get('z_4', 0.0))
        q = s*s
        return v*q/(k+q) - cu*row.get('u_ctrl', 0.0)
    def residual(p):
        return np.array([pred(p, r) for r in train_rows]) - y
    p0 = np.array([7.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lo = np.array([0.01, 1e-5, -10, -5, -5, -5, -5, -5, -5])
    hi = np.array([30, 100, 10, 5, 5, 5, 5, 5, 5])
    res = least_squares(residual, p0, bounds=(lo, hi), max_nfev=5000)
    _params = res.x
    return _params


def predict(row):
    v, k, b0, bx, b1, b2, b3, b4, cu = _params
    s = max(0.0, b0 + bx*row.get('x_ind', 0.0) + b1*row.get('z_1', 0.0) + b2*row.get('z_2', 0.0) + b3*row.get('z_3', 0.0) + b4*row.get('z_4', 0.0))
    q = s*s
    return float(v*q/(k+q) - cu*row.get('u_ctrl', 0.0))