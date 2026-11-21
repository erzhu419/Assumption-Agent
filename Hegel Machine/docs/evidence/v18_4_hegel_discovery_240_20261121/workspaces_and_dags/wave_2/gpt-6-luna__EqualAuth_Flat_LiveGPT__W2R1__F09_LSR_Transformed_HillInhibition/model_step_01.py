import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 2.5
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 1.0, 0.3, 0.1]), bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    p = fit(train_rows)
    return float(_value(row, p))