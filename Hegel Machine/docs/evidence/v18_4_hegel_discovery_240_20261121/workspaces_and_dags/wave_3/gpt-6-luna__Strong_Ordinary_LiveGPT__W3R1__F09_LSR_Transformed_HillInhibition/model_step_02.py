import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return vmax * (u * u / (k_half + u * u)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([3.0, 4.0, 0.2, 0.01]), bounds=([0.0, 1e-9, 0.0, 0.0], [100.0, 1e4, 100.0, 100.0]), max_nfev=20000)
    return result.x

def predict(row):
    return float(_value(row, fit([row])))