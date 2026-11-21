import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.1

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / TAU_SCALE
    hill = (u_n ** 2) / (k_half + u_n ** 2)
    return vmax * hill / (1.0 + alpha_inh * float(row['s_grad'])) - c_leak * u_n

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(z):
        p = np.exp(z)
        return np.array([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, np.log([4.0, 1.0, 0.3, 0.05]), max_nfev=20000)
    return np.exp(result.x)

def predict(row):
    p = fit(train_rows)
    return float(_model(row, p))