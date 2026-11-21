import math
import numpy as np
from scipy.optimize import least_squares

TAU = 2.5

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / TAU
    grad = float(row['s_grad'])
    hill = (u * u) / (k_half + u * u)
    return vmax * hill / (1.0 + alpha_inh * grad) - c_leak * u

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(
        residual,
        x0=np.array([6.0, 0.5, 0.5, 0.05]),
        bounds=(np.array([0.0, 1e-10, 0.0, 0.0]), np.array([np.inf, np.inf, np.inf, np.inf])),
        max_nfev=20000,
    )
    return result.x

def predict(row):
    p = fit(_FIT_ROWS)
    return float(_value(row, p))