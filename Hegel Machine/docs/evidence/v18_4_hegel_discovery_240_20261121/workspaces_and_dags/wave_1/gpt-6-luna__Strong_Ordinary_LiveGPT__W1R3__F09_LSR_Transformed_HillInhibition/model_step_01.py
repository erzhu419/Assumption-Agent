import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.1

def _predict_params(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    hill = (u_n * u_n) / (k_half + u_n * u_n)
    return vmax * hill / (1.0 + alpha_inh * s) - c_leak * u_n

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    starts = ([3.0, 2.0, 0.3, 0.01], [5.0, 10.0, 1.0, 0.0], [2.0, 0.5, 0.05, 0.05])
    fits = [least_squares(residual, x0, bounds=([1e-9, 1e-9, 0.0, 0.0], [1e3, 1e4, 1e3, 1e3]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    return tuple(float(v) for v in best.x)

def predict(row):
    return float(_predict_params(row, fit_parameters))