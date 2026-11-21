import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u, s = _features(row)
    u = max(1e-8, u)
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - ys
    starts = [[4.0, 1.0, 0.5, 0.3], [10.0, 4.0, 1.0, 0.5], [20.0, 10.0, 2.0, 1.0]]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=([1e-8,1e-8,0.0,0.0],[1e3,1e3,1e3,1e3]), max_nfev=20000)
        score = float(np.sum(result.fun**2))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    return float(_model(row, _fitted_params))