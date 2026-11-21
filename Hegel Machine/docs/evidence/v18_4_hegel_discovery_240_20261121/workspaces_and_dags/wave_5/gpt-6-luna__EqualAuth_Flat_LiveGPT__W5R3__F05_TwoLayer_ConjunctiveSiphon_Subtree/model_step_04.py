import math
import numpy as np
from scipy.optimize import least_squares

def _features(row, p):
    vmax, khalf, alpha, leak = p
    u = max(1e-4, float(row['u_now'])) / 4.0
    s = max(0.0, float(row['s_grad']))
    return vmax * (u*u / (khalf + u*u)) / (1.0 + alpha*s) - leak*u

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_features(r, p) for r in train_rows]) - y
    best = None
    for start in ([4.0, 0.5, 0.5, 0.0], [5.0, 1.0, 1.0, 0.1], [3.0, 0.2, 0.2, 0.0]):
        res = least_squares(residual, start, bounds=([0, 1e-6, 0, 0], [100, 100, 100, 100]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    return float(_features(row, _fit_params))