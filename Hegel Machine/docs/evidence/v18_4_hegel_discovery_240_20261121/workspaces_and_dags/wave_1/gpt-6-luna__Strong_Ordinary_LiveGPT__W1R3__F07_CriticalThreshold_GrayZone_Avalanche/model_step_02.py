import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return u, s

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u, s = _features(row)
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    best = None
    for start in ([4.0, 1.0, 0.5, 0.5], [8.0, 4.0, 1.0, 0.2], [3.0, 0.3, 0.2, 1.0], [10.0, 10.0, 2.0, 0.05]):
        result = least_squares(residual, start, bounds=([0.0, 1e-8, 0.0, 0.0], [100.0, 1e3, 100.0, 100.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_model(row, p))