import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    un = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return un, s

def _pred(p, row):
    vmax, k_half, alpha_inh, c_leak = p
    un, s = _features(row)
    return vmax * (un * un / (k_half + un * un)) / (1.0 + alpha_inh * s) - c_leak * un

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_pred(p, r) for r in rows]) - y
    best = None
    starts = ([3.0, 1.0, 0.2, 0.05], [5.0, 2.0, 0.5, 0.1], [10.0, 5.0, 0.1, 0.01], [2.0, 0.2, 1.0, 0.2])
    for x0 in starts:
        result = least_squares(residual, x0=np.asarray(x0), bounds=([0.0, 1e-9, 0.0, 0.0], [1000.0, 1e4, 100.0, 100.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return {'params': best[1].tolist()}

def predict(row):
    return float(_pred(_MODEL['params'], row))