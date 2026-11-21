import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(p, row):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_params(p, r) for r in rows]) - y
    starts = [
        [1.0, 0.5, 1.0, 10.0],
        [1.0, 1.0, 1.0, 20.0],
        [2.0, 0.5, 2.0, 15.0],
        [0.5, 0.2, 0.5, 8.0],
    ]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=([0.0, 1e-6, 0.0, 1e-6], [np.inf, np.inf, np.inf, np.inf]), max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def predict(row):
    return _predict_params(fit(_TRAIN_ROWS), row)