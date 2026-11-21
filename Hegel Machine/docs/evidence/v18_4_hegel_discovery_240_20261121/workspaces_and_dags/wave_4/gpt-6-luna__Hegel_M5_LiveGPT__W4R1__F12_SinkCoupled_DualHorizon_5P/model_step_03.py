import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(q):
        p = np.exp(q)
        return np.asarray([_pred(r, p) for r in train_rows]) - ys
    starts = [np.log([0.5, 1.0, 0.5, 20.0]), np.log([1.0, 0.5, 0.5, 10.0]), np.log([0.4, 2.0, 0.4, 30.0])]
    best = None
    for q0 in starts:
        result = least_squares(residual, q0, max_nfev=30000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, np.exp(result.x))
    return best[1]

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_pred(row, p))