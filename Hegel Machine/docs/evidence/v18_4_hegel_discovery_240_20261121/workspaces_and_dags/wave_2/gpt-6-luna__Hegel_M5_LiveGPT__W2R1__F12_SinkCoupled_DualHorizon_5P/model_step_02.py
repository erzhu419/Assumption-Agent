import math
import numpy as np
from scipy.optimize import least_squares

def _pred(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast))
    adapt = amp_adapt * u * (1.0 - math.exp(-h / tau_slow))
    return base - adapt

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_pred(p, r) for r in train_rows]) - y
    starts = [
        [6.0, 0.70, 0.7, 0.8, 12.0],
        [5.0, 0.70, 1.0, 0.5, 20.0],
        [8.0, 0.65, 0.4, 0.3, 8.0],
        [4.0, 0.75, 1.5, 0.2, 30.0],
    ]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0=x0,
            bounds=([0.01, 0.0, 0.01, 0.0, 0.05], [30.0, 0.999, 20.0, 10.0, 200.0]),
            max_nfev=20000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    return best[1]

def predict(row):
    p = fit(train_rows)
    return float(_pred(p, row))