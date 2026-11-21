import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    gain = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
    adapt = amp_adapt * u * (1.0 - math.exp(-H / tau_slow))
    return gain - adapt

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_pred(r, p) for r in rows]) - y
    best = None
    starts = [
        [4.0, 0.5, 0.5, 0.15, 10.0],
        [3.0, 0.2, 1.0, 0.1, 20.0],
        [5.0, 0.8, 0.2, 0.2, 8.0],
    ]
    for x0 in starts:
        result = least_squares(residual, x0=np.array(x0), bounds=([0.0, 0.0, 1e-5, 0.0, 1e-4], [20.0, 1.0, 20.0, 5.0, 200.0]), max_nfev=30000)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_pred(row, p))