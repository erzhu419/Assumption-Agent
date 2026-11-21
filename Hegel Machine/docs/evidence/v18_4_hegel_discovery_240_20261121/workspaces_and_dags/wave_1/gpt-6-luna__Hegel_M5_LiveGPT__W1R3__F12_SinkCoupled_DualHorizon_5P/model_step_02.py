import math
import numpy as np
from scipy.optimize import least_squares

# Parameters: vmax, rho, tau_fast, amp_adapt, tau_slow.
def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (-math.expm1(-h / tau_fast)) - amp_adapt * u * (-math.expm1(-h / tau_slow))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    lo = np.array([1.0, 0.05, 0.01, 0.0, 0.1])
    hi = np.array([15.0, 0.95, 10.0, 5.0, 200.0])
    starts = [
        [6.5, 0.70, 0.25, 0.8, 20.0],
        [5.0, 0.35, 0.10, 0.7, 35.0],
        [9.0, 0.85, 0.50, 1.0, 15.0],
        [7.0, 0.65, 1.0, 0.5, 60.0],
    ]
    best = None
    for start in starts:
        res = least_squares(
            lambda p: np.asarray([_model(r, p) for r in rows]) - y,
            np.clip(np.asarray(start, dtype=float), lo + 1e-9, hi - 1e-9),
            bounds=(lo, hi), max_nfev=30000
        )
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    return best[1]

def predict(row):
    return float(_model(row, fit([dict(row, y=0.0)])))