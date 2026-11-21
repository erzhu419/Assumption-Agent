import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return u, s, h

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u, s, h = _features(row)
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [
        [4.0, 0.5, 0.5, 0.2, 10.0],
        [6.0, 0.8, 1.0, 0.15, 20.0],
        [3.0, 0.2, 0.2, 0.1, 5.0],
        [8.0, 0.6, 2.0, 0.3, 30.0],
    ]
    lo = [1e-8, 0.0, 1e-5, 0.0, 1e-4]
    hi = [100.0, 0.999999, 100.0, 100.0, 1000.0]
    fits = [least_squares(residual, np.asarray(p), bounds=(lo, hi), max_nfev=20000) for p in starts]
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    return best.x

def predict(row):
    return float(_model(row, fit([dict(row, y=0.0)])))