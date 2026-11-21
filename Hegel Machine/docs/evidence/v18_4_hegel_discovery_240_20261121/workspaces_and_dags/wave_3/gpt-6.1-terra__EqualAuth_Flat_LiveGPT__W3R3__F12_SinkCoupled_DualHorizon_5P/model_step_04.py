import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for tf in (0.1, 0.5, 2.0):
        for ts in (3.0, 15.0, 50.0):
            result = least_squares(lambda p: _model(p, u, s, h) - y,
                [5.0, 0.5, tf, 1.0, ts],
                bounds=([0.0, 0.0, 0.001, 0.0, 0.01], [100.0, 1.0, 100.0, 100.0, 1000.0]),
                max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad']), float(row['H'])))