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
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    h = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.1], [100.0, 1.0, 20.0, 20.0, 1000.0])
    best = None
    for tf in (0.2, 0.7, 2.0):
        for ts in (5.0, 20.0, 60.0):
            result = least_squares(lambda p: _model(p, u, s, h) - y, [6.5, 0.7, tf, 1.0, ts], bounds=bounds, max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad']), float(row['H'])))