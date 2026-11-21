import numpy as np
from scipy.optimize import least_squares

params = None

def _values(rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    return u, s, h

def _model(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global params
    u, s, h = _values(train_rows)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.1], [100.0, 1.5, 20.0, 100.0, 1000.0])
    best = None
    for tf in (0.1, 0.5, 2.0):
        for ts in (5.0, 20.0, 80.0):
            result = least_squares(lambda p: _model(p, u, s, h) - y, [8.0, 0.7, tf, 1.0, ts], bounds=bounds, max_nfev=5000)
            if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
                best = result
    params = best.x
    return params

def predict(row):
    u, s, h = _values([row])
    return float(_model(params, u, s, h)[0])