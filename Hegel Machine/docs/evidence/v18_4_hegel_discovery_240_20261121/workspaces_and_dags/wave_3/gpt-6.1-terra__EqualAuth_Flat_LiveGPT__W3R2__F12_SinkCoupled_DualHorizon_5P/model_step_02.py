import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    return vmax * (1.0 - rho*s/(1.0+s)) * (u/(1.2+u)) * (-np.expm1(-h/tau_fast)) - amp_adapt*u*(-np.expm1(-h/tau_slow))

def fit(train_rows):
    global _params
    u = np.array [r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.05], [1000.0, 1.0, 50.0, 100.0, 1000.0])
    best = None
    for tf in (0.1, 0.5, 2.0):
        for ts in (3.0, 15.0, 50.0):
            res = least_squares(lambda p: _model(p, u, s, h)-y, [8.0, 0.5, tf, 1.5, ts], bounds=bounds, max_nfev=3000)
            if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
                best = res
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad']), float(row['H'])))