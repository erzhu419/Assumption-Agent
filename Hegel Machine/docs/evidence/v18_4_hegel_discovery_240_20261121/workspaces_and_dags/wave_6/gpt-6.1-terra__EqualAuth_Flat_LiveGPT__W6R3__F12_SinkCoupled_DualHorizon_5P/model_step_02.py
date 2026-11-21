import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, rows):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    return vmax * (1 - rho * s / (1 + s)) * u / (1.2 + u) * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [r['y'] for r in rows], dtype=float)
    bounds = ([0, 0, 0.001, 0, 0.01], [100, 2, 100, 100, 1000])
    best = None
    for start in ([8, 0.5, 0.5, 1, 10], [5, 0.8, 0.2, 2, 20], [10, 0.3, 1, 1, 30]):
        result = least_squares(lambda p: _model(p, rows) - y, start, bounds=bounds, max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_model(_params, [row])[0])