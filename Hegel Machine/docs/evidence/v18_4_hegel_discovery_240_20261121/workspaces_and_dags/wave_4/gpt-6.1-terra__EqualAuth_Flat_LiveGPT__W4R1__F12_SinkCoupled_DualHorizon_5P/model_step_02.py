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
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in train_rows], dtype=float)
    h = np.asarray([r['H'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return _model(p, u, s, h) - y
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.05], [100.0, 1.0, 10.0, 20.0, 500.0])
    best = None
    for start in ([5.0, 0.5, 0.3, 1.0, 10.0], [10.0, 0.8, 1.0, 2.0, 20.0], [4.0, 0.3, 0.1, 0.5, 5.0]):
        result = least_squares(residual, start, bounds=bounds, max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad']), float(row['H'])))
