import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, rows):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = np.asarray([r['u_now'] for r in rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in rows], dtype=float)
    h = np.asarray([r['H'] for r in rows], dtype=float)
    fast = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * (-np.expm1(-h / tau_fast))
    slow = amp_adapt * u * (-np.expm1(-h / tau_slow))
    return fast - slow

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    bounds = ([1.0, 0.05, 0.005, 0.0, 1.0], [15.0, 0.95, 5.0, 10.0, 200.0])
    best = None
    for tf in (0.15, 0.5, 1.0):
        for ts in (5.0, 15.0, 40.0):
            result = least_squares(lambda p: _model(p, rows) - y,
                                   [6.5, 0.7, tf, 1.0, ts],
                                   bounds=bounds, max_nfev=5000,
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    return float(_model(_params, [row])[0])