import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(rows):
    return (np.array([r['u_now'] for r in rows], dtype=float),
            np.array([r['s_grad'] for r in rows], dtype=float),
            np.array([r['H'] for r in rows], dtype=float))

def _model(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    return (vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            * (-np.expm1(-h / tau_fast))
            - amp_adapt * u * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    u, s, h = _values(train_rows)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for tf in (0.15, 0.5, 1.5):
        for ts in (5.0, 20.0, 60.0):
            result = least_squares(lambda p: _model(p, u, s, h) - y,
                [8.0, 0.7, tf, 1.0, ts],
                bounds=([0.0, 0.0, 0.001, 0.0, 0.1],
                        [1000.0, 1.5, 20.0, 100.0, 1000.0]),
                max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    u, s, h = _values([row])
    return float(_model(_params, u, s, h)[0])
