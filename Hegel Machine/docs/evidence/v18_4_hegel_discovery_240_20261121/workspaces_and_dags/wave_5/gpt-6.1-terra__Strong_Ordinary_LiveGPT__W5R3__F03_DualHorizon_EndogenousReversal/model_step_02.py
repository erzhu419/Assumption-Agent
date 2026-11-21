import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, u, h):
    fast_amp, slow_amp, tau_fast, delta_tau = p
    tau_slow = tau_fast + delta_tau
    return u * (fast_amp * (-np.expm1(-h / tau_fast)) - slow_amp * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for tf in (0.3, 1.0, 2.0):
        for ts in (4.0, 12.0, 30.0):
            if ts <= tf:
                continue
            result = least_squares(lambda p: _values(p, u, h) - y,
                                   [2.0, 3.0, tf, ts - tf],
                                   bounds=([0.0, 0.0, 1e-5, 1e-5],
                                           [1000.0, 1000.0, 1000.0, 1000.0]),
                                   max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_values(_params, float(row['u_now']), float(row['H'])))
