import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    siphon = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return siphon * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return _values(p, u, s, h) - y
    best = None
    for tf in (0.15, 0.5, 1.5):
        for ts in (3.0, 10.0, 30.0):
            result = least_squares(residual, [5.0, 0.7, tf, 1.0, ts], bounds=([0.0, 0.0, 0.001, 0.0, 0.05], [1000.0, 1.5, 100.0, 100.0, 1000.0]), max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
                best = result
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    return float(_values(_params, float(row['u_now']), float(row['s_grad']), float(row['H'])))