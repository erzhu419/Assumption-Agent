import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, rows):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    best = None
    for tf in (0.1, 0.5, 1.5):
        for ts in (3.0, 10.0, 30.0):
            result = least_squares(lambda p: _values(p, rows) - y,
                [5.0, 0.7, tf, 1.0, ts],
                bounds=([0.0, 0.0, 0.001, 0.0, 0.01], [1000.0, 1.0, 100.0, 100.0, 1000.0]),
                max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or result.cost < best.cost:
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_values(_params, [row])[0])