import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, rows):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    inflow = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return inflow * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    bounds = ([0.01, 0.0, 0.01, 0.0, 0.1], [50.0, 1.0, 20.0, 10.0, 200.0])
    best = None
    for tf in (0.3, 1.0, 2.0):
        for ts in (5.0, 15.0, 40.0):
            result = least_squares(lambda p: _evaluate(p, rows) - y, [6.5, 0.7, tf, 1.0, ts], bounds=bounds, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or result.cost < best.cost:
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate(_params, [row])[0])