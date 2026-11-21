import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, rows):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    h = np.array([float(r['H']) for r in rows])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    best = None
    for initial in ([5.0, 0.5, 0.5, 1.0, 10.0], [8.0, 1.0, 1.0, 1.5, 20.0], [3.0, 0.2, 0.2, 0.5, 5.0]):
        result = least_squares(lambda p: _evaluate(p, rows) - y, initial, bounds=([0.0, 0.0, 0.001, 0.0, 0.01], [100.0, 2.0, 100.0, 100.0, 1000.0]), max_nfev=20000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or result.cost < best.cost:
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate(_params, [row])[0])