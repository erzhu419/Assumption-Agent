import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [float(r['y']) for r in rows]
    def residual(p):
        return np.array([_model(p, r) for r in rows]) - y
    best = None
    for tf in (0.1, 0.5, 1.5):
        for ts in (3.0, 10.0, 30.0):
            result = least_squares(residual, [5.0, 0.7, tf, 1.0, ts], bounds=([0.0, 0.0, 0.001, 0.0, 0.01], [100.0, 2.0, 100.0, 20.0, 1000.0]), max_nfev=10000)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_model(_params, row))