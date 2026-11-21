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
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_model(p, r) for r in rows]) - y
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.01], [100.0, 1.5, 100.0, 100.0, 1000.0])
    starts = [[5.0, 0.7, 0.3, 1.0, 10.0], [10.0, 0.7, 1.0, 1.0, 20.0], [5.0, 0.7, 0.1, 2.0, 5.0]]
    results = [least_squares(residual, p, bounds=bounds, max_nfev=10000) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return float(_model(_params, row))