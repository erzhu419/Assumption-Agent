import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_model(p, r) - float(r['y']) for r in rows])
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.01], [100.0, 1.0, 100.0, 100.0, 1000.0])
    starts = [[6.5, 0.7, tf, a, ts] for tf in (0.2, 1.0, 3.0) for a in (0.8, 2.0) for ts in (5.0, 20.0)]
    results = [least_squares(residual, p, bounds=bounds, max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_model(_params, row))