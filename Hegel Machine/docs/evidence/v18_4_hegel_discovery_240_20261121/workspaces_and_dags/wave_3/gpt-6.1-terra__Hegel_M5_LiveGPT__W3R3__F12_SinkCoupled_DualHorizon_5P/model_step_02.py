import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            * (-np.expm1(-h / tau_fast))
            - amp_adapt * u * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])
    def residual(p):
        return np.array([_value(row, p) for row in rows]) - y
    bounds = ([0.0, 0.0, 0.01, 0.0, 0.1], [30.0, 1.5, 20.0, 10.0, 200.0])
    starts = [[6.5, 0.7, tf, a, ts] for tf in (0.5, 1.0, 2.0) for a in (0.8, 2.0) for ts in (5.0, 20.0)]
    results = [least_squares(residual, p, bounds=bounds, max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_value(row, _params))
