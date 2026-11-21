import numpy as np
from scipy.optimize import least_squares

_params = None

def _gain(h, log_params):
    amp_fast, amp_slow, tau_fast, tau_gap = np.exp(log_params)
    tau_slow = tau_fast + tau_gap
    return -amp_fast * np.expm1(-h / tau_fast) + amp_slow * np.expm1(-h / tau_slow)

def fit(train_rows):
    global _params
    h, u, y = np.array([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(p):
        return u * _gain(h, p) - y

    best = None
    for tau_fast, tau_slow in ((0.3, 8.0), (0.6, 4.0), (0.2, 20.0), (1.0, 12.0)):
        p0 = np.log([1.7, 2.3, tau_fast, tau_slow - tau_fast])
        result = least_squares(residual, p0, bounds=(-12.0, 12.0), max_nfev=1500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or result.cost < best.cost:
            best = result
    _params = best.x

def predict(row) -> float:
    return float(row['u_now'] * _gain(row['H'], _params))
