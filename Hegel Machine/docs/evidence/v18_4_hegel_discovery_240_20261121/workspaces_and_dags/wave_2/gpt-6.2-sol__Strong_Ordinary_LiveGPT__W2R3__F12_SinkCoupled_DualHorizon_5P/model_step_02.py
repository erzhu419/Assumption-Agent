import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return drive * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(p):
        return _law(p, u, s, h) - y
    bounds = ([0.0, 0.0, 1e-5, 0.0, 1e-5], [np.inf, 1.0, np.inf, np.inf, np.inf])
    result = least_squares(residual, [8.0, 0.7, 0.35, 1.7, 11.0], bounds=bounds, x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params

def predict(row):
    return float(_law(_params, row['u_now'], row['s_grad'], row['H']))