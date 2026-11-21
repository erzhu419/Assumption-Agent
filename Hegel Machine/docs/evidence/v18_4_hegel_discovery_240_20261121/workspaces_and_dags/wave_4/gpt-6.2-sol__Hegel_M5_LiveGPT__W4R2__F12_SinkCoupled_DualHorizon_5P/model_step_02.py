import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([(r['u_now'], r['s_grad'], r['H'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(lambda p: _law(p, u, s, h) - y,
                            x0=[6.5, 0.70, 0.35, 1.6, 10.0],
                            bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6],
                                    [np.inf, 1.0, np.inf, np.inf, np.inf]),
                            x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11).x
    return _params

def predict(row) -> float:
    return float(_law(_params, row['u_now'], row['s_grad'], row['H']))
