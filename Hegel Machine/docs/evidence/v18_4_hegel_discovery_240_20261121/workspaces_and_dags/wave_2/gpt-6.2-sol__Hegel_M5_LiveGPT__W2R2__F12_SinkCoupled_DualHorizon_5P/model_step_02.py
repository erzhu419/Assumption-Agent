import numpy as np
from scipy.optimize import least_squares

_params = None


def _response(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_gap = p
    tau_slow = tau_fast + tau_gap
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow


def fit(train_rows):
    global _params
    u, s, h, y = np.array([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(p):
        return _response(p, u, s, h) - y
    bounds = ([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf])
    best = None
    for seed in ([6.5, 0.7, 0.25, 1.5, 7.0], [8.0, 0.6, 0.6, 2.0, 18.0], [5.0, 0.8, 0.1, 1.2, 3.0]):
        result = least_squares(residual, seed, bounds=bounds, x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x


def predict(row):
    return float(_response(_params, row['u_now'], row['s_grad'], row['H']))
