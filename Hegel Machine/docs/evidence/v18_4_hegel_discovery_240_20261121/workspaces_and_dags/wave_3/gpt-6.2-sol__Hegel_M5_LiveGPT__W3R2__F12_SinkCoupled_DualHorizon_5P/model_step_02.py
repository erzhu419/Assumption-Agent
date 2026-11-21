import numpy as np
from scipy.optimize import least_squares

_params = None


def _response(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow


def fit(train_rows):
    global _params
    u, s, h, y = np.array([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T

    def residual(p):
        return _response(p, u, s, h) - y

    bounds = ([0.0, 0.0, 1e-5, 0.0, 1e-5], [np.inf, 1.0, np.inf, np.inf, np.inf])
    fits = (least_squares(residual, [6.5, 0.7, tf, 1.0, ts], bounds=bounds, x_scale='jac', max_nfev=1500) for tf in (0.15, 0.6) for ts in (4.0, 15.0, 40.0))
    _params = min(fits, key=lambda result: result.cost).x


def predict(row) -> float:
    return float(_response(_params, row['u_now'], row['s_grad'], row['H']))
