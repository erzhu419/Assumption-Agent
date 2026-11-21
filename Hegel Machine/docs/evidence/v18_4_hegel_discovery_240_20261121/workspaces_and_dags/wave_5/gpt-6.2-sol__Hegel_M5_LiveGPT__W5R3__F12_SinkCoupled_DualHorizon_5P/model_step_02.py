import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float)
    u, s, h, y = data.T
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, tau_fast, amp_adapt, tau_slow = p
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return vmax * (1.0 - rho * sink) * drive * fast - amp_adapt * u * slow - y

    bounds = ([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf])
    starts = ([6.5, 0.7, 0.3, 1.5, 8.0], [8.0, 0.7, 0.6, 2.0, 20.0], [5.0, 0.6, 0.15, 1.5, 3.0])
    _params = min((least_squares(residual, start, bounds=bounds, x_scale='jac', max_nfev=600) for start in starts), key=lambda result: result.cost).x

def predict(row) -> float:
    vmax, rho, tau_fast, amp_adapt, tau_slow = _params
    u, s, h = row['u_now'], row['s_grad'], row['H']
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / tau_slow)
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow)
