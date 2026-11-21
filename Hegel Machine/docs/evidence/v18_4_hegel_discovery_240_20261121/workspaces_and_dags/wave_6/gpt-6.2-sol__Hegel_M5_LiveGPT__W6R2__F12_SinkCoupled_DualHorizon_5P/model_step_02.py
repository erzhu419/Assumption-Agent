import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([(r['u_now'], r['s_grad'], r['H'], r['y']) for r in train_rows], dtype=float).T
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, amp, log_fast, log_gap = p
        tau_fast = math.exp(log_fast)
        tau_slow = tau_fast + math.exp(log_gap)
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return vmax * (1.0 - rho * sink) * drive * fast - amp * u * slow - y

    bounds = ([0.0, 0.0, 0.0, -20.0, -20.0], [np.inf, 1.0, np.inf, 20.0, 20.0])
    best = None
    for fast0, gap0 in [(0.3, 12.0), (0.1, 4.0), (0.8, 25.0)]:
        result = least_squares(residual, [6.5, 0.70, 1.5, math.log(fast0), math.log(gap0)], bounds=bounds, x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x


def predict(row):
    vmax, rho, amp, log_fast, log_gap = _params
    tau_fast = math.exp(log_fast)
    tau_slow = tau_fast + math.exp(log_gap)
    u, s, h = row['u_now'], row['s_grad'], row['H']
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / tau_slow)
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp * u * slow)
