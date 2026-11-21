import math
import numpy as np
from scipy.optimize import least_squares

_tau_scale = 0.05
_params = None

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u_n = row['u_now'] / _tau_scale
    s = row['s_grad']
    j_siph = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.001)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + j_siph,
                         (0.52 * xx - zz) / tau_fast], dtype=float)
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in rows]) - y
    lower = np.array([0.0, 0.0, 0.0, 0.0, 1e-4])
    upper = np.array([20.0, 20.0, 20.0, 1.0, 10.0])
    starts = [np.array([1.0, 0.5, 2.0, 0.5, t]) for t in (0.01, 0.05, 0.2, 1.0)]
    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=(lower, upper), max_nfev=3000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return _simulate(row, _params)
