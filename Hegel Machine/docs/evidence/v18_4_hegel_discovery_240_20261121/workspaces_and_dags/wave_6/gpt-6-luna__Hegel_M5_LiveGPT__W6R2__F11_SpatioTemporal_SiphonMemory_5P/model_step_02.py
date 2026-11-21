import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u_n = row['u_now'] / 0.005
    s = row['s_grad']
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    # Resolve the fast memory timescale with small RK4 steps.
    n = max(1, int(math.ceil(H / min(0.00025, tau_fast / 8.0))))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + drive,
                         (0.52 * xx - zz) / tau_fast], dtype=float)
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    lower = np.array([0.0, -2.0, 1.0, 0.05, 0.0002])
    upper = np.array([20.0, 2.0, 15.0, 0.95, 0.1])
    starts = [np.array([1.0, 0.1, 6.5, 0.70, 0.005]),
              np.array([2.0, -0.5, 5.0, 0.65, 0.01]),
              np.array([0.5, 0.8, 8.0, 0.75, 0.002])]
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.array([_simulate(r, p) for r in rows]) - y,
                            x0, bounds=(lower, upper), max_nfev=500)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return _simulate(row, _PARAMS)
