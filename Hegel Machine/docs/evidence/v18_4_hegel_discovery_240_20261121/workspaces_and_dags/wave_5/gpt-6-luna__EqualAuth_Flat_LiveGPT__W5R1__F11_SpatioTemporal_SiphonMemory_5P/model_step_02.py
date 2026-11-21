import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 2.0

def _simulate(row, p):
    a_slow, b_couple, vmax, tau_fast, rho = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    H = float(row['H'])
    siph = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(H * 120)))
    dt = H / n
    def f(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + siph,
                         (0.52 * xx - zz) / tau_fast], dtype=float)
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    starts = [np.array([0.5, 0.5, 2.0, 1.0, 0.5]),
              np.array([1.0, 1.0, 3.0, 2.0, 0.8]),
              np.array([0.2, 1.5, 1.5, 0.5, 0.2])]
    sols = [least_squares(residual, x0, bounds=([0.001, 0.001, 0.001, 0.03, 0.0], [5.0, 5.0, 10.0, 10.0, 2.0]), max_nfev=5000) for x0 in starts]
    best = min(sols, key=lambda q: float(np.dot(q.fun, q.fun)))
    _PARAMS = best.x.copy()
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return _simulate(row, _PARAMS)
