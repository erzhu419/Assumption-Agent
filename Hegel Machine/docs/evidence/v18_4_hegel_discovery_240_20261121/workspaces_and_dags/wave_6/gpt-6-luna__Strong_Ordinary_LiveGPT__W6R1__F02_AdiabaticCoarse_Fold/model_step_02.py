import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + c_drive * u,
                         (0.52 * xx - zz) / tau_fast])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    starts = [np.array([0.4, 0.1, 2.0, 0.25]), np.array([0.2, 0.3, 1.5, 0.5]), np.array([0.8, 0.05, 2.2, 0.1])]
    sols = [least_squares(residual, p0, bounds=([1e-5, -2.0, -5.0, 1e-4], [5.0, 2.0, 5.0, 10.0]), max_nfev=3000) for p0 in starts]
    return min(sols, key=lambda s: float(np.sum(s.fun**2))).x

def predict(row):
    return _simulate(row, _params)
