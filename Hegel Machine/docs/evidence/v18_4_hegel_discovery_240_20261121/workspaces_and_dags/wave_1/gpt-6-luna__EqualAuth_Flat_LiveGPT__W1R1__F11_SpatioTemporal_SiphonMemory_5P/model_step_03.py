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
    def rhs(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + c_drive * u,
                         (0.52 * xx - zz) / tau_fast])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - y
    starts = [[0.5, 0.2, 0.5, 0.2], [1.0, 0.1, 1.0, 0.5], [0.2, 0.5, 0.2, 1.0]]
    sols = [least_squares(residual, s, bounds=([0, 0, 0, 1e-4], [10, 10, 10, 10]), max_nfev=3000) for s in starts]
    return min(sols, key=lambda sol: float(np.dot(sol.fun, sol.fun))).x

def predict(row):
    return _simulate(row, _FIT_PARAMS)
