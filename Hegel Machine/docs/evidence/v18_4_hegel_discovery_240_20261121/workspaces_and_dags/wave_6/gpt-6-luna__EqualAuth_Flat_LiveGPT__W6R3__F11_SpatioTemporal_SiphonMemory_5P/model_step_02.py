import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 2.0

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    H = float(row['H'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    n = max(12, int(math.ceil(H * 100)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + J,
                         (0.52 * xx - zz) / tau_fast], dtype=float)
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    lower = np.array([1e-6, 0.0, 1e-6, 0.0, 1e-3])
    upper = np.array([20.0, 2.0, 10.0, 10.0, 20.0])
    starts = [[2.0, 0.5, 0.5, 0.5, 1.0], [4.0, 0.8, 1.0, 1.0, 0.5], [1.0, 0.2, 0.2, 2.0, 2.0]]
    results = [least_squares(residual, np.array(st, dtype=float), bounds=(lower, upper), max_nfev=5000) for st in starts]
    return min(results, key=lambda r: float(np.dot(r.fun, r.fun))).x

def predict(row):
    return _simulate(row, fit([row]))