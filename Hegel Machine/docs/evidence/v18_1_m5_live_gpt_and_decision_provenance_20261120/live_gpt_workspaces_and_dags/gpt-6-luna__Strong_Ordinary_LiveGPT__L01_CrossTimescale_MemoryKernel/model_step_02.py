import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    H = float(row['H'])
    if H <= 0:
        return x
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + c_drive * u,
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
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = ([0.5, 0.5, 0.5, 0.3], [1.0, 1.0, 1.0, 1.0], [0.2, 2.0, 0.5, 0.1])
    best = None
    for start in starts:
        result = least_squares(residual, np.asarray(start, dtype=float),
                               bounds=([0.0, 0.0, 0.0, 0.01], [10.0, 10.0, 10.0, 10.0]),
                               max_nfev=3000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def predict(row):
    return _simulate(row, fit([dict(row, y=0.0)]))