import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u_n = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    H = float(row['H'])
    x, z = float(row['x_now']), float(row['mem_conv'])
    n = max(80, int(math.ceil(400.0 * H)))
    dt = H / n
    def f(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + drive,
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
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = [[0.5, 0.3, 6.0, 0.7, 0.8], [1.0, 0.8, 4.0, 0.5, 1.5], [0.2, 1.2, 8.0, 0.85, 0.4]]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=([1e-5, 0.0, 1e-5, 0.0, 1e-4], [10.0, 10.0, 30.0, 0.999, 20.0]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return {'params': best.x.tolist()}

def predict(row):
    p = row.get('_fit_params')
    if p is None:
        p = row.get('params')
    if p is None:
        raise ValueError('predict requires fitted parameters; pass them in row["_fit_params"] or row["params"]')
    return _simulate(row, p)
