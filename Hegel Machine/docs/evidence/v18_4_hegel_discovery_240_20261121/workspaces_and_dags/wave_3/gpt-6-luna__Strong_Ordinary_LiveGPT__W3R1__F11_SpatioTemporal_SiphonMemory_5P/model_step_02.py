import math
import numpy as np
from scipy.optimize import least_squares


def _rk4(row, p):
    a_slow, b_couple, tau_fast, vmax, rho = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * ((u / 1.0) / (1.2 + u / 1.0))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(x, z):
        return -a_slow * x + b_couple * z + j, (0.52 * x - z) / tau_fast
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x


def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(q):
        p = np.exp(q)
        return np.array([_rk4(r, p) for r in train_rows]) - y
    starts = [np.log([0.3, 0.15, 0.2, 2.0, 0.5]), np.log([0.8, 0.5, 0.5, 3.0, 0.7]), np.log([0.1, 0.05, 1.0, 1.0, 0.3])]
    best = None
    for q0 in starts:
        result = least_squares(residual, q0, max_nfev=3000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return np.exp(best.x)


def predict(row):
    return float(_rk4(row, _MODEL_PARAMS))