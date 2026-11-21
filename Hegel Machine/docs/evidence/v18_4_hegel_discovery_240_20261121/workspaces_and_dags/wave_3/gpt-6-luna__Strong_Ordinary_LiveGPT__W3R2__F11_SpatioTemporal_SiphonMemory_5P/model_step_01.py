import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    a_slow, b_couple, c_drive, tau_fast, rho = p
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    j = (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a_slow * xv + b_couple * zv + c_drive * j, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = [np.array([0.5, 0.3, 0.4, 0.05, 0.7]), np.array([1.0, 0.1, 0.2, 0.02, 0.5]), np.array([0.2, 0.6, 0.6, 0.1, 0.9])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 0.0, 0.0, 0.001, 0.0], [10.0, 10.0, 10.0, 2.0, 1.5]), max_nfev=3000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x

def predict(row):
    return float(_simulate(row, fit_cache))