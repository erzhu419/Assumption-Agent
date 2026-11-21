import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    a_slow, b_couple, tau_fast, vmax, rho = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    H = float(row['H'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    dt = 0.004
    n = max(1, int(math.ceil(H / dt)))
    h = H / n
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + J, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*h*k1x, z + 0.5*h*k1z)
        k3x, k3z = deriv(x + 0.5*h*k2x, z + 0.5*h*k2z)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x += h*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += h*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    starts = [np.array([0.2, 0.1, 0.05, 1.0, 0.5]), np.array([0.5, 0.2, 0.1, 2.0, 0.7]), np.array([0.05, 0.05, 0.02, 1.0, 0.2])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 0.0, 0.001, 0.0, 0.0], [5.0, 5.0, 2.0, 20.0, 1.0]), max_nfev=4000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x

def predict(row):
    return float(_simulate(row, fit(_TRAIN_ROWS)))