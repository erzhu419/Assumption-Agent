import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 2.0
DT = 0.004

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(H / DT)))
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
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - y
    starts = [np.array([2.0, 0.5, 0.5, 0.3, 0.5]), np.array([4.0, 0.7, 1.0, 0.8, 1.0]), np.array([1.0, 0.2, 0.2, 0.1, 0.2])]
    lo = np.array([0.0, 0.0, 0.0, -5.0, 0.02])
    hi = np.array([20.0, 0.999, 5.0, 5.0, 10.0])
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=(lo, hi), max_nfev=4000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x

def predict(row):
    return float(_simulate(row, fit_params))