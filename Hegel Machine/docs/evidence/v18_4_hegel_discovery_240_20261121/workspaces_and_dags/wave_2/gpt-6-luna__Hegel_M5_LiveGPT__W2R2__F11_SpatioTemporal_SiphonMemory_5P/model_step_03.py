import math
import numpy as np
from scipy.optimize import least_squares

_tau_scale = 2.0
_params = None

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / _tau_scale
    sg = float(row['s_grad'])
    H = float(row['H'])
    drive = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + drive, (0.52 * xv - zv) / tau_fast
    n = max(1, int(math.ceil(H / 0.005)))
    dt = H / n
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return x

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = [np.array([0.5, 0.1, 6.5, 0.7, 1.0]), np.array([0.3, 0.4, 5.0, 0.5, 2.0]), np.array([1.0, -0.2, 8.0, 0.8, 0.5])]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=([0.001, -3.0, 0.01, 0.001, 0.02], [5.0, 3.0, 20.0, 0.999, 10.0]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _params))