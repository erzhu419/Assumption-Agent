import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + drive, (0.52 * xv - zv) / tau_fast
    n = max(40, int(math.ceil(H * 240)))
    dt = H / n
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - y
    starts = [[0.5, 0.2, 6.5, 0.7, 0.5], [1.0, 0.5, 4.0, 0.65, 1.0], [0.2, 1.0, 8.0, 0.75, 1.5]]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=([0.001, -3.0, 0.01, 0.001, 0.03], [5.0, 3.0, 30.0, 0.999, 10.0]), max_nfev=5000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _params))