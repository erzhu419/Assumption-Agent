import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / 0.05
    sg = float(row['s_grad'])
    H = float(row['H'])
    J = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
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
        x += (h/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        z += (h/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
    return x

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    lo = np.array([0.0, 0.0, 0.0, 0.0, 0.001])
    hi = np.array([20.0, 1.0, 20.0, 20.0, 2.0])
    starts = [np.array([2.0, 0.5, 0.5, 0.2, 0.05]), np.array([5.0, 0.7, 1.0, 0.5, 0.1]), np.array([1.0, 0.3, 0.2, 1.0, 0.02])]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _params))