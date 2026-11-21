import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now'])  # tau_scale = 1.0
    sg = float(row['s_grad'])
    H = float(row['H'])
    j_siph = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + j_siph, (0.52 * xv - zv) / tau_fast
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
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = [np.array([3.0, 0.5, 1.0, 0.5, 1.0]), np.array([2.0, 0.2, 0.5, 1.0, 0.5]), np.array([4.0, 0.8, 2.0, 0.2, 2.0])]
    lo = np.array([0.0, 0.0, 0.0, 0.0, 0.03])
    hi = np.array([20.0, 1.0, 20.0, 20.0, 20.0])
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_simulate(row, _params))