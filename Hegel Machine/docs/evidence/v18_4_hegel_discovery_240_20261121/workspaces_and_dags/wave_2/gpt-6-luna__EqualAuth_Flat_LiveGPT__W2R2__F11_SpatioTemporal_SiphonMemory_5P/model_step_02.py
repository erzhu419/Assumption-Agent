import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 2.0
_params = None

def _simulate(row, p):
    a_slow, b_couple, tau_fast, vmax, rho = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    u_n = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + j, (0.52 * xv - zv) / tau_fast
    n = max(8, int(math.ceil(H * 160)))
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
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = [np.array([0.5, 0.3, 1.0, 2.0, 0.5]), np.array([1.0, 0.6, 2.0, 3.0, 0.2])]
    results = [least_squares(residual, q, bounds=([0.001, 0.001, 0.05, 0.001, 0.0], [5.0, 5.0, 10.0, 20.0, 1.5]), max_nfev=5000) for q in starts]
    _params = min(results, key=lambda r: float(np.sum(r.fun**2))).x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_simulate(row, _params))