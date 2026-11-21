import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 0.05

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    H = float(row['H'])
    u_n = float(row['u_now']) / TAU_SCALE
    sg = float(row['s_grad'])
    J = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    n = max(1, int(math.ceil(H / 0.002)))
    dt = H / n
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + J, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = [np.array([0.5, 0.5, 1.0, 0.3, 0.05]), np.array([1.0, 1.0, 2.0, 0.6, 0.1]), np.array([0.2, 2.0, 1.0, 0.1, 0.02])]
    fits = [least_squares(residual, s, bounds=([0.0, 0.0, 0.0, -5.0, 1e-4], [20.0, 20.0, 20.0, 5.0, 5.0]), max_nfev=5000) for s in starts]
    best = min(fits, key=lambda q: float(np.sum(q.fun*q.fun)))
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _PARAMS))