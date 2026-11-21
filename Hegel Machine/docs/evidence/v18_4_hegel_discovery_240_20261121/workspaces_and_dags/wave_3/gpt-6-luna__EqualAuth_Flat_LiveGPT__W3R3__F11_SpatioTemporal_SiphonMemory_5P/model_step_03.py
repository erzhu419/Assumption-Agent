import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _simulate(row, p):
    a_slow, b_couple, tau_fast, vmax, rho = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    u_n = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(H / 0.005)))
    dt = H / n
    def f(xv, zv):
        return -a_slow * xv + b_couple * zv + J, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = f(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return x

def fit(train_rows):
    global _P
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = [np.array([0.5, 0.5, 0.5, 3.0, 0.5]), np.array([1.0, 1.0, 1.0, 4.0, 0.3])]
    lo = np.array([0.001, 0.001, 0.01, 0.001, 0.0])
    hi = np.array([5.0, 5.0, 5.0, 20.0, 1.0])
    sols = [least_squares(residual, p, bounds=(lo, hi), max_nfev=4000) for p in starts]
    _P = min(sols, key=lambda q: float(np.dot(q.fun, q.fun))).x
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _P))