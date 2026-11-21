import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    u_n = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def rhs(xv, zv):
        return -a_slow * xv + b_couple * zv + J, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = rhs(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = rhs(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in rows]) - y
    lower = [0.0, 0.0, 0.0, 0.0, 0.005]
    upper = [20.0, 1.0, 10.0, 10.0, 5.0]
    starts = [[2.0, 0.5, 0.5, 0.5, 0.2], [4.0, 0.7, 1.0, 1.0, 0.5], [1.0, 0.2, 0.2, 2.0, 1.0]]
    results = [least_squares(residual, p0, bounds=(lower, upper), max_nfev=5000) for p0 in starts]
    _PARAMS = min(results, key=lambda r: np.sum(r.fun**2)).x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _PARAMS))