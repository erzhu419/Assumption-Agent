import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / 0.05
    sg = float(row['s_grad'])
    J = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    dtmax = 0.004
    n = max(1, int(math.ceil(H / dtmax)))
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
    global _params
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in rows]) - y
    lo = [0.0, 0.0, 0.0, 0.0, 0.002]
    hi = [20.0, 1.5, 10.0, 10.0, 2.0]
    starts = ([1.0, 0.7, 0.5, 0.3, 0.05], [2.0, 0.5, 1.0, 0.8, 0.1], [0.5, 0.9, 0.2, 0.1, 0.02])
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=5000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _params))