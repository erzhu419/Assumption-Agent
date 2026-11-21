import math
import numpy as np
from scipy.optimize import least_squares

_tau_scale = 0.05

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    u_n = float(row['u_now']) / _tau_scale
    s = float(row['s_grad'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return (-a_slow * xv + b_couple * zv + J, (0.52 * xv - zv) / tau_fast)
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _P
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - ys
    best = None
    for start in ([1.0, 0.7, 0.3, 0.2, 0.05], [2.0, 0.4, 0.8, 0.5, 0.1], [0.5, 0.9, 0.1, 0.1, 0.02]):
        result = least_squares(residual, np.asarray(start), bounds=([0.0, 0.0, 0.0, 0.0, 0.001], [20.0, 1.0, 10.0, 10.0, 2.0]), max_nfev=5000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _P = best.x
    return _P

def predict(row) -> float:
    return float(_simulate(row, _P))