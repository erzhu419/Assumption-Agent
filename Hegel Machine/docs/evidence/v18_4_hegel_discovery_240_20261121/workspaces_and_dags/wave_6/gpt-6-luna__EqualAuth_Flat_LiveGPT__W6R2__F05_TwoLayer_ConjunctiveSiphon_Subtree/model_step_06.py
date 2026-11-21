import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _simulate(row, p):
    a, b, c, tau, vmax, rho = p
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'] or 0.0)
    u = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a * xv + b * zv + c * J, (0.52 * xv - zv) / tau
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
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - y
    starts = [np.array([0.3, 0.2, 1.0, 0.05, 4.0, 0.5]), np.array([1.0, 0.5, 2.0, 0.1, 5.0, 0.7])]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=([0.001,0.0,0.0,0.002,0.01,0.0],[10.0,10.0,30.0,2.0,100.0,1.5]), max_nfev=3000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _P = best.x
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _P))