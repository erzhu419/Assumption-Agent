import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(rows, p):
    a, b, vmax, rho, tau = p
    u = np.array([r['u_now'] / 0.005 for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    n = max(1, int(np.ceil(np.max(h) / min(0.004, tau * 0.5))))
    dt = h / n
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    def deriv(xx, zz):
        return -a * xx + b * zz + j, (0.52 * xx - zz) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x = x + dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z = z + dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    target = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return _evaluate(rows, p) - target
    best = None
    for tau in (0.03, 0.15, 0.6):
        result = least_squares(residual, [0.8, 0.8, 3.0, 0.7, tau], bounds=([0, 0, 0, 0, 0.003], [20, 20, 50, 2, 5]), max_nfev=500, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate([row], _params)[0])