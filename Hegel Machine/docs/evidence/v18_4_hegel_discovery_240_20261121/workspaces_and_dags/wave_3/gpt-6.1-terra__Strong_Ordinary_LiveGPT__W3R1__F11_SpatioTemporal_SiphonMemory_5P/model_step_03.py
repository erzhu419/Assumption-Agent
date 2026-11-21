import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(rows, p):
    a, b, vmax, rho, tau = p
    u = np.asarray([r['u_now'] for r in rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in rows], dtype=float)
    x = np.asarray([r['x_now'] for r in rows], dtype=float)
    z = np.asarray([r['mem_conv'] for r in rows], dtype=float)
    h = np.asarray([r['H'] for r in rows], dtype=float)
    if not len(rows):
        return x
    n = max(1, int(math.ceil(float(np.max(h)) / 0.004)))
    dt = h / n
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    def deriv(xv, zv):
        return -a * xv + b * zv + j, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5 * dt * k1x, z + 0.5 * dt * k1z)
        k3x, k3z = deriv(x + 0.5 * dt * k2x, z + 0.5 * dt * k2z)
        k4x, k4z = deriv(x + dt * k3x, z + dt * k3z)
        x = x + dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z = z + dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.02], [10.0, 10.0, 20.0, 1.0, 10.0])
    best = None
    for start in ([0.5, 0.8, 3.0, 0.7, 0.3], [1.0, 1.0, 4.0, 0.7, 1.0]):
        result = least_squares(lambda p: _evaluate(rows, p) - y, start, bounds=bounds, max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate([row], _params)[0])