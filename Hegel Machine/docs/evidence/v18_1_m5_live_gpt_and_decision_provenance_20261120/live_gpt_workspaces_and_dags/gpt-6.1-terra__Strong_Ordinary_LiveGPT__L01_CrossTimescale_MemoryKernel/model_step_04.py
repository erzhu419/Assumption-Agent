import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _integrate(rows, params):
    a, b, c, tau = params
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    n = max(1, int(math.ceil(float(np.max(h)) / 0.004)))
    dt = h / n
    def deriv(xv, zv):
        return -a * xv + b * zv + c * u, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5 * dt * k1x, z + 0.5 * dt * k1z)
        k3x, k3z = deriv(x + 0.5 * dt * k2x, z + 0.5 * dt * k2z)
        k4x, k4z = deriv(x + dt * k3x, z + dt * k3z)
        x = x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        z = z + dt * (k1z + 2 * k2z + 2 * k3z + k4z) / 6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _integrate(rows, p) - y,
                           x0=[1.0, 0.5, 1.0, 0.3],
                           bounds=([0.0, -10.0, -10.0, 0.02],
                                   [10.0, 10.0, 10.0, 10.0]),
                           max_nfev=500)
    _params = result.x

def predict(row):
    return float(_integrate([row], _params)[0])
