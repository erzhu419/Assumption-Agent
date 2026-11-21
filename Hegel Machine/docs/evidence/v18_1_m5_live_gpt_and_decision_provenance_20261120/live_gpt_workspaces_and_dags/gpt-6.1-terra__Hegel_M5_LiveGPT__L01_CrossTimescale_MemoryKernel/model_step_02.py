import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(rows, params):
    a, b, c, tau = params
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    horizons = np.array([r['H'] for r in rows], dtype=float)
    n = max(1, int(math.ceil(float(np.max(np.abs(horizons))) / 0.004)))
    h = horizons / n
    def deriv(xv, zv):
        return -a * xv + b * zv + c * u, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + h*k1x/2, z + h*k1z/2)
        k3x, k3z = deriv(x + h*k2x/2, z + h*k2z/2)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x = x + h*(k1x + 2*k2x + 2*k3x + k4x)/6
        z = z + h*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _evaluate(rows, p) - y,
                           x0=[0.8, 0.5, 1.0, 0.5],
                           bounds=([0.0, -5.0, -5.0, 0.02], [10.0, 5.0, 5.0, 10.0]),
                           max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    return float(_evaluate([row], _params)[0])