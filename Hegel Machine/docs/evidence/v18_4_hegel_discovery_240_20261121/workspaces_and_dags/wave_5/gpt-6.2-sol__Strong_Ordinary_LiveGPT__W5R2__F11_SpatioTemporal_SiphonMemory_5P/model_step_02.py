import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')

def _features(rows):
    u, x, z, s, H = np.asarray([[r[k] for k in _keys] for r in rows], dtype=float).T
    u = u / 0.005
    return u / (1.2 + u), x, z, s / (1.0 + s), H

def _solve(p, data):
    a, b, vmax, rho, tau = p
    drive, x, z, sink, H = data
    J = vmax * (1.0 - rho * sink) * drive
    n = max(1, math.ceil(float(np.max(H)) / 0.004 - 1e-12))
    h = H / n
    ht = h / tau
    for _ in range(n):
        k1x = h * (-a*x + b*z + J)
        k1z = ht * (0.52*x - z)
        x2, z2 = x + 0.5*k1x, z + 0.5*k1z
        k2x = h * (-a*x2 + b*z2 + J)
        k2z = ht * (0.52*x2 - z2)
        x3, z3 = x + 0.5*k2x, z + 0.5*k2z
        k3x = h * (-a*x3 + b*z3 + J)
        k3z = ht * (0.52*x3 - z3)
        x4, z4 = x + k3x, z + k3z
        k4x = h * (-a*x4 + b*z4 + J)
        k4z = ht * (0.52*x4 - z4)
        x = x + (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z = z + (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _features(rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    def residual(p):
        return _solve(p, data) - y
    best = None
    for tau0 in (0.15, 0.03, 0.65):
        result = least_squares(residual, (1.0, 1.0, 4.0, 0.7, tau0), bounds=([0, 0, 0, 0, 0.002], [30, 30, 100, 1, 10]), x_scale='jac', max_nfev=200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if np.sqrt(2.0 * best.cost / len(y)) < 1e-6:
            break
    _params = best.x
    return _params

def predict(row) -> float:
    return float(_solve(_params, _features([row]))[0])