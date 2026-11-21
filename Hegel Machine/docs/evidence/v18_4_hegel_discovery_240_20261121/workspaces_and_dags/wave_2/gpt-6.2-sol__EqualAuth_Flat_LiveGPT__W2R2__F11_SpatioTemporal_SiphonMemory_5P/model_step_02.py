import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _simulate(p, rows):
    a, b, tau, vmax, rho = p
    u, x, z, s, H = rows.T
    u = u / 2.0
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, math.ceil(float(np.max(H)) / 0.005))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + J, (0.52 * x - z) / tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5 * h * kx, z + 0.5 * h * kz)
        mx, mz = rhs(x + 0.5 * h * lx, z + 0.5 * h * lz)
        nx, nz = rhs(x + h * mx, z + h * mz)
        x = x + h * (kx + 2.0 * lx + 2.0 * mx + nx) / 6.0
        z = z + h * (kz + 2.0 * lz + 2.0 * mz + nz) / 6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.asarray([[r[k] for k in _KEYS] for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    bounds = ([0.0, 0.0, 0.01, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, 1.0])
    best = None
    for tau0 in (0.35, 0.08, 1.5):
        result = least_squares(lambda p: _simulate(p, X) - y, [0.6, 0.8, tau0, 3.0, 0.5], bounds=bounds, x_scale='jac', max_nfev=250)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x
    return _params.copy()

def predict(row) -> float:
    X = np.asarray([[row[k] for k in _KEYS]], dtype=float)
    return float(_simulate(_params, X)[0])