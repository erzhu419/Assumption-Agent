import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None


def _evaluate(p, inputs):
    a, b, vmax, rho, tau = p
    u, x0, z0, s, horizon = inputs
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, math.ceil(float(np.max(horizon)) / 0.01))
    h = horizon / n
    x, z = x0.copy(), z0.copy()

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5 * h * kx, z + 0.5 * h * kz)
        mx, mz = rhs(x + 0.5 * h * lx, z + 0.5 * h * lz)
        nx, nz = rhs(x + h * mx, z + h * mz)
        x += (h / 6.0) * (kx + 2.0 * lx + 2.0 * mx + nx)
        z += (h / 6.0) * (kz + 2.0 * lz + 2.0 * mz + nz)
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    inputs = np.array([[r[k] for k in _KEYS] for r in rows], dtype=float).T
    target = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        return _evaluate(p, inputs) - target

    best = None
    for tau0 in (0.35, 0.1, 1.2):
        result = least_squares(residual, [1.2, 0.8, 6.5, 0.70, tau0],
                               bounds=([0, 0, 0, 0, 0.01], [np.inf, np.inf, np.inf, 1, np.inf]),
                               x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11,
                               max_nfev=250)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-11:
            break
    _params = best.x.copy()
    return _params


def predict(row) -> float:
    inputs = np.array([[row[k]] for k in _KEYS], dtype=float)
    return float(_evaluate(_params, inputs)[0])
