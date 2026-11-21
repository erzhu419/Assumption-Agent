import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')

def _forecast(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, horizon = data.T
    u = u / 2.0
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, math.ceil(float(np.max(horizon)) / 0.004))
    h = horizon / n

    def rhs(x, z):
        return -a * x + b * z + flux, (0.52 * x - z) / tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5 * h * kx, z + 0.5 * h * kz)
        mx, mz = rhs(x + 0.5 * h * lx, z + 0.5 * h * lz)
        nx, nz = rhs(x + h * mx, z + h * mz)
        x = x + (h / 6.0) * (kx + 2.0 * lx + 2.0 * mx + nx)
        z = z + (h / 6.0) * (kz + 2.0 * lz + 2.0 * mz + nz)
    return x

def fit(train_rows):
    global _params
    data = np.array([[r[k] for k in _keys] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        return _forecast(p, data) - y

    bounds = ([0.0, 0.0, 0.0, 0.0, 0.005], [np.inf, np.inf, np.inf, 1.0, np.inf])
    best = least_squares(residual, [0.8, 1.0, 4.0, 0.7, 0.25], bounds=bounds, x_scale='jac')
    if np.linalg.norm(best.fun) > 1e-5:
        for tau in (0.07, 1.0):
            trial = least_squares(residual, [0.8, 1.0, 4.0, 0.7, tau], bounds=bounds, x_scale='jac')
            if trial.cost < best.cost:
                best = trial
    _params = best.x
    return _params

def predict(row):
    data = np.array([[row[k] for k in _keys]], dtype=float)
    return float(_forecast(_params, data)[0])
