import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _simulate(data, p):
    u, x, z, s, h = data.T
    a, b, tau, vmax, rho = p
    source = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    steps = max(1, math.ceil(float(np.max(h)) / 0.01))
    dt = h / steps

    def slope(x, z):
        return -a * x + b * z + source, (0.52 * x - z) / tau

    for _ in range(steps):
        kx, kz = slope(x, z)
        lx, lz = slope(x + 0.5 * dt * kx, z + 0.5 * dt * kz)
        mx, mz = slope(x + 0.5 * dt * lx, z + 0.5 * dt * lz)
        nx, nz = slope(x + dt * mx, z + dt * mz)
        x = x + dt * (kx + 2 * lx + 2 * mx + nx) / 6
        z = z + dt * (kz + 2 * lz + 2 * mz + nz) / 6
    return x

def fit(train_rows):
    global _params
    data = np.array([[r[k] for k in _KEYS] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: _simulate(data, p) - y,
                           [1.0, 1.0, 0.5, 4.0, 0.5],
                           bounds=([0, 0, 0.01, 0, 0], [20, np.inf, np.inf, np.inf, 1]),
                           ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params

def predict(row) -> float:
    data = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_simulate(data, _params)[0])