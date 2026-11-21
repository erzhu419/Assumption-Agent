import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _forecast(p, data):
    a, b, tau, vmax, rho = p
    u, x, z, s, h = data.T
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, math.ceil(float(np.max(h)) / 0.004))
    dt = h / n

    def step(x, z):
        return dt * (-a*x + b*z + flux), dt * (0.52*x - z) / tau

    for _ in range(n):
        kx, kz = step(x, z)
        lx, lz = step(x + 0.5*kx, z + 0.5*kz)
        mx, mz = step(x + 0.5*lx, z + 0.5*lz)
        nx, nz = step(x + mx, z + mz)
        x = x + (kx + 2*lx + 2*mx + nx) / 6.0
        z = z + (kz + 2*lz + 2*mz + nz) / 6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = np.array([[r[k] for k in _KEYS] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    _params = least_squares(lambda p: _forecast(p, data) - y,
                            [1.0, 1.0, 0.2, 5.0, 0.7],
                            bounds=([0, 0, 0.01, 0, 0], [np.inf, np.inf, np.inf, np.inf, 1]),
                            ftol=1e-11, xtol=1e-11, gtol=1e-11).x
    return _params

def predict(row) -> float:
    data = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_forecast(_params, data)[0])