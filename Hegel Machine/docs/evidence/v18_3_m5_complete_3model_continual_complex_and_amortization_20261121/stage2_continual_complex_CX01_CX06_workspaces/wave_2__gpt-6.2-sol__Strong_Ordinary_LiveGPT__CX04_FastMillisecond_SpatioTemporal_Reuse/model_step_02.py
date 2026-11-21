import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _rows(rows):
    return np.asarray([[float(row[key]) for key in _KEYS] for row in rows], dtype=float)

def _forecast(p, data):
    a, b, tau, vmax, rho = p
    u, x, z, s, H = data.T
    un = u / 0.005
    source = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    n = max(1, math.ceil(float(np.max(H)) / min(0.01, 0.5 * tau)))
    dt = H / n
    def rates(x, z):
        return -a * x + b * z + source, (0.52 * x - z) / tau
    for _ in range(n):
        kx, kz = rates(x, z)
        lx, lz = rates(x + 0.5 * dt * kx, z + 0.5 * dt * kz)
        mx, mz = rates(x + 0.5 * dt * lx, z + 0.5 * dt * lz)
        nx, nz = rates(x + dt * mx, z + dt * mz)
        x = x + dt * (kx + 2.0 * lx + 2.0 * mx + nx) / 6.0
        z = z + dt * (kz + 2.0 * lz + 2.0 * mz + nz) / 6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _rows(rows)
    y = np.asarray([float(row['y']) for row in rows])
    start = [1.5, 1.0, float(np.median(data[:, 4])), 6.0, 0.7]
    result = least_squares(lambda p: _forecast(p, data) - y, start,
                           bounds=([0, 0, 0.001, 0, 0], [np.inf, np.inf, np.inf, np.inf, 1]),
                           x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10)
    _params = result.x
    return _params

def predict(row):
    return float(_forecast(_params, _rows([row]))[0])