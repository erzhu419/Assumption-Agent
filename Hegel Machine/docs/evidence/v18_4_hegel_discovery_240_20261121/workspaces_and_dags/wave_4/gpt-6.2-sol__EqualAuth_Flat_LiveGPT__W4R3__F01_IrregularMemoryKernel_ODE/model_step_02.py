import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    z = row['mem_conv']
    prefix = None
    if z is None:
        samples = sorted(row['raw_prefix'], key=lambda s: s['t'])
        t = np.array([s['t'] for s in samples], dtype=float)
        x = np.array([s['x'] for s in samples], dtype=float)
        prefix = (t[-1] - t, x)
    return float(row['x_now']), float(row['u_now']), float(row['H']), z, prefix

def _initial_memory(z, prefix, tau):
    if z is not None:
        return float(z)
    lag, x = prefix
    kernel = np.exp(-lag / tau)
    f = x * kernel
    return 0.52 * (kernel[0] * x[0] + np.dot(lag[:-1] - lag[1:], f[:-1] + f[1:]) / (2 * tau))

def _forecast(p, rows, x0, u, h, n):
    a, b, c, tau = p
    x = x0.copy()
    z = np.array([_initial_memory(r[3], r[4], tau) for r in rows])
    drive = c * u
    d = h / n
    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau
    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5 * d * kx, z + 0.5 * d * kz)
        mx, mz = rhs(x + 0.5 * d * lx, z + 0.5 * d * lz)
        nx, nz = rhs(x + d * mx, z + d * mz)
        x += (d / 6) * (kx + 2 * lx + 2 * mx + nx)
        z += (d / 6) * (kz + 2 * lz + 2 * mz + nz)
    return x

def _batch(rows):
    x, u, h = np.array([r[:3] for r in rows], dtype=float).T
    return rows, x, u, h, max(1, math.ceil(float(h.max()) / 0.004))

def fit(train_rows):
    global _params
    train_rows = list(train_rows)
    batch = _batch([_prepare(r) for r in train_rows])
    y = np.array([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: _forecast(p, *batch) - y, [1.5, 1.0, 1.5, 0.7], bounds=([0, 0, 0, 0.02], [np.inf] * 4), x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10)
    _params = result.x

def predict(row) -> float:
    return float(_forecast(_params, *_batch([_prepare(row)]))[0])