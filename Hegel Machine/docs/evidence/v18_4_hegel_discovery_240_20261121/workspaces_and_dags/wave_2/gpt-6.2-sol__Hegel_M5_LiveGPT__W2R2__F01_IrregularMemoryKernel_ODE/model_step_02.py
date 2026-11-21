import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    H = float(row['H'])
    n = max(1, math.ceil(H / 0.004))
    z = row.get('mem_conv')
    history = None
    if z is None:
        prefix = row['raw_prefix']
        t = np.array([p['t'] for p in prefix], dtype=float)
        x = np.array([p['x'] for p in prefix], dtype=float)
        history = (t[-1] - t, np.diff(t), x)
    return float(row['x_now']), float(row['u_now']), n, H / n, z, history

def _forecast(p, data):
    a, b, c, tau = map(float, p)
    x, u, n, h, z, history = data
    if z is None:
        lag, widths, hist_x = history
        weights = np.exp(-lag / tau)
        v = weights * hist_x
        z = float(0.52 * (v[0] + np.dot(widths, v[:-1] + v[1:]) / (2.0 * tau)))
    else:
        z = float(z)
    drive = c * u
    inv_tau = 1.0 / tau
    half = 0.5 * h
    sixth = h / 6.0
    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z)*inv_tau
    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + half*kx, z + half*kz)
        mx, mz = rhs(x + half*lx, z + half*lz)
        nx, nz = rhs(x + h*mx, z + h*mz)
        x += sixth*(kx + 2.0*lx + 2.0*mx + nx)
        z += sixth*(kz + 2.0*lz + 2.0*mz + nz)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = [_prepare(row) for row in rows]
    y = np.array([row['y'] for row in rows], dtype=float)
    def residual(p):
        return np.array([_forecast(p, d) for d in data]) - y
    best = None
    for tau0 in (0.08, 0.3, 1.0):
        result = least_squares(residual, [1.7, 0.8, 1.0, tau0], bounds=([0.0, 0.0, 0.0, 0.015], [15.0, 15.0, 15.0, 10.0]), x_scale='jac', max_nfev=200)
        if best is None or result.cost < best.cost:
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_forecast(_params, _prepare(row)))
