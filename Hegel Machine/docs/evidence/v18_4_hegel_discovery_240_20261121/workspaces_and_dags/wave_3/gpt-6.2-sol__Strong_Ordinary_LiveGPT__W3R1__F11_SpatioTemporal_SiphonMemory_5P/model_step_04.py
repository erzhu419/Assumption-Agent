import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _inputs(rows):
    keys = ('x_now', 'mem_conv', 'u_now', 's_grad', 'H')
    return np.asarray([[r[k] for k in keys] for r in rows], dtype=float).T

def _forecast(p, data):
    a, b, tau, vmax, rho = p
    x, z, u, s, h = data
    source = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, math.ceil(float(np.max(h)) / 0.004))
    dt = h / n
    q = dt / tau
    for _ in range(n):
        kx = dt * (-a*x + b*z + source)
        kz = q * (0.52*x - z)
        xm, zm = x + 0.5*kx, z + 0.5*kz
        lx = dt * (-a*xm + b*zm + source)
        lz = q * (0.52*xm - zm)
        xm, zm = x + 0.5*lx, z + 0.5*lz
        mx = dt * (-a*xm + b*zm + source)
        mz = q * (0.52*xm - zm)
        xe, ze = x + mx, z + mz
        nx = dt * (-a*xe + b*ze + source)
        nz = q * (0.52*xe - ze)
        x = x + (kx + 2.0*lx + 2.0*mx + nx) / 6.0
        z = z + (kz + 2.0*lz + 2.0*mz + nz) / 6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _inputs(rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    best = None
    for tau0 in (0.25, 0.06, 0.9):
        result = least_squares(lambda p: _forecast(p, data) - y,
                               [0.7, 1.0, tau0, 4.0, 0.7],
                               bounds=([0.0, 0.0, 0.005, 0.0, 0.0],
                                       [np.inf, np.inf, np.inf, np.inf, 1.0]),
                               x_scale='jac', max_nfev=200)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x
    return _params

def predict(row):
    return float(_forecast(_params, _inputs([row]))[0])