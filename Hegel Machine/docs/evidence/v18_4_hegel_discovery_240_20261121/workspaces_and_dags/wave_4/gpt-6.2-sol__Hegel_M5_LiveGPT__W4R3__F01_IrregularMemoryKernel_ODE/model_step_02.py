import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _prepare(row):
    x = float(row['x_now'])
    u = float(row['u_now'])
    horizon = float(row['H'])
    n = max(1, math.ceil(horizon / 0.004))
    memory = row.get('mem_conv')
    hist = None
    if memory is None:
        prefix = row['raw_prefix']
        t_end = float(prefix[-1]['t'])
        lag = np.array([t_end - float(point['t']) for point in prefix])
        values = np.array([float(point['x']) for point in prefix])
        hist = (lag, values, lag[:-1] - lag[1:])
    return x, u, n, horizon / n, memory, hist


def _forecast(params, prepared):
    a, b, c, tau = map(float, params)
    x, u, n, h, memory, hist = prepared
    if h == 0.0:
        return x
    if memory is None:
        lag, values, intervals = hist
        weighted = np.exp(-lag / tau) * values
        z = 0.52 * (weighted[0] + 0.5 * np.dot(intervals, weighted[:-1] + weighted[1:]) / tau)
    else:
        z = float(memory)
    drive = c * u
    inv_tau = 1.0 / tau
    half = h / 2.0
    sixth = h / 6.0
    for _ in range(n):
        kx = -a*x + b*z + drive
        kz = (0.52*x - z) * inv_tau
        x2, z2 = x + half*kx, z + half*kz
        lx = -a*x2 + b*z2 + drive
        lz = (0.52*x2 - z2) * inv_tau
        x3, z3 = x + half*lx, z + half*lz
        mx = -a*x3 + b*z3 + drive
        mz = (0.52*x3 - z3) * inv_tau
        x4, z4 = x + h*mx, z + h*mz
        nx = -a*x4 + b*z4 + drive
        nz = (0.52*x4 - z4) * inv_tau
        x += sixth * (kx + 2.0*(lx + mx) + nx)
        z += sixth * (kz + 2.0*(lz + mz) + nz)
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    prepared = [_prepare(row) for row in rows]
    y = np.array([float(row['y']) for row in rows])

    def residual(params):
        return np.array([_forecast(params, row) for row in prepared]) - y

    best = None
    for tau0 in (0.22, 0.07, 0.65):
        result = least_squares(residual, (1.3, 0.8, 1.3, tau0), bounds=([0.0, 0.0, 0.0, 0.012], [20.0, 20.0, 20.0, 8.0]), x_scale='jac', max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost <= 0.5 * len(rows) * (2e-6)**2:
            break
    _params = best.x


def predict(row):
    return _forecast(_params, _prepare(row))
