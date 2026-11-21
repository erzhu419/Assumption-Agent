import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    z = row.get('mem_conv')
    prefix = None
    if z is None:
        samples = row['raw_prefix']
        prefix = (0.52 * float(samples[0]['x']), [
            (float(right['t']) - float(left['t']), float(left['x']), float(right['x']))
            for left, right in zip(samples, samples[1:])
            if float(right['t']) > float(left['t'])
        ])
    return float(row['x_now']), float(row['u_now']), float(row['H']), z, prefix

def _forecast(data, params):
    a, b, c, tau = map(float, params)
    x, u, horizon, z, prefix = data
    if horizon == 0.0:
        return x
    if z is None:
        z, intervals = prefix
        for dt, left, right in intervals:
            r = dt / tau
            q = -math.expm1(-r)
            # Exact exponential-kernel update with linear x on this interval.
            w = r * (0.5 - r / 6.0 + r * r / 24.0) if r < 1e-4 else 1.0 - q / r
            z += 0.52 * (q * left + w * (right - left)) - q * z
    else:
        z = float(z)
    n = max(1, math.ceil(horizon / 0.004))
    h = horizon / n
    drive = c * u
    inv_tau = 1.0 / tau

    def rhs(xx, zz):
        return -a * xx + b * zz + drive, (0.52 * xx - zz) * inv_tau

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
    data = [_prepare(row) for row in train_rows]
    y = np.asarray([row['y'] for row in train_rows], dtype=float)

    def residuals(params):
        return np.asarray([_forecast(row, params) for row in data]) - y

    result = least_squares(residuals, [1.6, 1.0, 1.1, 0.5],
                           bounds=([0.0, 0.0, 0.0, 0.01], np.inf),
                           x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10)
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_forecast(_prepare(row), _params))