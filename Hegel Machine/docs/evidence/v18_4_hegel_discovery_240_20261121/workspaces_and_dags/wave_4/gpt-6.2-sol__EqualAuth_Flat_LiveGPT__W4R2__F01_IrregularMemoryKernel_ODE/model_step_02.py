import math
import numpy as np
from scipy.optimize import least_squares


def _forecast(row, params):
    decay, coupling, drive, tau = params
    a = decay + 0.52 * coupling
    x = float(row['x_now'])
    z = row.get('mem_conv')
    if z is None:
        prefix = sorted(row.get('raw_prefix') or [], key=lambda p: p['t'])
        z = 0.52 * (float(prefix[0]['x']) if prefix else x)
        # Integrate the exponential kernel against piecewise-linear x.
        for left, right in zip(prefix, prefix[1:]):
            r = (right['t'] - left['t']) / tau
            if r <= 0.0:
                continue
            w = -math.expm1(-r)
            ramp = r * (0.5 - r / 6.0) if r < 1e-5 else 1.0 - w / r
            z = math.exp(-r) * z + 0.52 * (w * left['x'] + ramp * (right['x'] - left['x']))
    forcing = drive * float(row['u_now'])
    horizon = float(row['H'])
    n = max(1, math.ceil(horizon / 0.004))
    dt = horizon / n

    def rhs(x, z):
        return -a * x + coupling * z + forcing, (0.52 * x - z) / tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5 * dt * kx, z + 0.5 * dt * kz)
        mx, mz = rhs(x + 0.5 * dt * lx, z + 0.5 * dt * lz)
        nx, nz = rhs(x + dt * mx, z + dt * mz)
        x += dt * (kx + 2.0 * lx + 2.0 * mx + nx) / 6.0
        z += dt * (kz + 2.0 * lz + 2.0 * mz + nz) / 6.0
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([row['y'] for row in rows], dtype=float)
    def residual(params):
        return np.array([_forecast(row, params) for row in rows]) - y
    _params = least_squares(residual, [1.3, 1.3, 1.2, 0.6], bounds=([1e-6, 0.0, 0.0, 0.02], [np.inf, np.inf, np.inf, np.inf]), x_scale='jac').x
    return _params


def predict(row):
    return _forecast(row, _params)
