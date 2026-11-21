import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(p, row):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * prefix[0]['x']
        for left, right in zip(prefix, prefix[1:]):
            q = (right['t'] - left['t']) / tau
            w = -math.expm1(-q)
            ramp = 1.0 - w / q if q > 1e-5 else q * (0.5 - q / 6.0)
            z = (1.0 - w) * z + 0.52 * (w * left['x'] + ramp * (right['x'] - left['x']))
    else:
        z = float(z)

    drive = c * u
    def rhs(X, Z):
        return -a * X + b * Z + drive, (0.52 * X - Z) / tau

    n = max(1, math.ceil(H / 0.004))
    h = H / n
    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + h * kx / 2, z + h * kz / 2)
        mx, mz = rhs(x + h * lx / 2, z + h * lz / 2)
        nx, nz = rhs(x + h * mx, z + h * mz)
        x += h * (kx + 2 * lx + 2 * mx + nx) / 6
        z += h * (kz + 2 * lz + 2 * mz + nz) / 6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residuals(p):
        return np.array([_forecast(p, r) for r in rows]) - y
    _params = least_squares(residuals, [2.0, 1.0, 1.0, 0.5],
                            bounds=([0, 0, 0, 0.01], np.inf),
                            x_scale='jac', max_nfev=200).x
    return _params

def predict(row):
    return _forecast(_params, row)
