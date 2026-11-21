import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, params):
    a, b, c, tau = params
    x = float(row['x_now'])
    u = float(row['u_now'])
    z = row.get('mem_conv')
    if z is None:
        prefix = sorted(row['raw_prefix'], key=lambda sample: sample['t'])
        z = 0.52 * prefix[0]['x']
        for left, right in zip(prefix, prefix[1:]):
            q = (right['t'] - left['t']) / tau
            decay = math.exp(-q)
            z = decay * z + 0.26 * q * (decay * left['x'] + right['x'])

    H = float(row['H'])
    n = max(1, math.ceil(H / 0.004))
    h = H / n
    drive = c * u

    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z)/tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + h*k1/2, z + h*l1/2)
        k3, l3 = rhs(x + h*k2/2, z + h*l2/2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        z += (h/6) * (l1 + 2*l2 + 2*l3 + l4)
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([row['y'] for row in rows], dtype=float)

    def residuals(params):
        return np.asarray([_forecast(row, params) for row in rows]) - y

    _params = least_squares(residuals, [1.5, 1.0, 1.0, 0.6], bounds=([0.0, 0.0, 0.0, 0.01], np.inf), x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10).x

def predict(row):
    return _forecast(row, _params)
