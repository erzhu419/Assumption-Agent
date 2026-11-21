import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, params):
    a, b, c, tau = map(float, params)
    x = float(row['x_now'])
    u = float(row['u_now'])
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * float(prefix[0]['x'])
        for left, right in zip(prefix, prefix[1:]):
            q = (float(right['t']) - float(left['t'])) / tau
            decay = math.exp(-q)
            z = decay * z + 0.26 * q * (decay * float(left['x']) + float(right['x']))
    else:
        z = float(z)

    horizon = float(row['H'])
    steps = max(1, math.ceil(horizon / 0.004))
    h = horizon / steps

    def rhs(x, z):
        return -a * x + b * z + c * u, (0.52 * x - z) / tau

    for _ in range(steps):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        z += (h/6) * (l1 + 2*l2 + 2*l3 + l4)
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    targets = np.array([float(row['y']) for row in rows])

    def residual(params):
        return np.array([_forecast(row, params) for row in rows]) - targets

    result = least_squares(residual, [2.0, 1.0, 1.3, 0.65], bounds=([0, 0, 0, 0.02], [np.inf]*4), x_scale='jac')
    _params = result.x

def predict(row) -> float:
    return _forecast(row, _params)
