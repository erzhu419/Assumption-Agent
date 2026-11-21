import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def forecast(row, params):
    a, b, c, tau = map(float, params)
    x = float(row['x_now'])
    drive = c * float(row['u_now'])
    H = float(row['H'])
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * prefix[0]['x']
        for left, right in zip(prefix, prefix[1:]):
            r = (right['t'] - left['t']) / tau
            w = math.exp(-r)
            z = w*z + 0.26*r*(w*left['x'] + right['x'])
    z = float(z)
    n = max(1, math.ceil(H / min(0.004, 0.25*tau)))
    h = H / n
    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z)/tau
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + h*k1/2, z + h*l1/2)
        k3, l3 = rhs(x + h*k2/2, z + h*l2/2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += h*(k1 + 2*k2 + 2*k3 + k4)/6
        z += h*(l1 + 2*l2 + 2*l3 + l4)/6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([row['y'] for row in rows], dtype=float)
    def residuals(params):
        return np.array([forecast(row, params) for row in rows]) - y
    result = least_squares(residuals, [1.5, 1.0, 1.5, 0.5], bounds=([0, 0, 0, 0.02], np.inf), x_scale='jac')
    _params = result.x
    return _params

def predict(row):
    return forecast(row, _params)
