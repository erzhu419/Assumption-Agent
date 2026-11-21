import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')

def _readout(p, u, x, z, s, horizon):
    a, b, vmax, rho, tau = p
    un = u / 2.0
    source = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    n = max(16, math.ceil(horizon / 0.0125))
    h = horizon / n
    half = h / 2.0
    for _ in range(n):
        x1 = -a*x + b*z + source
        z1 = (0.52*x - z) / tau
        x2 = x1 + half*(-a*x1 + b*z1)
        z2 = z1 + half*(0.52*x1 - z1) / tau
        x3 = x1 + half*(-a*x2 + b*z2)
        z3 = z1 + half*(0.52*x2 - z2) / tau
        x4 = x1 + h*(-a*x3 + b*z3)
        z4 = z1 + h*(0.52*x3 - z3) / tau
        x += h*(x1 + 2.0*x2 + 2.0*x3 + x4) / 6.0
        z += h*(z1 + 2.0*z2 + 2.0*z3 + z4) / 6.0
    return x

def fit(train_rows):
    global _params
    rows = [(tuple(float(row[k]) for k in _keys), float(row['y'])) for row in train_rows]
    def residual(p):
        return np.array([_readout(p, *features) - y for features, y in rows])
    result = least_squares(residual, [0.8, 0.8, 6.5, 0.7, 0.3], bounds=([0, 0, 0, 0, 0.01], [20, 20, 30, 1, 10]), x_scale='jac')
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_readout(_params, *(float(row[k]) for k in _keys)))