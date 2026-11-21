import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _readout(features, p):
    u, x, z, s, H = features.T
    a, b, vmax, rho, tau = p
    u_n = u / 0.05
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u_n / (1.2 + u_n)
    n = max(1, math.ceil(float(np.max(H)) / 0.005))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * h * k1, z + 0.5 * h * l1)
        k3, l3 = rhs(x + 0.5 * h * k2, z + 0.5 * h * l2)
        k4, l4 = rhs(x + h * k3, z + h * l3)
        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        z = z + h * (l1 + 2 * l2 + 2 * l3 + l4) / 6
    return x

def fit(train_rows):
    global _params
    X = np.array([[row[k] for k in _KEYS] for row in train_rows], dtype=float)
    y = np.array([row['y'] for row in train_rows], dtype=float)
    result = least_squares(lambda p: _readout(X, p) - y,
                           [2.0, 1.0, 6.5, 0.70, 0.4],
                           bounds=([0, 0, 0, 0, 0.005], [np.inf, np.inf, np.inf, 1, np.inf]),
                           x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x

def predict(row):
    X = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_readout(X, _params)[0])