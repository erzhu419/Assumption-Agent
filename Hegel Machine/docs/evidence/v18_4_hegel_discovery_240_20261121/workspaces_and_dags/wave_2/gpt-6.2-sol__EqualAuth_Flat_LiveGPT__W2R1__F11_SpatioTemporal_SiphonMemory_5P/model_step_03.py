import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _endpoint(data, p):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = data.T
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(32, math.ceil(float(np.max(H)) * max(a + b, 1.52 / tau)))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + J, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * h * k1, z + 0.5 * h * l1)
        k3, l3 = rhs(x + 0.5 * h * k2, z + 0.5 * h * l2)
        k4, l4 = rhs(x + h * k3, z + h * l3)
        x = x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        z = z + h * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.asarray([[r[k] for k in _KEYS] for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _endpoint(X, p) - y,
                           [1.0, 1.0, 5.0, 0.5, 0.5],
                           bounds=([0.0, 0.0, 0.0, 0.0, 0.005],
                                   [np.inf, np.inf, np.inf, 1.0, np.inf]),
                           x_scale='jac', max_nfev=200)
    _params = result.x
    return _params

def predict(row) -> float:
    X = np.asarray([[row[k] for k in _KEYS]], dtype=float)
    return float(_endpoint(X, _params)[0])