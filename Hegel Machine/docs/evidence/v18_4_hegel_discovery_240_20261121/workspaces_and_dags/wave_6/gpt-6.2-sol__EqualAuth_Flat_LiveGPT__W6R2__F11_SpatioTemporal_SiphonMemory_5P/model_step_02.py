import math
import numpy as np
from scipy.optimize import least_squares

_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _endpoint(data, params):
    a, b, tau, vmax, rho = params
    u, x, z, s, H = data.T
    u = u / 0.005
    source = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(80, math.ceil(float(np.max(H)) / tau))
    h = H / n

    def increment(x, z):
        return h * (-a*x + b*z + source), (h/tau) * (0.52*x - z)

    for _ in range(n):
        kx, kz = increment(x, z)
        lx, lz = increment(x + 0.5*kx, z + 0.5*kz)
        mx, mz = increment(x + 0.5*lx, z + 0.5*lz)
        nx, nz = increment(x + mx, z + mz)
        x = x + (kx + 2*lx + 2*mx + nx) / 6.0
        z = z + (kz + 2*lz + 2*mz + nz) / 6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = np.array([[r[k] for k in _keys] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _endpoint(data, p) - y,
                           [0.6, 0.8, 0.15, 4.0, 0.4],
                           bounds=([0, 0, 0.001, 0, 0], [np.inf, np.inf, np.inf, np.inf, 1]),
                           x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params

def predict(row):
    data = np.array([[row[k] for k in _keys]], dtype=float)
    return float(_endpoint(data, _params)[0])