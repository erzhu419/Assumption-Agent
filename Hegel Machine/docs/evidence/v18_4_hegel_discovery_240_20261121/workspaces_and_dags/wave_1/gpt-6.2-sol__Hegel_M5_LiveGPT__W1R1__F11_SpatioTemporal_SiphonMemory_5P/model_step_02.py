import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _endpoint(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = data.T
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(80, math.ceil(float(np.max(np.abs(H))) / 0.01))
    h = H / n
    for _ in range(n):
        k1 = -a*x + b*z + drive
        l1 = (0.52*x - z) / tau
        k2 = k1 + 0.5*h*(-a*k1 + b*l1)
        l2 = l1 + 0.5*h*(0.52*k1 - l1)/tau
        k3 = k1 + 0.5*h*(-a*k2 + b*l2)
        l3 = l1 + 0.5*h*(0.52*k2 - l2)/tau
        k4 = k1 + h*(-a*k3 + b*l3)
        l4 = l1 + h*(0.52*k3 - l3)/tau
        x = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        z = z + (h/6.0)*(l1 + 2*l2 + 2*l3 + l4)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = np.array([[r[k] for k in _KEYS] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    _params = least_squares(lambda p: _endpoint(p, data) - y,
                            [1.0, 1.0, 6.5, 0.70, 0.25],
                            bounds=([0, 0, 0, 0, 0.01], [np.inf, np.inf, np.inf, 1, np.inf]),
                            x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11).x

def predict(row):
    data = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_endpoint(_params, data)[0])