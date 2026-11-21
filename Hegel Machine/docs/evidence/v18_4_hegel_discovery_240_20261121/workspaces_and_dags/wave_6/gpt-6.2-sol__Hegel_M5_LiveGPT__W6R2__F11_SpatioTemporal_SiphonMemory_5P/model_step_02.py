import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')

def _endpoint(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = data.T
    un = u / 0.005
    drive = vmax * (1.0 - rho*s/(1.0+s)) * un/(1.2+un)
    n = max(32, math.ceil(float(np.max(H)) / 0.01))
    h = H / n

    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z)/tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        z = z + (h/6.0)*(l1 + 2*l2 + 2*l3 + l4)
    return x

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    data = np.array([[r[k] for k in _KEYS] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _endpoint(p, data) - y,
                           [2.0, 1.5, 6.5, 0.7, 0.4],
                           bounds=([0, 0, 0, 0, 0.01], [30, 20, 30, 1, 10]),
                           x_scale='jac', gtol=1e-11)
    _PARAMS = result.x

def predict(row) -> float:
    data = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_endpoint(_PARAMS, data)[0])