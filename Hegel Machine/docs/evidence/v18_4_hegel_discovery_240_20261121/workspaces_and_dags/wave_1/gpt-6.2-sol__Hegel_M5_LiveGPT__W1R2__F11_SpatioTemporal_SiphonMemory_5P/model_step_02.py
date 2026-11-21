import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _endpoint(row, p):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = (float(row[k]) for k in ('u_now', 'x_now', 'mem_conv', 's_grad', 'H'))
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(64, math.ceil(abs(H) * max(a, b, 1.0 / tau) / 0.2))
    h = H / n
    def rhs(x, z):
        return -a * x + b * z + J, (0.52 * x - z) / tau
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        z += h * (l1 + 2*l2 + 2*l3 + l4) / 6.0
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])
    def residual(p):
        return np.array([_endpoint(row, p) for row in rows]) - y
    result = least_squares(residual, [1.0, 1.0, 6.5, 0.70, 0.5], bounds=([0.0, 0.0, 1.0, 0.05, 0.01], [np.inf, np.inf, 15.0, 0.95, np.inf]), x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x

def predict(row):
    return _endpoint(row, _params)
