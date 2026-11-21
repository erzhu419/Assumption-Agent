import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _endpoint(p, row):
    a, b, vmax, tau, rho = p
    u = row['u_now']
    s = row['s_grad']
    x, z = row['x_now'], row['mem_conv']
    H = row['H']
    drive = vmax * (u / (1.2 + u)) * (1.0 - rho * s / (1.0 + s))
    n = max(1, math.ceil(H / 0.004))
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
    y = np.array([r['y'] for r in rows], dtype=float)

    def residuals(p):
        return np.array([_endpoint(p, r) for r in rows]) - y

    _params = least_squares(
        residuals, [1.5, 1.0, 5.0, 0.5, 0.7],
        bounds=([0, 0, 0, 0.01, 0], [np.inf, np.inf, np.inf, np.inf, 1]),
        ftol=1e-11, xtol=1e-11, gtol=1e-11,
    ).x

def predict(row) -> float:
    return _endpoint(_params, row)
