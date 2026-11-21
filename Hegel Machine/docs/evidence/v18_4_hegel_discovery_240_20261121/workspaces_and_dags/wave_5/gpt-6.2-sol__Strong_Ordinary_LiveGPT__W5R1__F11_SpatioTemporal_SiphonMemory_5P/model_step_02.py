import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(p, row):
    a, b, tau, vmax, rho = map(float, p)
    u = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x, z, H = (float(row[k]) for k in ('x_now', 'mem_conv', 'H'))
    n = max(1, math.ceil(H / min(0.004, 0.5 * tau)))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * h * k1, z + 0.5 * h * l1)
        k3, l3 = rhs(x + 0.5 * h * k2, z + 0.5 * h * l2)
        k4, l4 = rhs(x + h * k3, z + h * l3)
        x += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        z += h * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])

    def residuals(p):
        return np.array([_forecast(p, row) for row in rows]) - y

    result = least_squares(residuals, [1.0, 1.0, 0.2, 5.0, 0.7], bounds=([0.0, 0.0, 0.004, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, 1.0]), ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x

def predict(row) -> float:
    return _forecast(_params, row)
