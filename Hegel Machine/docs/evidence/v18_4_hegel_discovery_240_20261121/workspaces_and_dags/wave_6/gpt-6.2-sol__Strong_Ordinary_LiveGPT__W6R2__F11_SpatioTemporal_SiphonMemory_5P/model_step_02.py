import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    vmax, rho, a, b, tau = p
    u = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, math.ceil(H / 0.004))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * h * k1, z + 0.5 * h * l1)
        k3, l3 = rhs(x + 0.5 * h * k2, z + 0.5 * h * l2)
        k4, l4 = rhs(x + h * k3, z + h * l3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        z += h * (l1 + 2 * l2 + 2 * l3 + l4) / 6.0
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])

    def residuals(p):
        return np.array([_forecast(row, p) for row in rows]) - y

    _params = least_squares(residuals, [5.0, 0.7, 1.0, 1.0, 0.4],
                            bounds=([0.0, 0.0, 0.0, 0.0, 0.004],
                                    [np.inf, 1.0, np.inf, np.inf, np.inf]),
                            x_scale='jac', gtol=1e-12).x
    return _params

def predict(row) -> float:
    return _forecast(row, _params)
