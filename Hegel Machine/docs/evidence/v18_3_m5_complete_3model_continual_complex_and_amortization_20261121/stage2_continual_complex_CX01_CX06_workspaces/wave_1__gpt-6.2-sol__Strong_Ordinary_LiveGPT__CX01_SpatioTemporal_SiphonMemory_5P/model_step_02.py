import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(p, row):
    a, b, tau, vmax, rho = map(float, p)
    u = float(row['u_now'])
    s = float(row['s_grad'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, math.ceil(H / 0.004))
    h = H / n

    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z)/tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        z += (h/6.0)*(l1 + 2*l2 + 2*l3 + l4)
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([row['y'] for row in rows], dtype=float)

    def residual(p):
        return np.array([_forecast(p, row) for row in rows]) - y

    result = least_squares(residual, [1.0, 1.0, 1.0, 5.0, 0.65],
                           bounds=([0.0, 0.0, 0.01, 0.0, 0.0],
                                   [np.inf, np.inf, np.inf, np.inf, 1.0]),
                           x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params

def predict(row) -> float:
    return _forecast(_params, row)
