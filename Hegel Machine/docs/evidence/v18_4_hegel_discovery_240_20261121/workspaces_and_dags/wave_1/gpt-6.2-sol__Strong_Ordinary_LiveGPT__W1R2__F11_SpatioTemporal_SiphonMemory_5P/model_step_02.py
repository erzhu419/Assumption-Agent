import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _forecast(p, row):
    a, b, tau, vmax, rho = p
    u = float(row['u_now'])
    g = float(row['s_grad'])
    h = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    drive = vmax * (1.0 - rho * g / (1.0 + g)) * u / (1.2 + u)
    n = max(1, math.ceil(h / min(0.004, tau)))
    dt = h / n

    def rhs(x, z):
        return drive - a*x + b*z, (0.52*x - z)/tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5*dt*kx, z + 0.5*dt*kz)
        mx, mz = rhs(x + 0.5*dt*lx, z + 0.5*dt*lz)
        nx, nz = rhs(x + dt*mx, z + dt*mz)
        x += dt*(kx + 2*lx + 2*mx + nx)/6.0
        z += dt*(kz + 2*lz + 2*mz + nz)/6.0
    return float(x)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])
    result = least_squares(
        lambda p: np.array([_forecast(p, row) for row in rows]) - y,
        x0=[1.0, 1.0, 0.5, 4.0, 0.7],
        bounds=([0.0, 0.0, 0.001, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, 1.0]),
        ftol=1e-11, xtol=1e-11, gtol=1e-11,
    )
    _params = result.x
    return _params


def predict(row):
    return _forecast(_params, row)
