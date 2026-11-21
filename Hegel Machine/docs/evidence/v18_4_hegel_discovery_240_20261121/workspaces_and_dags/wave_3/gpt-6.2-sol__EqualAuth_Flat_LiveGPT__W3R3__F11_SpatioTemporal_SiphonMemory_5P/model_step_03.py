import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _forecast(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, horizon = data.T
    u = u / 2.0
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, math.ceil(float(np.max(horizon)) / 0.01))
    h = horizon / n

    def rhs(x, z):
        return -a*x + b*z + flux, (0.52*x - z)/tau

    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + h*kx/2, z + h*kz/2)
        mx, mz = rhs(x + h*lx/2, z + h*lz/2)
        nx, nz = rhs(x + h*mx, z + h*mz)
        x = x + h*(kx + 2*lx + 2*mx + nx)/6
        z = z + h*(kz + 2*lz + 2*mz + nz)/6
    return x

def fit(train_rows):
    global _params
    data = np.array([[r[k] for k in _KEYS] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return _forecast(p, data) - y
    bounds = ([0, 0, 0, 0, 0.01], [np.inf, np.inf, np.inf, 1, np.inf])
    best = None
    for tau0 in (0.25, 1.0, 0.06):
        result = least_squares(residual, [0.5, 0.8, 3.0, 0.5, tau0], bounds=bounds, x_scale='jac', max_nfev=250)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x

def predict(row) -> float:
    data = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_forecast(_params, data)[0])