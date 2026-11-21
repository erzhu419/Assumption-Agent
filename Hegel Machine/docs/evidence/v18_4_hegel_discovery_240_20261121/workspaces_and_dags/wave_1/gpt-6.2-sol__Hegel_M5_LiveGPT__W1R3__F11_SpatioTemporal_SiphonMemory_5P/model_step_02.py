import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')


def _simulate(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = data.T
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, math.ceil(float(np.max(H)) / 0.01))
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
    global _params
    rows = list(train_rows)
    data = np.array([[r[k] for k in _keys] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return _simulate(p, data) - y
    bounds = ([0.0, 0.0, 1.0, 0.05, 0.01], [20.0, 20.0, 15.0, 0.95, 10.0])
    best = least_squares(residual, [1.0, 1.0, 6.5, 0.703, 0.25], bounds=bounds, x_scale='jac', max_nfev=250)
    if np.sqrt(np.mean(best.fun**2)) > 1e-5:
        for tau0 in (0.07, 0.9):
            trial = least_squares(residual, [1.5, 1.5, 6.5, 0.703, tau0], bounds=bounds, x_scale='jac', max_nfev=250)
            if trial.cost < best.cost:
                best = trial
    _params = best.x
    return _params


def predict(row) -> float:
    data = np.array([[row[k] for k in _keys]], dtype=float)
    return float(_simulate(_params, data)[0])
