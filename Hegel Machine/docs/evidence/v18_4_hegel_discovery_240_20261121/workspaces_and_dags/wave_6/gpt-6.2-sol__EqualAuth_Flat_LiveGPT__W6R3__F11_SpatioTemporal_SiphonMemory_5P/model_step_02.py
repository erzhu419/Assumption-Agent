import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _matrix(rows):
    return np.asarray([[row[k] for k in _KEYS] for row in rows], dtype=float).T

def _forecast(data, p):
    a_slow, b_couple, tau_fast, vmax, rho = p
    u, x, z, s, h = data
    u_n = u / 2.0
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u_n / (1.2 + u_n)
    n = max(48, math.ceil(128.0 * float(np.max(h))))
    dt = h / n

    def rhs(x, z):
        return -a_slow * x + b_couple * z + j, (0.52 * x - z) / tau_fast

    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5 * dt * k1x, z + 0.5 * dt * k1z)
        k3x, k3z = rhs(x + 0.5 * dt * k2x, z + 0.5 * dt * k2z)
        k4x, k4z = rhs(x + dt * k3x, z + dt * k3z)
        x = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        z = z + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _matrix(rows)
    y = np.asarray([row['y'] for row in rows], dtype=float)
    bounds = ([0.0, 0.0, 0.02, 0.0, 0.0], [10.0, 10.0, 10.0, 40.0, 1.0])
    best = None
    for tau0 in (0.35, 1.2, 0.08):
        result = least_squares(lambda p: _forecast(data, p) - y,
                               [0.5, 0.8, tau0, 3.0, 0.5], bounds=bounds,
                               x_scale='jac', max_nfev=250,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or result.cost < best.cost:
            best = result
        if math.sqrt(2.0 * best.cost / len(rows)) < 1e-5:
            break
    _params = best.x

def predict(row) -> float:
    return float(_forecast(_matrix([row]), _params)[0])