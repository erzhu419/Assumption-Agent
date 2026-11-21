import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _evaluate(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, g, horizon = data.T
    u = u / 2.0
    source = vmax * (1.0 - rho * g / (1.0 + g)) * u / (1.2 + u)
    n = max(32, math.ceil(float(np.max(np.abs(horizon))) / 0.01))
    dt = horizon / n

    def rhs(x, z):
        return -a * x + b * z + source, (0.52 * x - z) / tau

    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5 * dt * k1x, z + 0.5 * dt * k1z)
        k3x, k3z = rhs(x + 0.5 * dt * k2x, z + 0.5 * dt * k2z)
        k4x, k4z = rhs(x + dt * k3x, z + dt * k3z)
        x = x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        z = z + dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = np.asarray([[r[k] for k in _KEYS] for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    best = None
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.01], [20.0, 20.0, 30.0, 1.0, np.inf])
    for tau0 in (0.25, 0.08, 0.8):
        result = least_squares(lambda p: _evaluate(p, data) - y,
                               [0.7, 1.0, 6.5, 0.70, tau0],
                               bounds=bounds, x_scale='jac', max_nfev=250,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or result.cost < best.cost:
            best = result
        if np.sqrt(2.0 * best.cost / len(y)) < 1e-6:
            break
    _params = best.x

def predict(row):
    data = np.asarray([[row[k] for k in _KEYS]], dtype=float)
    return float(_evaluate(_params, data)[0])