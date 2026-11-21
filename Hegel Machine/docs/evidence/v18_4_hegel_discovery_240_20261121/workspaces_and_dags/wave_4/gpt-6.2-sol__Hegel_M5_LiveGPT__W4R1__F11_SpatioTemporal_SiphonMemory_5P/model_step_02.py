import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _pack(rows):
    return np.array([[r[k] for k in _KEYS] for r in rows], dtype=float).T

def _simulate(p, data, steps):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = data
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    h = H / steps
    def rhs(x, z):
        return -a*x + b*z + J, (0.52*x - z)/tau
    for _ in range(steps):
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
    data = _pack(rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    steps = max(48, math.ceil(float(np.max(np.abs(data[-1]))) / 0.01))
    def residual(p):
        return _simulate(p, data, steps) - y
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.005], [20.0, 30.0, 30.0, 1.0, 20.0])
    best = None
    for start in ([1.5, 1.0, 6.5, 0.70, 0.2], [2.5, 2.0, 8.0, 0.72, 0.6], [1.0, 3.0, 4.0, 0.5, 0.06]):
        result = least_squares(residual, start, bounds=bounds, x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=250)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 0.5 * y.size * 1e-10:
            break
    _params = best.x

def predict(row):
    steps = max(48, math.ceil(abs(float(row['H'])) / 0.01))
    return float(_simulate(_params, _pack([row]), steps)[0])