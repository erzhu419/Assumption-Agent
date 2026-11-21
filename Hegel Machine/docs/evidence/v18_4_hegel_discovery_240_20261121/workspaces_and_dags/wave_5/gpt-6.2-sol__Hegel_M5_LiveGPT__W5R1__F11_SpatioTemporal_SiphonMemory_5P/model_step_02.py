import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')

def _inputs(rows):
    return np.asarray([[r[k] for k in _keys] for r in rows], dtype=float).T

def _endpoint(p, inputs):
    a, b, vmax, rho, tau = p
    u, x0, z0, s, H = inputs
    un = u / 2.0
    J = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    n = max(1, math.ceil(float(np.max(H)) / 0.01))
    dt = H / n
    x, z = x0.copy(), z0.copy()

    def rhs(x, z):
        return -a*x + b*z + J, (0.52*x - z)/tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*dt*k1, z + 0.5*dt*l1)
        k3, l3 = rhs(x + 0.5*dt*k2, z + 0.5*dt*l2)
        k4, l4 = rhs(x + dt*k3, z + dt*l3)
        x += (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        z += (dt/6.0)*(l1 + 2*l2 + 2*l3 + l4)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    inputs = _inputs(rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)

    def residual(p):
        return _endpoint(p, inputs) - y

    bounds = ([0, 0, 0, 0, 0.01], [np.inf, np.inf, np.inf, 1, np.inf])
    best = None
    for tau0 in (0.25, 0.08, 1.0):
        result = least_squares(residual, [1.0, 1.0, 6.5, 0.70, tau0],
                               bounds=bounds, x_scale='jac', max_nfev=250,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x

def predict(row) -> float:
    return float(_endpoint(_params, _inputs([row]))[0])