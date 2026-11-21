import math
import numpy as np
from scipy.optimize import least_squares

_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None


def _inputs(rows):
    return np.asarray([[float(row[key]) for key in _keys] for row in rows], dtype=float)


def _forecast(p, inputs):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = inputs.T
    un = u / 0.05
    J = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    n = max(128, math.ceil(float(np.max(np.abs(H))) / 0.005))
    h = H / n

    def rhs(x, z):
        return -a * x + b * z + J, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * h * k1, z + 0.5 * h * l1)
        k3, l3 = rhs(x + 0.5 * h * k2, z + 0.5 * h * l2)
        k4, l4 = rhs(x + h * k3, z + h * l3)
        x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z = z + h / 6.0 * (l1 + 2.0 * l2 + 2.0 * l3 + l4)
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = _inputs(rows)
    y = np.asarray([float(row['y']) for row in rows])
    bounds = ([0., 0., 0., 0., 0.004], [30., 30., 40., 1., 3.])

    def residual(p):
        return _forecast(p, X) - y

    best = None
    for tau0 in (0.08, 0.025, 0.35):
        result = least_squares(residual, [2., 1., 6.5, 0.70, tau0], bounds=bounds, x_scale='jac', max_nfev=400)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x
    return _params


def predict(row) -> float:
    return float(_forecast(_params, _inputs([row]))[0])
