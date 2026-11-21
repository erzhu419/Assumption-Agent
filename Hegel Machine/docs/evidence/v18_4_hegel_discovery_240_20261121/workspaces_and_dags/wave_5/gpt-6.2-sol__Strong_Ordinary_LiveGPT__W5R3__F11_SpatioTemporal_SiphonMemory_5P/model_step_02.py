import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _inputs(rows):
    u, s, x, z, h = np.asarray([[r['u_now'], r['s_grad'], r['x_now'], r['mem_conv'], r['H']] for r in rows], dtype=float).T
    u = u / 2.0
    return u / (1.2 + u), s / (1.0 + s), x, z, h

def _solve(p, data, n):
    a, b, vmax, rho, tau = p
    base, sink, x, z, h = data
    j = vmax * (1.0 - rho * sink) * base
    dt = h / n
    def rhs(x, z):
        return -a * x + b * z + j, (0.52 * x - z) / tau
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * dt * k1, z + 0.5 * dt * l1)
        k3, l3 = rhs(x + 0.5 * dt * k2, z + 0.5 * dt * l2)
        k4, l4 = rhs(x + dt * k3, z + dt * l3)
        x = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z = z + dt / 6.0 * (l1 + 2.0 * l2 + 2.0 * l3 + l4)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _inputs(rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    n = max(1, math.ceil(float(np.max(data[-1])) / 0.004))
    best = None
    for tau0 in (0.25, 0.08, 0.9):
        result = least_squares(lambda p: _solve(p, data, n) - y, [0.6, 0.7, 4.0, 0.7, tau0], bounds=([0.0, 0.0, 0.0, 0.0, 0.005], [np.inf, np.inf, np.inf, 1.0, np.inf]), x_scale='jac', max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost <= 0.5 * len(y) * 1e-12:
            break
    _params = best.x

def predict(row) -> float:
    n = max(1, math.ceil(float(row['H']) / 0.004))
    return float(_solve(_params, _inputs([row]), n)[0])
