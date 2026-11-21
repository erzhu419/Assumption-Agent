import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _simulate(data, p):
    vmax, rho, a, b, tau = p
    u, x0, z0, s, H = data.T
    source = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = np.array([max(1, math.ceil(float(t) / 0.004)) for t in H])
    steps = H / n
    x, z = x0.copy(), z0.copy()

    def rhs(x, z):
        return -a * x + b * z + source, (0.52 * x - z) / tau

    for i in range(int(n.max())):
        h = np.where(i < n, steps, 0.0)
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        z += (h/6.0) * (l1 + 2*l2 + 2*l3 + l4)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = np.array([[r[k] for k in _KEYS] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        return _simulate(data, p) - y

    bounds = ([0.0, 0.0, 0.0, 0.0, 0.004], [np.inf, 1.0, np.inf, np.inf, np.inf])
    best = least_squares(residual, [4.0, 0.7, 1.0, 1.0, 0.3], bounds=bounds, x_scale='jac', max_nfev=250)
    if np.linalg.norm(best.fun) > 1e-5:
        for tau in (0.08, 1.0):
            trial = least_squares(residual, [4.0, 0.7, 1.0, 1.0, tau], bounds=bounds, x_scale='jac', max_nfev=250)
            if trial.cost < best.cost:
                best = trial
    _params = best.x

def predict(row):
    data = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_simulate(data, _params)[0])