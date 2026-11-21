import math
import numpy as np
from scipy.optimize import least_squares

_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _evaluate(p, data):
    a, b, tau, vmax, rho = p
    u, x, z, s, h = data.T
    un = u / 0.05
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    A = np.array([[-a, b, 1.0], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    # Compose a power-of-two number of equal RK4 steps by repeated squaring.
    size = np.max(np.abs(h)) * np.linalg.norm(A, ord=np.inf)
    k = max(0, math.ceil(math.log2(max(1.0, size / 0.05))))
    B = (h / (2 ** k))[:, None, None] * A
    I = np.eye(3)
    R = I + B @ (I + B @ (I / 2 + B @ (I / 6 + B / 24)))
    for _ in range(k):
        R = R @ R
    return np.sum(R[:, 0, :] * np.column_stack((x, z, flux)), axis=1)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = np.array([[r[key] for key in _keys] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    bounds = ([0, 0, 1e-4, 0, 0], [np.inf, np.inf, np.inf, np.inf, 1])
    def residual(p):
        return _evaluate(p, data) - y
    results = [least_squares(residual, [1.0, 1.0, tau, 4.0, 0.5], bounds=bounds, x_scale='jac', max_nfev=500, ftol=1e-11, xtol=1e-11, gtol=1e-11) for tau in (0.05, 0.2, 0.6)]
    _params = min(results, key=lambda r: r.cost).x

def predict(row) -> float:
    data = np.array([[row[key] for key in _keys]], dtype=float)
    return float(_evaluate(_params, data)[0])