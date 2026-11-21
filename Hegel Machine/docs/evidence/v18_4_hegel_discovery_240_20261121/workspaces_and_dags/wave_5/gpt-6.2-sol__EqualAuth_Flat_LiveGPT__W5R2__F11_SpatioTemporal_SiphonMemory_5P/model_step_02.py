import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _forecast(p, rows):
    a, b, tau, vmax, rho = p
    u, x, z, s, horizon = rows.T
    un = u / 0.005
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    # Augment the state with a constant to include the siphon flux.
    A = np.zeros((len(rows), 3, 3))
    A[:, 0, 0] = -a
    A[:, 0, 1] = b
    A[:, 0, 2] = flux
    A[:, 1, 0] = 0.52 / tau
    A[:, 1, 1] = -1.0 / tau
    n = max(64, math.ceil(5.0 * float(np.max(np.abs(horizon))) * (a + b + 1.52 / tau)))
    M = A * (horizon / n)[:, None, None]
    # For a constant linear system this polynomial is one classical RK4 step.
    I = np.eye(3)
    step = I + M @ (I + M @ (I / 2 + M @ (I / 6 + M / 24)))
    transition = np.linalg.matrix_power(step, n)
    return transition[:, 0, 0] * x + transition[:, 0, 1] * z + transition[:, 0, 2]

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = np.array([[r[k] for k in _KEYS] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return _forecast(p, data) - y
    bounds = ([0., 0., 1e-5, 0., 0.], [np.inf, np.inf, np.inf, np.inf, 1.])
    best = None
    for tau in (0.15, 0.03, 0.6):
        result = least_squares(residual, [0.9, 1.2, tau, 2.5, 0.4], bounds=bounds, x_scale='jac', max_nfev=400, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if 2.0 * best.cost / len(y) < 1e-12:
            break
    _params = best.x

def predict(row):
    data = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_forecast(_params, data)[0])