import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _prepare(row):
    z = row.get('mem_conv')
    history = None
    if z is None:
        prefix = row['raw_prefix']
        t = np.array([p['t'] for p in prefix], dtype=float)
        x = np.array([p['x'] for p in prefix], dtype=float)
        history = (t[-1] - t, np.diff(t), x)
    H = float(row['H'])
    n = max(1, math.ceil(H / 0.004))
    return float(row['x_now']), float(row['u_now']), z, history, n, H / n


def _forecast(p, row):
    a, b, c, tau = p
    x, u, z, history, n, h = row
    if z is None:
        lag, dt, hx = history
        w = np.exp(-lag / tau)
        wx = w * hx
        z = 0.52 * (wx[0] + np.dot(dt, wx[:-1] + wx[1:]) / (2 * tau))
    # Apply the RK4 step matrix for the affine system (x, z, 1).
    M = h * np.array([[-a, b, c * u], [0.52 / tau, -1 / tau, 0.0], [0.0, 0.0, 0.0]])
    I = np.eye(3)
    R = I + M @ (I + M @ (I / 2 + M @ (I / 6 + M / 24)))
    return float(np.linalg.matrix_power(R, n)[0] @ np.array([x, z, 1.0]))


def fit(train_rows):
    global _params
    rows = [_prepare(r) for r in train_rows]
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        return np.array([_forecast(p, r) for r in rows]) - y

    best = None
    for tau0 in (0.25, 0.08, 0.8):
        result = least_squares(residual, [1.6, 0.8, 1.2, tau0], bounds=([0, 0, 0, 0.004], [np.inf] * 4), x_scale='jac', max_nfev=400, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 0.5 * len(rows) * 1e-10:
            break
    _params = best.x


def predict(row) -> float:
    return _forecast(_params, _prepare(row))
