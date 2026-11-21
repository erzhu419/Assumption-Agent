import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(rows):
    state = np.array([[r['x_now'], r['u_now'], r['H']] for r in rows], dtype=float).T
    memory = []
    for row in rows:
        z = row.get('mem_conv')
        if z is not None:
            memory.append(float(z))
        else:
            prefix = row['raw_prefix']
            t = np.array([p['t'] for p in prefix], dtype=float)
            x = np.array([p['x'] for p in prefix], dtype=float)
            memory.append((t[-1] - t, np.diff(t), x))
    return state, memory

def _forecast(params, data):
    a, b, c, tau = params
    (x, u, horizon), memory = data
    z = np.empty_like(x)
    for i, m in enumerate(memory):
        if isinstance(m, tuple):
            age, dt, hist_x = m
            weighted = np.exp(-age / tau) * hist_x
            z[i] = 0.52 * (weighted[0] + np.dot(dt, weighted[:-1] + weighted[1:]) / (2 * tau))
        else:
            z[i] = m
    n = max(1, math.ceil(float(horizon.max()) / 0.004))
    h = horizon / n
    drive = c * u
    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        z = z + (h/6) * (l1 + 2*l2 + 2*l3 + l4)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = _prepare(rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(params):
        return _forecast(params, data) - y
    best = None
    for tau0 in (0.15, 0.5, 1.0):
        result = least_squares(residual, [1.5, 1.0, 1.0, tau0], bounds=([0, 0, 0, 0.005], [np.inf]*4), x_scale='jac', max_nfev=250)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x
    return _params

def predict(row) -> float:
    return float(_forecast(_params, _prepare([row]))[0])