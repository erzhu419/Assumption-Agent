import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pts], dtype=float)
    xs = np.array([float(p['x']) for p in pts], dtype=float)
    z = 0.52 * xs[0]
    for i in range(len(ts) - 1):
        dt = ts[i + 1] - ts[i]
        x0, x1 = xs[i], xs[i + 1]
        def f(q, x):
            return (0.52 * x - q) / tau
        k1 = f(z, x0)
        k2 = f(z + 0.5 * dt * k1, 0.5 * (x0 + x1))
        k3 = f(z + 0.5 * dt * k2, 0.5 * (x0 + x1))
        k4 = f(z + dt * k3, x1)
        z += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return z

def _predict_params(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a * xx + b * zz + c * u, (0.52 * xx - zz) / tau])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in train_rows]) - y
    best = None
    for start in ([0.5, 0.5, 0.5, 0.5], [1.0, 0.3, 0.5, 1.0], [0.2, 0.8, 0.3, 2.0]):
        result = least_squares(residual, start, bounds=([0.001, 0.001, 0.001, 0.03], [5.0, 5.0, 5.0, 10.0]), max_nfev=3000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return _predict_params(row, _PARAMS)