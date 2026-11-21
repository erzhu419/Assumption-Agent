import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    hist = row.get('raw_prefix') or []
    if not hist:
        return 0.52 * float(row.get('x_now', 0.0))
    t0 = float(hist[0]['t'])
    tlast = float(hist[-1]['t'])
    z = 0.52 * float(hist[0]['x'])
    for left, right in zip(hist[:-1], hist[1:]):
        dt = float(right['t']) - float(left['t'])
        if dt <= 0:
            continue
        decay = math.exp(-dt / tau)
        xbar = 0.5 * (float(left['x']) + float(right['x']))
        z = decay * z + 0.52 * xbar * (1.0 - decay)
    return z

def _simulate(row, p):
    a, b, c, tau = [float(v) for v in p]
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xx, zz):
        return -a * xx + b * zz + c * u, (0.52 * xx - zz) / tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    best = None
    for start in ([0.3, 0.1, 0.5, 0.2], [0.8, 0.4, 0.8, 0.5], [0.1, 0.8, 0.2, 1.0]):
        res = least_squares(residual, np.asarray(start), bounds=([0.001, 0.0, -5.0, 0.01], [5.0, 5.0, 5.0, 5.0]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _PARAMS))