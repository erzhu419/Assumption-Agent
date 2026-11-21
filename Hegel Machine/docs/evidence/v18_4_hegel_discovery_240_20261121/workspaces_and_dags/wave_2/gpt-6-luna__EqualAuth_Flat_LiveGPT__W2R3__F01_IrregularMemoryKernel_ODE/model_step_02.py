import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    pts = row.get('raw_prefix')
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    if not pts:
        return 0.52 * float(row['x_now'])
    ts = np.array([p['t'] for p in pts], dtype=float)
    xs = np.array([p['x'] for p in pts], dtype=float)
    z = 0.52 * xs[0]
    for i in range(len(ts) - 1):
        dt = float(ts[i + 1] - ts[i])
        if dt <= 0:
            continue
        e = math.exp(-dt / tau)
        slope = (xs[i + 1] - xs[i]) / dt
        z = e * z + 0.52 * (xs[i] * (1.0 - e) + slope * (dt - tau * (1.0 - e)))
    return z

def _simulate(row, pars):
    a, b, c, tau = pars
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a * xv + b * zv + c * u, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5 * dt * k1x, z + 0.5 * dt * k1z)
        k3x, k3z = f(x + 0.5 * dt * k2x, z + 0.5 * dt * k2z)
        k4x, k4z = f(x + dt * k3x, z + dt * k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.5, 0.4, 0.8, 0.5]),
                           bounds=([0.001, 0.001, 0.001, 0.03], [5.0, 5.0, 5.0, 5.0]),
                           max_nfev=3000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _PARAMS))