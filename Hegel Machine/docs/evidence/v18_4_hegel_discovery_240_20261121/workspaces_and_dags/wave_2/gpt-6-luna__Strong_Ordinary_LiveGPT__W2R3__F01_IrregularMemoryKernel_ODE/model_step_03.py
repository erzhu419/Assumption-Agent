import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _initial_memory(row, tau):
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(prefix[0]['x'])
    t_latest = float(prefix[-1]['t'])
    for p, q in zip(prefix[:-1], prefix[1:]):
        dt = float(q['t']) - float(p['t'])
        decay = math.exp(-dt / tau)
        xbar = 0.5 * (float(p['x']) + float(q['x']))
        z = z * decay + 0.52 * xbar * (1.0 - decay)
    return z

def _simulate(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _initial_memory(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def rhs(xv, zv):
        return -a * xv + b * zv + c * u, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = rhs(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = rhs(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = ([0.5, 0.1, 0.5, 0.2], [1.0, 0.3, 1.0, 0.5], [0.2, 0.05, 0.3, 1.0])
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.001, 0.0, 0.0, 0.005], [10.0, 10.0, 10.0, 10.0]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _PARAMS))