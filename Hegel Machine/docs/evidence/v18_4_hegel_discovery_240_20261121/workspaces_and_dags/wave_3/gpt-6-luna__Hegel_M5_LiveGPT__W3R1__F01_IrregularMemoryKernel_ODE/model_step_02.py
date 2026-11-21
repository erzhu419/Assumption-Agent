import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    t0 = float(pref[0]['t'])
    te = float(pref[-1]['t'])
    z = 0.52 * float(pref[0]['x'])
    for a, b in zip(pref[:-1], pref[1:]):
        ta, tb = float(a['t']), float(b['t'])
        xa, xb = float(a['x']), float(b['x'])
        dt = tb - ta
        wa = math.exp(-(te - ta) / tau)
        wb = math.exp(-(te - tb) / tau)
        z += 0.52 * 0.5 * (wa * xa + wb * xb) * dt / tau
    return math.exp(-(te - t0) / tau) * (0.52 * float(pref[0]['x'])) + (z - 0.52 * float(pref[0]['x']))

def _integrate(row, p):
    a, b, c, tau = map(float, p)
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
        k2x, k2z = f(x + .5*dt*k1x, z + .5*dt*k1z)
        k3x, k3z = f(x + .5*dt*k2x, z + .5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return x

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_integrate(r, p) for r in train_rows]) - y
    starts = ([0.6, 0.5, 0.8, 0.25], [0.3, 0.8, 0.5, 0.5], [1.0, 0.3, 1.0, 1.0])
    best = None
    for s in starts:
        sol = least_squares(residual, s, bounds=([0.001, 0.001, -5.0, 0.01], [5.0, 5.0, 5.0, 5.0]), max_nfev=3000)
        if best is None or np.sum(sol.fun**2) < np.sum(best.fun**2):
            best = sol
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_integrate(row, _PARAMS))