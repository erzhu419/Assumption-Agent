import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(pts[0]['x'])
    for p, q in zip(pts[:-1], pts[1:]):
        dt = float(q['t']) - float(p['t'])
        xm = 0.5 * (float(p['x']) + float(q['x']))
        e = math.exp(-dt / max(tau, 1e-8))
        z = z * e + 0.52 * xm * (1.0 - e)
    return z

def _simulate(row, pars):
    a, b, c, tau = [float(v) for v in pars]
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
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in rows]) - y
    best = None
    starts = ([0.5, 0.2, 0.5, 0.2], [1.0, 0.5, 1.0, 0.5], [0.2, 0.1, 0.8, 1.0])
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([1e-5, 0.0, -10.0, 0.01], [10.0, 10.0, 10.0, 10.0]), max_nfev=5000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return np.asarray(best.x, dtype=float)

def predict(row):
    return float(_simulate(row, fit_params))