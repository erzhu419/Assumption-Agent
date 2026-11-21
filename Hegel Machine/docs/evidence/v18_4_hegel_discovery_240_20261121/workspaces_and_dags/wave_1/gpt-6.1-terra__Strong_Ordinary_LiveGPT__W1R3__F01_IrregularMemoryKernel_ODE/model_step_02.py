import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict(row, p):
    a, b, c, tau = p
    if row.get('mem_conv') is not None:
        z = float(row['mem_conv'])
    else:
        prefix = row['raw_prefix']
        t0 = float(prefix[0]['t'])
        latest = float(prefix[-1]['t'])
        z = 0.52 * float(prefix[0]['x']) * math.exp(-(latest - t0) / tau)
        for left, right in zip(prefix[:-1], prefix[1:]):
            tl, tr = float(left['t']), float(right['t'])
            z += 0.52 * (tr - tl) / (2.0 * tau) * (
                float(left['x']) * math.exp(-(latest - tl) / tau)
                + float(right['x']) * math.exp(-(latest - tr) / tau))
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x+dt*k1x/2, z+dt*k1z/2)
        k3x, k3z = f(x+dt*k2x/2, z+dt*k2z/2)
        k4x, k4z = f(x+dt*k3x, z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_predict(r, p)-float(r['y']) for r in rows])
    bounds = ([0.0, 0.0, 0.0, 0.025], [20.0, 20.0, 20.0, 10.0])
    starts = [[2.0, 1.0, 1.5, 0.2], [2.0, 2.0, 1.5, 0.6], [1.0, 0.5, 1.0, 1.5]]
    results = [least_squares(residual, p, bounds=bounds, max_nfev=1500, xtol=1e-11, ftol=1e-11, gtol=1e-11) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return _predict(row, _params)
