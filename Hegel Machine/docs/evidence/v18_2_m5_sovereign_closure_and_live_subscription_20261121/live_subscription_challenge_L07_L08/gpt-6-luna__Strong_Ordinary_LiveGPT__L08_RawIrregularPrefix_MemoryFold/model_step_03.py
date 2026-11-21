import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0_from_prefix(row, tau):
    prefix = row.get('raw_prefix')
    if not prefix:
        m = row.get('mem_conv')
        return float(m) if m is not None else 0.52 * float(row['x_now'])
    z = 0.52 * float(prefix[0]['x'])
    t_prev = float(prefix[0]['t'])
    x_prev = float(prefix[0]['x'])
    for p in prefix[1:]:
        t = float(p['t']); x = float(p['x'])
        dt = t - t_prev
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        slope = (x - x_prev) / dt if dt else 0.0
        for j in range(n):
            xa = x_prev + slope * (j * h)
            xm = xa + 0.5 * slope * h
            xb = xa + slope * h
            f = lambda zz, xx: (0.52 * xx - zz) / tau
            k1 = f(z, xa)
            k2 = f(z + 0.5*h*k1, xm)
            k3 = f(z + 0.5*h*k2, xm)
            k4 = f(z + h*k3, xb)
            z += h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t_prev, x_prev = t, x
    return z

def _forecast(row, pars):
    a, b, c, tau = map(float, pars)
    x = float(row['x_now'])
    m = row.get('mem_conv')
    z = float(m) if m is not None else _z0_from_prefix(row, tau)
    H = float(row['H']); u = float(row['u_now'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx - zz)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z=f(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z=f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += h*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.asarray([_forecast(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([0.8, 0.5, 0.8, 0.12]),
                           bounds=([0.001, 0.001, 0.001, 0.001], [10.0, 10.0, 10.0, 10.0]),
                           max_nfev=3000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_forecast(row, _PARAMS))
