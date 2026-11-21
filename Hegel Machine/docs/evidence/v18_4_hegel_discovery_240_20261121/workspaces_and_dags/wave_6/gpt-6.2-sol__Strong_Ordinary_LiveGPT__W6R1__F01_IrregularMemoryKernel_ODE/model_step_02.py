import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    x = float(row['x_now'])
    mem = row.get('mem_conv')
    prefix = ()
    if mem is None and row.get('raw_prefix'):
        prefix = tuple(sorted((float(p['t']), float(p['x'])) for p in row['raw_prefix']))
    return x, float(row['u_now']), float(row['H']), mem, prefix

def _forecast(p, data):
    a, b, c, tau = map(float, p)
    x, u, H, mem, prefix = data
    if mem is not None:
        z = float(mem)
    elif prefix:
        t0, x0 = prefix[0]
        z = 0.52 * x0
        for t1, x1 in prefix[1:]:
            q = (t1 - t0) / tau
            e = math.exp(-q)
            z = e*z + 0.26*q*(e*x0 + x1)
            t0, x0 = t1, x1
    else:
        z = 0.52 * x
    n = max(1, math.ceil(H / 0.004))
    h = H / n
    drive = c*u
    inv_tau = 1.0/tau
    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z)*inv_tau
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        z += (h/6)*(l1 + 2*l2 + 2*l3 + l4)
    return float(x)

def fit(train_rows):
    global _params
    rows = [(_prepare(row), float(row['y'])) for row in train_rows]
    def residuals(p):
        return np.array([_forecast(p, data) - y for data, y in rows])
    _params = least_squares(residuals, [2.0, 1.0, 1.0, 0.3], bounds=([0.0, 0.0, 0.0, 0.01], np.inf), x_scale='jac').x
    return _params

def predict(row):
    return _forecast(_params, _prepare(row))
