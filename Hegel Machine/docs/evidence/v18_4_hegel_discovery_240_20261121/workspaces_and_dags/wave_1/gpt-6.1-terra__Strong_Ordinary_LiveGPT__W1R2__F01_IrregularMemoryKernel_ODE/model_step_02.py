import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    p = row['raw_prefix']
    end = float(p[-1]['t'])
    z = 0.52 * float(p[0]['x']) * math.exp(-(end - float(p[0]['t'])) / tau)
    for left, right in zip(p[:-1], p[1:]):
        t0, t1 = float(left['t']), float(right['t'])
        f0 = float(left['x']) * math.exp(-(end - t0) / tau)
        f1 = float(right['x']) * math.exp(-(end - t1) / tau)
        z += 0.52 * (t1 - t0) * (f0 + f1) / (2.0 * tau)
    return z

def _forecast(row, par):
    a, b, c, tau = par
    x = float(row['x_now'])
    z = _memory(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1, l1 = deriv(x, z)
        k2, l2 = deriv(x + dt*k1/2, z + dt*l1/2)
        k3, l3 = deriv(x + dt*k2/2, z + dt*l2/2)
        k4, l4 = deriv(x + dt*k3, z + dt*l3)
        x += dt*(k1 + 2*k2 + 2*k3 + k4)/6
        z += dt*(l1 + 2*l2 + 2*l3 + l4)/6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    best = None
    for tau in (0.12, 0.4, 1.0, 2.0):
        result = least_squares(residual, [1.0, 0.5, 1.0, tau], bounds=([0.0, -5.0, 0.0, 0.05], [10.0, 10.0, 10.0, 10.0]), max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    return _forecast(row, _params)
