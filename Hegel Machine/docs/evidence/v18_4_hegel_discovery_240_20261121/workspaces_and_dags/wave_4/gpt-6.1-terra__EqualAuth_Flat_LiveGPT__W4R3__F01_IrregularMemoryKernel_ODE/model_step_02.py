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
        z += 0.52 * (t1 - t0) / (2.0 * tau) * (float(left['x']) * math.exp(-(end - t0) / tau) + float(right['x']) * math.exp(-(end - t1) / tau))
    return z

def _forecast(row, params):
    a, b, c, tau = params
    x = float(row['x_now'])
    z = _memory(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H) / 0.004)))
    h = H / n
    def f(x, z):
        return -a * x + b * z + c * u, (0.52 * x - z) / tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + h*k1x/2, z + h*k1z/2)
        k3x, k3z = f(x + h*k2x/2, z + h*k2z/2)
        k4x, k4z = f(x + h*k3x, z + h*k3z)
        x += h * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += h * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    best = None
    for tau in (0.15, 0.5, 1.0, 2.0):
        result = least_squares(residual, [1.0, 1.0, 1.0, tau], bounds=([0.0, -10.0, -10.0, 0.03], [10.0, 10.0, 10.0, 10.0]), max_nfev=1000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _forecast(row, _params)
