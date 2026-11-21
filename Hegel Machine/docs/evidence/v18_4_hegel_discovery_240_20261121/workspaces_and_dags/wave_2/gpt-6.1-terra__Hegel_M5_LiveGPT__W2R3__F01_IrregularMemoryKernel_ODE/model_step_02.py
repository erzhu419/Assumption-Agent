import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, c, tau = p
    if row.get('mem_conv') is not None:
        z = float(row['mem_conv'])
    else:
        pref = row['raw_prefix']
        end = float(pref[-1]['t'])
        duration = end - float(pref[0]['t'])
        conv = 0.0
        for left, right in zip(pref[:-1], pref[1:]):
            dt = float(right['t']) - float(left['t'])
            v0 = math.exp(-(end - float(left['t'])) / tau) * float(left['x'])
            v1 = math.exp(-(end - float(right['t'])) / tau) * float(right['x'])
            conv += 0.5 * (v0 + v1) * dt / tau
        z = 0.52 * (math.exp(-duration / tau) * float(pref[0]['x']) + conv)
    x = float(row['x_now'])
    u = float(row['u_now'])
    h_total = float(row['H'])
    n = max(1, int(math.ceil(abs(h_total) / 0.004)))
    h = h_total / n
    def deriv(xv, zv):
        return -a * xv + b * zv + c * u, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + h*k1x/2, z + h*k1z/2)
        k3x, k3z = deriv(x + h*k2x/2, z + h*k2z/2)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x += h * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += h * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in train_rows])
    best = None
    for tau0 in (0.15, 0.5, 1.5):
        result = least_squares(residual, [1.0, 0.8, 1.0, tau0], bounds=([0.0, -10.0, -10.0, 0.02], [10.0, 10.0, 10.0, 10.0]), max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _forecast(row, _params)
