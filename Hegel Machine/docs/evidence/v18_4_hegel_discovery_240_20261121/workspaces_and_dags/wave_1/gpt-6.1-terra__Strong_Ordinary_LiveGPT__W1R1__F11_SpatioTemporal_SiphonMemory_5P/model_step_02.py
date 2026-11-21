import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, drive, rho, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    forcing = drive * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, int(math.ceil(abs(h) / 0.004)))
    dt = h / n
    def deriv(xv, zv):
        return -a * xv + b * zv + forcing, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5 * dt * k1x, z + 0.5 * dt * k1z)
        k3x, k3z = deriv(x + 0.5 * dt * k2x, z + 0.5 * dt * k2z)
        k4x, k4z = deriv(x + dt * k3x, z + dt * k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    starts = [[1, 1, 4, 0.7, 0.3], [0.5, 0.5, 5, 0.7, 1], [2, 2, 6, 0.7, 0.1], [0.2, 1, 3, 0.5, 2]]
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.02], [20.0, 20.0, 50.0, 1.5, 20.0])
    results = [least_squares(residual, p, bounds=bounds, max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return _forecast(row, _params)
