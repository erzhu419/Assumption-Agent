import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = row['u_now'] / 0.005
    s = row['s_grad']
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x, z = float(row['x_now']), float(row['mem_conv'])
    n = max(1, int(math.ceil(float(row['H']) / 0.004)))
    dt = float(row['H']) / n
    def deriv(x, z):
        return -a * x + b * z + j, (0.52 * x - z) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - r['y'] for r in rows])
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.002], [20.0, 20.0, 30.0, 1.5, 10.0])
    best = None
    for tau in (0.02, 0.15, 0.6, 2.0):
        result = least_squares(residual, [0.8, 1.0, 3.0, 0.7, tau], bounds=bounds, max_nfev=1200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_forecast(row, _params))