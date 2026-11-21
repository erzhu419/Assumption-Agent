import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = row['u_now'] / 0.05
    s = row['s_grad']
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    h = float(row['H'])
    n = max(1, int(math.ceil(h / 0.004)))
    dt = h / n
    def f(x, z):
        return -a * x + b * z + j, (0.52 * x - z) / tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = f(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return x

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(r, p) - r['y'] for r in train_rows])
    result = least_squares(residual, [1.0, 1.0, 3.0, 0.7, 0.05], bounds=([0.0, 0.0, 0.0, 0.0, 0.005], [20.0, 20.0, 30.0, 1.5, 2.0]), max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return float(_forecast(row, _params))