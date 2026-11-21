import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    h = float(row['H'])
    n = max(1, int(math.ceil(abs(h) / min(0.005, tau / 4.0))))
    dt = h / n
    x, z = float(row['x_now']), float(row['mem_conv'])
    def deriv(x, z):
        return -a * x + b * z + j, (0.52 * x - z) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    lower = [0.0, 0.0, 0.01, 0.0, 0.005]
    upper = [20.0, 20.0, 30.0, 1.0, 2.0]
    starts = [[1.0, 1.0, 6.5, 0.7, 0.1], [2.0, 2.0, 6.5, 0.7, 0.4], [1.0, 3.0, 6.5, 0.7, 0.03]]
    results = [least_squares(residual, p, bounds=(lower, upper), max_nfev=1000) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    return _forecast(row, _params)