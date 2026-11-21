import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(rows, p):
    a, b, vmax, rho, tau = p
    x = np.array([float(r['x_now']) for r in rows])
    z = np.array([float(r['mem_conv']) for r in rows])
    h = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows]) / 0.005
    s = np.array([float(r['s_grad']) for r in rows])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, int(math.ceil(float(np.max(h)) / min(0.005, tau * 0.5))))
    dt = h / n
    def deriv(xv, zv):
        return -a * xv + b * zv + j, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x = x + dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z = z + dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return _forecast(rows, p) - y
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.002], [10.0, 10.0, 20.0, 2.0, 5.0])
    starts = [[0.5, 0.5, 3.0, 0.5, 0.05], [1.0, 1.0, 3.0, 0.5, 0.5], [0.5, 1.0, 3.0, 0.5, 1.5]]
    results = [least_squares(residual, p, bounds=bounds, max_nfev=500, xtol=1e-10, ftol=1e-10, gtol=1e-10) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_forecast([row], _params)[0])