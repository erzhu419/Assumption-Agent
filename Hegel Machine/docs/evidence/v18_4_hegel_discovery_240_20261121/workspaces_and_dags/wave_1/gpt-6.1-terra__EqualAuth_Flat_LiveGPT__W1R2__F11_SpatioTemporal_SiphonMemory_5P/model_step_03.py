import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict(row, p):
    a, b, vmax, rho, c, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(max(0.0, u))
    n = max(1, int(math.ceil(abs(h) / 0.01)))
    dt = h / n
    def deriv(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau
    for _ in range(n):
        k1, l1 = deriv(x, z)
        k2, l2 = deriv(x + dt*k1/2, z + dt*l1/2)
        k3, l3 = deriv(x + dt*k2/2, z + dt*l2/2)
        k4, l4 = deriv(x + dt*k3, z + dt*l3)
        x += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        z += dt * (l1 + 2*l2 + 2*l3 + l4) / 6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    targets = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict(r, p) for r in rows]) - targets
    bounds = ([0.0, -5.0, 0.0, 0.0, -5.0, 0.03], [10.0, 5.0, 20.0, 1.5, 10.0, 10.0])
    starts = [[1.0, 0.5, 3.0, 0.7, 0.5, 0.5], [2.0, 1.0, 5.0, 0.7, 0.0, 1.0], [0.5, 0.2, 2.0, 0.7, 1.0, 0.1]]
    results = [least_squares(residual, p, bounds=bounds, max_nfev=1200) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return _predict(row, _params)
