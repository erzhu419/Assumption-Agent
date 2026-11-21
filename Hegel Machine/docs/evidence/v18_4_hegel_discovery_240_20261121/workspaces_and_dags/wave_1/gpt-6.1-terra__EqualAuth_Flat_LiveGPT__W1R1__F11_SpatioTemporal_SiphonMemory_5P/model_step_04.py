import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict(row, p):
    a, b, tau, vmax, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    drive = vmax * (1.0 - 0.70 * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(max(0.0, u))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    n = max(1, int(math.ceil(abs(h) / 0.004)))
    dt = h / n
    def deriv(x, z):
        return -a*x + b*z + drive, (0.52*x - z)/tau
    for _ in range(n):
        k1, l1 = deriv(x, z)
        k2, l2 = deriv(x + dt*k1/2, z + dt*l1/2)
        k3, l3 = deriv(x + dt*k2/2, z + dt*l2/2)
        k4, l4 = deriv(x + dt*k3, z + dt*l3)
        x += dt*(k1 + 2*k2 + 2*k3 + k4)/6
        z += dt*(l1 + 2*l2 + 2*l3 + l4)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_predict(r, p) - float(r['y']) for r in rows])
    starts = [[1, 1, 0.2, 2, 0.5], [2, 0.5, 1, 3, 1], [0.5, 2, 0.5, 1, 1], [1, -1, 1, 4, 0]]
    bounds = ([0, -10, 0.02, -20, -20], [10, 10, 10, 20, 20])
    fits = [least_squares(residual, p, bounds=bounds, max_nfev=1500) for p in starts]
    _params = min(fits, key=lambda f: np.dot(f.fun, f.fun)).x
    return _params

def predict(row):
    return float(_predict(row, _params))