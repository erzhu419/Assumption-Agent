import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict(row, p):
    vmax, rho, c_loc, a, b, c, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv - zv)/tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    base = vmax*(1-rho*s/(1+s))*u/(1.2+u) + c_loc*math.sqrt(max(0.0, u))
    return x + base

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_predict(r, p) for r in rows]) - y
    lower = [0, 0.69, -5, 0, 0, 0, 0.03]
    upper = [10, 0.72, 5, 10, 10, 5, 10]
    starts = [[2, 0.705, 0.2, 1, 0.5, 0.2, 0.5],
              [1, 0.705, 0.5, 2, 1, 0.1, 1.0],
              [3, 0.705, -0.2, 0.5, 0.3, 0.3, 0.2]]
    results = [least_squares(residual, p, bounds=(lower, upper), max_nfev=1500) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_predict(row, _params))
