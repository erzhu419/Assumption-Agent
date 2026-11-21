import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict(row, p):
    vmax, rho, a, b, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, int(math.ceil(abs(h) / 0.004)))
    dt = h / n
    def deriv(x, z):
        return -a*x + b*z + j, (0.52*x - z)/tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_predict(r, p) - float(r['y']) for r in rows])
    best = None
    for tau in (0.15, 0.5, 1.5):
        result = least_squares(residual, [3.0, 0.7, 0.5, 0.5, tau], bounds=([0, 0, 0, 0, 0.02], [50, 1.5, 10, 10, 20]), max_nfev=1500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_predict(row, _params))