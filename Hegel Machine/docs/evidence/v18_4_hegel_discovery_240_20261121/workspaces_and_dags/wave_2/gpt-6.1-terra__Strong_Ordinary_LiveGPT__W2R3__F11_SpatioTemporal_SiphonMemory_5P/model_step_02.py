import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _trajectory(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(x, z):
        return -a*x + b*z + flux, (0.52*x - z)/tau
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
        return np.array([_trajectory(r, p) - float(r['y']) for r in rows])
    lower = [0.0, 0.0, 0.0, 0.0, 0.005]
    upper = [20.0, 20.0, 50.0, 2.0, 2.0]
    starts = [[1.0, 1.0, 3.0, 0.7, 0.05], [0.5, 2.0, 4.0, 0.7, 0.15], [2.0, 3.0, 5.0, 0.7, 0.3]]
    fits = [least_squares(residual, start, bounds=(lower, upper), max_nfev=2000, ftol=1e-11, xtol=1e-11, gtol=1e-11) for start in starts]
    _params = min(fits, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    return float(_trajectory(row, _params))
