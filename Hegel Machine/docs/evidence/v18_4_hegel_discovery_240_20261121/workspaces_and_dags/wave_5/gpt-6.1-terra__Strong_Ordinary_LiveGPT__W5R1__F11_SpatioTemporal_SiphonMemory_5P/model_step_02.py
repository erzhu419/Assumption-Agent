import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H) / 0.004)))
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
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    best = None
    for tau in (0.15, 0.5, 1.5):
        result = least_squares(residual, [0.4, 0.8, 4.0, 0.7, tau], bounds=([0, 0, 0, 0, 0.02], [10, 20, 50, 2, 20]), max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_forecast(row, _params))