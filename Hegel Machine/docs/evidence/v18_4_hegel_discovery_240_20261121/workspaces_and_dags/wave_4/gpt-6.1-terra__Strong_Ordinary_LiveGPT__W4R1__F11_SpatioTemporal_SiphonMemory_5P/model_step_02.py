import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, tau, vmax, rho = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(x, z):
        return -a*x + b*z + J, (0.52*x - z)/tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, [0.5, 0.8, 0.5, 3.0, 0.7], bounds=([0, 0, 0.02, 0, 0], [10, 10, 10, 30, 2]), max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return _forecast(row, _params)
