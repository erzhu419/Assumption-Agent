import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x, z = float(row['x_now']), float(row['mem_conv'])
    h = float(row['H']) / 48.0
    def deriv(x, z):
        return -a_slow * x + b_couple * z + j, (0.52 * x - z) / tau_fast
    for _ in range(48):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + h*k1x/2, z + h*k1z/2)
        k3x, k3z = deriv(x + h*k2x/2, z + h*k2z/2)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x += h * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += h * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in train_rows])
    best = None
    for guess in ([2, 0.5, 0.3, 0.5, 0.5], [3, 0.8, 0.8, 1.0, 1.0], [1, 0.2, 0.1, 0.2, 0.2]):
        result = least_squares(residual, guess, bounds=([0, 0, 0, 0, 0.03], [20, 1.5, 5, 5, 10]), max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _forecast(row, _params)
