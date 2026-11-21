import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, tau, vmax, rho = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    h = float(row['H']) / 40.0
    def deriv(x, z):
        return -a*x + b*z + j, (0.52*x - z)/tau
    for _ in range(40):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + h*k1x/2, z + h*k1z/2)
        k3x, k3z = deriv(x + h*k2x/2, z + h*k2z/2)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x += h*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += h*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    starts = [[0.5, 0.5, 0.5, 2.0, 0.5], [1.0, 1.0, 1.0, 3.0, 0.8], [0.2, 0.8, 0.2, 1.5, 0.3]]
    fits = [least_squares(residual, p, bounds=([0, 0, 0.05, 0, 0], [10, 10, 10, 20, 1]), max_nfev=2000, xtol=1e-11, ftol=1e-11, gtol=1e-11) for p in starts]
    _params = min(fits, key=lambda f: np.dot(f.fun, f.fun)).x
    return _params

def predict(row):
    return float(_forecast(row, _params))