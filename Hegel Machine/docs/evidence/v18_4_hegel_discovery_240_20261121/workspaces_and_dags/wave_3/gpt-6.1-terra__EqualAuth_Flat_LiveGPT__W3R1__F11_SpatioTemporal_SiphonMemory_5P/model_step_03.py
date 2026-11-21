import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    forcing = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    h = float(row['H'])
    n = max(16, int(np.ceil(abs(h) / 0.01)))
    dt = h / n
    def deriv(x, z):
        return -a*x + b*z + forcing, (0.52*x-z)/tau
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
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    best = None
    for start in ([0.5, 0.5, 2.0, 0.5, 0.5], [1.0, 1.0, 3.0, 0.8, 1.0], [0.2, 0.2, 1.0, 0.2, 0.2]):
        result = least_squares(residual, start, bounds=([0, 0, 0, 0, 0.03], [10, 10, 20, 2, 10]), max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return _forecast(row, _params)
