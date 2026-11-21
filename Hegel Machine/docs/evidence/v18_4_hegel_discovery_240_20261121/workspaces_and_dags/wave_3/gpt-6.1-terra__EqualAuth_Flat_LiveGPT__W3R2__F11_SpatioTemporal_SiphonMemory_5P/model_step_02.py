import numpy as np
from scipy.optimize import least_squares

_params = None

def _simulate(rows, p):
    a, b, tau, vmax, rho = p
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float) / 160.0
    u = np.array([r['u_now'] for r in rows], dtype=float) / 0.05
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    def deriv(x, z):
        return -a*x + b*z + J, (0.52*x - z)/tau
    for _ in range(160):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + h*k1x/2, z + h*k1z/2)
        k3x, k3z = deriv(x + h*k2x/2, z + h*k2z/2)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x = x + h*(k1x + 2*k2x + 2*k3x + k4x)/6
        z = z + h*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    bounds = ([0.0, 0.0, 0.005, 0.0, 0.0], [10.0, 10.0, 3.0, 20.0, 2.0])
    best = None
    for start in ([0.5, 0.5, 0.05, 3.0, 0.5], [1.0, 1.0, 0.2, 3.0, 1.0], [0.2, 2.0, 0.5, 2.0, 0.5]):
        result = least_squares(lambda p: _simulate(rows, p) - y, start, bounds=bounds, max_nfev=600, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_simulate([row], _params)[0])