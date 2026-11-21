import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(rows, p):
    a, b, vmax, rho, tau = p
    u = np.array([r['u_now'] for r in rows], dtype=float) / 2.0
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    dt = np.array([r['H'] for r in rows], dtype=float) / 40.0
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    def f(x, z):
        return -a*x + b*z + j, (0.52*x-z)/tau
    for _ in range(40):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = f(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x = x + dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z = z + dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return _evaluate(rows, p) - y
    best = None
    for start in ([0.3, 0.5, 2.0, 0.5, 0.5], [0.6, 1.0, 3.0, 0.8, 1.5], [0.1, 0.2, 1.0, 0.2, 0.2]):
        result = least_squares(residual, start, bounds=([0, 0, 0, 0, 0.08], [5, 5, 20, 2, 10]), max_nfev=1200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_evaluate([row], _params)[0])
