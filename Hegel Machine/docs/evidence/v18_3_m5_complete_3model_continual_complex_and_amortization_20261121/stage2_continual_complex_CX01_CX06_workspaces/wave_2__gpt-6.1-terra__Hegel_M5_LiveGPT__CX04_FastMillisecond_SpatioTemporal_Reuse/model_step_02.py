import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(rows, p):
    a, b, vmax, rho, tau = p
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    H = np.array([r['H'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float) / 0.005
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = 200
    dt = H / n
    def deriv(x, z):
        return -a * x + b * z + J, (0.52 * x - z) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x = x + dt * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z = z + dt * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [r['y'] for r in rows]
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.005], [10.0, 10.0, 20.0, 1.0, 5.0])
    best = None
    for tau in (0.03, 0.2, 1.0):
        result = least_squares(lambda p: _evaluate(rows, p) - y, [1.0, 1.0, 6.5, 0.7, tau], bounds=bounds, max_nfev=600, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate([row], _params)[0])