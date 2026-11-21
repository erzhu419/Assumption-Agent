import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(rows, p):
    a, b, vmax, rho, tau = p
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float) / 0.005
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float) / 256.0
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    def deriv(xv, zv):
        return -a * xv + b * zv + j, (0.52 * xv - zv) / tau
    for _ in range(256):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*h*k1x, z + 0.5*h*k1z)
        k3x, k3z = deriv(x + 0.5*h*k2x, z + 0.5*h*k2z)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x = x + h*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z = z + h*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    lower = [0.0, 0.0, 0.0, 0.0, 0.005]
    upper = [10.0, 10.0, 20.0, 1.0, 5.0]
    starts = [[0.7, 0.8, 2.0, 0.5, 0.2], [1.5, 2.0, 3.0, 0.3, 0.8], [0.3, 0.5, 1.5, 0.8, 0.05]]
    best = None
    for start in starts:
        result = least_squares(lambda p: _evaluate(rows, p) - y, start, bounds=(lower, upper), max_nfev=500, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _params = best.x

def predict(row):
    return float(_evaluate([row], _params)[0])