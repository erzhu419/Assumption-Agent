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
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.003], [20.0, 20.0, 30.0, 1.0, 10.0])
    starts = [[1.0, 1.0, 6.5, 0.7, 0.2], [0.5, 2.0, 6.5, 0.7, 0.05], [2.0, 1.0, 6.5, 0.7, 1.0]]
    best = None
    for start in starts:
        result = least_squares(lambda p: _evaluate(rows, p) - y, start, bounds=bounds, max_nfev=500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _params = best.x

def predict(row):
    return float(_evaluate([row], _params)[0])