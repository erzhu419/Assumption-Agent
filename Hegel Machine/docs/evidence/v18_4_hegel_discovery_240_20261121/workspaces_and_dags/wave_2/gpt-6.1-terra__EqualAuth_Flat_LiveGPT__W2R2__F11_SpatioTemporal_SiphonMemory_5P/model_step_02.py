import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(rows, p):
    a, b, vmax, rho, tau = p
    u = np.array([r['u_now'] for r in rows], dtype=float) / 2.0
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float) / 48.0
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    def deriv(x, z):
        return -a * x + b * z + J, (0.52 * x - z) / tau
    for _ in range(48):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + h * k1x / 2, z + h * k1z / 2)
        k3x, k3z = deriv(x + h * k2x / 2, z + h * k2z / 2)
        k4x, k4z = deriv(x + h * k3x, z + h * k3z)
        x = x + h * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z = z + h * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return _evaluate(rows, p) - y
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.05], [10.0, 10.0, 20.0, 1.0, 20.0])
    starts = [[0.5, 0.5, 3.0, 0.5, 1.0], [1.0, 1.0, 5.0, 0.8, 0.3], [0.2, 0.2, 2.0, 0.2, 3.0]]
    results = [least_squares(residual, p, bounds=bounds, max_nfev=1500, ftol=1e-12, xtol=1e-12, gtol=1e-12) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_evaluate([row], _params)[0])