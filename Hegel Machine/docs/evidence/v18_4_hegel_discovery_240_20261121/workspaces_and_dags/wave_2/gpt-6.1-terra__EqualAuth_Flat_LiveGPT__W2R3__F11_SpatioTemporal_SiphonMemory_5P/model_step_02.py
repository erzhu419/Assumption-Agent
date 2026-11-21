import numpy as np
from scipy.optimize import least_squares

_params = None

def _simulate(rows, p):
    a, b, vmax, rho, tau = p
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float) / 0.05
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float) / 160.0
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    def deriv(xx, zz):
        return -a * xx + b * zz + j, (0.52 * xx - zz) / tau
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
    bounds = ([0, 0, 0, 0, 0.005], [10, 10, 20, 2, 2])
    starts = [[1, 1, 3, 0.5, 0.05], [0.5, 2, 4, 1, 0.2], [2, 1, 5, 0.5, 0.01]]
    results = [least_squares(lambda p: _simulate(rows, p) - y, start, bounds=bounds, max_nfev=1200, ftol=1e-11, xtol=1e-11, gtol=1e-11) for start in starts]
    _params = min(results, key=lambda r: np.sum(r.fun**2)).x

def predict(row):
    return float(_simulate([row], _params)[0])