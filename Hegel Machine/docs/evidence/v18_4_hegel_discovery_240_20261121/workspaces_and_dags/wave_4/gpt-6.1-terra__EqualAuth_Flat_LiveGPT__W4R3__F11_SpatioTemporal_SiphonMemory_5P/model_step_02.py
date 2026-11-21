import numpy as np
from scipy.optimize import least_squares

_params = None

def _simulate(rows, p):
    a, b, vmax, rho, tau = p
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float) / 0.05
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    H = np.array([r['H'] for r in rows], dtype=float)
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, int(np.ceil(np.max(H) / min(0.005, tau / 5.0))))
    h = H / n
    def deriv(xx, zz):
        return -a * xx + b * zz + J, (0.52 * xx - zz) / tau
    for _ in range(n):
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
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.01], [10.0, 10.0, 20.0, 2.0, 2.0])
    starts = [[0.7, 0.8, 2.0, 0.5, 0.05], [1.5, 1.5, 3.0, 0.8, 0.2], [0.2, 0.3, 1.0, 0.2, 0.5]]
    results = [least_squares(lambda p: _simulate(rows, p) - y, start, bounds=bounds, max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10) for start in starts]
    _params = min(results, key=lambda result: np.sum(result.fun ** 2)).x

def predict(row):
    return float(_simulate([row], _params)[0])