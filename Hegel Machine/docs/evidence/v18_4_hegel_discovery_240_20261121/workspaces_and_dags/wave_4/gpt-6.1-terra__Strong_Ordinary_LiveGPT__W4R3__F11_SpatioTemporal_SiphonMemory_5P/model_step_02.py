import numpy as np
from scipy.optimize import least_squares

_params = None

def _simulate(rows, p):
    a, b, vmax, rho, tau = p
    u = np.array([r['u_now'] for r in rows], dtype=float) / 0.05
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = np.maximum(1, np.ceil(h / 0.004).astype(int))
    dt = h / n
    def deriv(xv, zv):
        return -a * xv + b * zv + j, (0.52 * xv - zv) / tau
    for i in range(int(np.max(n))):
        d = np.where(i < n, dt, 0.0)
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + d*k1x/2, z + d*k1z/2)
        k3x, k3z = deriv(x + d*k2x/2, z + d*k2z/2)
        k4x, k4z = deriv(x + d*k3x, z + d*k3z)
        x = x + d*(k1x + 2*k2x + 2*k3x + k4x)/6
        z = z + d*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [r['y'] for r in rows] if False else np.array([r['y'] for r in rows], dtype=float)
    lower = [0.0, 0.0, 0.0, 0.0, 0.008]
    upper = [20.0, 20.0, 30.0, 2.0, 1.0]
    starts = [[1.0, 1.0, 3.0, 0.7, 0.05], [2.0, 3.0, 5.0, 0.7, 0.1], [0.5, 0.5, 2.0, 0.7, 0.025]]
    best = None
    for start in starts:
        result = least_squares(lambda p: _simulate(rows, p) - y, start, bounds=(lower, upper), max_nfev=1500, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_simulate([row], _params)[0])