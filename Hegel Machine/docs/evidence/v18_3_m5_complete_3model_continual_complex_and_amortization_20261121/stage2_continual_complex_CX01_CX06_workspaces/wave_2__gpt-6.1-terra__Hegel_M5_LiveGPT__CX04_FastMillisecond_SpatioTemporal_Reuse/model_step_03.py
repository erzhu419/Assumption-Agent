import numpy as np
from scipy.optimize import least_squares

_params = None

def _simulate(rows, p):
    a, b, vmax, rho, tau = p
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float) / 0.005
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, int(np.ceil(np.max(h) / min(0.002, 0.8 * tau))))
    dt = h / n
    def deriv(xx, zz):
        return -a * xx + b * zz + j, (0.52 * xx - zz) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = deriv(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x = x + dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z = z + dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.002], [20.0, 20.0, 30.0, 1.0, 3.0])
    best = None
    for tau0 in (0.01, 0.1, 0.5):
        result = least_squares(lambda p: _simulate(rows, p) - y, [1.0, 1.0, 6.5, 0.7, tau0], bounds=bounds, max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _params = best.x

def predict(row):
    return float(_simulate([row], _params)[0])