import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_keys = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')

def _forecast(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = data.T
    u = u / 2.0
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(64, math.ceil(4.0 * float(np.max(H)) * max(a, b, 1.0 / tau)))
    h = H / n
    def rhs(x, z):
        return -a*x + b*z + J, (0.52*x - z)/tau
    for _ in range(n):
        kx, kz = rhs(x, z)
        lx, lz = rhs(x + 0.5*h*kx, z + 0.5*h*kz)
        mx, mz = rhs(x + 0.5*h*lx, z + 0.5*h*lz)
        nx, nz = rhs(x + h*mx, z + h*mz)
        x = x + (h/6.0)*(kx + 2*lx + 2*mx + nx)
        z = z + (h/6.0)*(kz + 2*lz + 2*mz + nz)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = np.array([[r[k] for k in _keys] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.01], [40.0, 40.0, 30.0, 1.0, 20.0])
    best = None
    for start in ([1.0, 1.0, 6.5, 0.70, 0.35], [2.0, 2.0, 8.0, 0.65, 1.0], [0.5, 1.5, 4.0, 0.75, 0.12]):
        result = least_squares(lambda p: _forecast(p, data) - y, start, bounds=bounds, x_scale='jac', max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or result.cost < best.cost:
            best = result
        if math.sqrt(2.0 * best.cost / len(y)) < 1e-6:
            break
    _params = best.x
    return _params

def predict(row) -> float:
    data = np.array([[row[k] for k in _keys]], dtype=float)
    return float(_forecast(_params, data)[0])