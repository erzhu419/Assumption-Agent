import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _forecast(p, data):
    a, b, vmax, rho, tau = p
    u, x, z, g, H = data.T
    u = u / 0.05
    flux = vmax * (1.0 - rho * g / (1.0 + g)) * u / (1.2 + u)
    n = max(1, math.ceil(float(np.max(np.abs(H))) / 0.004))
    h = H / n
    def rhs(x, z):
        return -a*x + b*z + flux, (0.52*x - z)/tau
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + h*k1/2, z + h*l1/2)
        k3, l3 = rhs(x + h*k2/2, z + h*l2/2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x = x + h*(k1 + 2*k2 + 2*k3 + k4)/6
        z = z + h*(l1 + 2*l2 + 2*l3 + l4)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = np.asarray([[r[k] for k in _KEYS] for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    def residual(p):
        return _forecast(p, data) - y
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.003], [50.0, 50.0, 60.0, 1.0, 5.0])
    best = None
    for tau in (0.08, 0.3, 0.8):
        result = least_squares(residual, [1.0, 1.0, 6.5, 0.7, tau], bounds=bounds, x_scale='jac', max_nfev=180)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-10:
            break
    _params = best.x
    return _params

def predict(row):
    data = np.asarray([[row[k] for k in _KEYS]], dtype=float)
    return float(_forecast(_params, data)[0])