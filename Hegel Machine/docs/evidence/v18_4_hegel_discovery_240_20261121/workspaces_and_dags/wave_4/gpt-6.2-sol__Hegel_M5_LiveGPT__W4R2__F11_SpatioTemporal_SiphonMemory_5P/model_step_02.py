import math
import numpy as np
from scipy.optimize import least_squares

KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _features(rows):
    return np.array([[float(row[k]) for k in KEYS] for row in rows], dtype=float).T

def _forecast(p, features):
    a, b, vmax, rho, tau = p
    u, x, z, s, H = features
    un = u / 2.0
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    n = max(1, math.ceil(float(np.max(H)) / 0.006))
    dt = H / n

    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z)/tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*dt*k1, z + 0.5*dt*l1)
        k3, l3 = rhs(x + 0.5*dt*k2, z + 0.5*dt*l2)
        k4, l4 = rhs(x + dt*k3, z + dt*l3)
        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        z = z + (dt/6.0)*(l1 + 2*l2 + 2*l3 + l4)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    features = _features(rows)
    y = np.array([float(row['y']) for row in rows])
    best = None
    for tau0 in (0.3, 0.08, 1.2):
        result = least_squares(lambda p: _forecast(p, features) - y,
                               [1.0, 1.0, 6.5, 0.7, tau0],
                               bounds=([0, 0, 0, 0, 0.01], [np.inf, np.inf, np.inf, 1, np.inf]),
                               x_scale='jac', max_nfev=250,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or result.cost < best.cost:
            best = result
        if best.cost < 1e-12:
            break
    _params = best.x.copy()
    return _params

def predict(row) -> float:
    return float(_forecast(_params, _features([row]))[0])
