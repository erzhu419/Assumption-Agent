import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None

def _forecast(p, rows):
    vmax, rho, a, b, tau = p
    u, x, z, s, horizon = rows.T
    u = u / 2.0
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, math.ceil(float(np.max(horizon)) / 0.004))
    h = horizon / n

    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        z = z + (h/6.0)*(l1 + 2*l2 + 2*l3 + l4)
    return x

def fit(train_rows):
    global _params
    rows = np.array([[r[k] for k in _KEYS] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    bounds = ([0, 0, 0, 0, 0.004], [np.inf, 1, np.inf, np.inf, np.inf])
    def solve(tau0):
        return least_squares(lambda p: _forecast(p, rows) - y,
                             [3.0, 0.7, 0.8, 1.0, tau0],
                             bounds=bounds, x_scale='jac', max_nfev=250,
                             ftol=1e-10, xtol=1e-10, gtol=1e-10)
    best = solve(0.25)
    if best.cost > 1e-8:
        best = min((best, solve(0.06), solve(0.9)), key=lambda result: result.cost)
    _params = best.x

def predict(row):
    rows = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_forecast(_params, rows)[0])