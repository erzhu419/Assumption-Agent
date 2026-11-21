import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    vmax, rho, a, b, tau = p
    u = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x, z = float(row['x_now']), float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(x, z):
        return -a*x + b*z + flux, (0.52*x-z)/tau
    for _ in range(n):
        k1, l1 = f(x, z)
        k2, l2 = f(x + dt*k1/2, z + dt*l1/2)
        k3, l3 = f(x + dt*k2/2, z + dt*l2/2)
        k4, l4 = f(x + dt*k3, z + dt*l3)
        x += dt*(k1 + 2*k2 + 2*k3 + k4)/6
        z += dt*(l1 + 2*l2 + 2*l3 + l4)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.008], [30.0, 2.0, 20.0, 30.0, 2.0])
    best = None
    for tau in (0.05, 0.15, 0.4):
        result = least_squares(residual, [3.0, 0.7, 1.0, 1.0, tau], bounds=bounds, max_nfev=1200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_forecast(row, _params))