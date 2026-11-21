import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(1, int(math.ceil(abs(H) / 0.01)))
    dt = H / n
    def rhs(x, z):
        return -a*x + b*z + J, (0.52*x - z)/tau
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + dt*k1/2, z + dt*l1/2)
        k3, l3 = rhs(x + dt*k2/2, z + dt*l2/2)
        k4, l4 = rhs(x + dt*k3, z + dt*l3)
        x += dt*(k1 + 2*k2 + 2*k3 + k4)/6
        z += dt*(l1 + 2*l2 + 2*l3 + l4)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    best = None
    for tau in (0.2, 0.6, 1.5):
        result = least_squares(residual, [0.5, 0.5, 6.5, 0.7, tau], bounds=([0.0, 0.0, 0.0, 0.0, 0.05], [10.0, 10.0, 30.0, 1.0, 10.0]), max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_forecast(row, _params))