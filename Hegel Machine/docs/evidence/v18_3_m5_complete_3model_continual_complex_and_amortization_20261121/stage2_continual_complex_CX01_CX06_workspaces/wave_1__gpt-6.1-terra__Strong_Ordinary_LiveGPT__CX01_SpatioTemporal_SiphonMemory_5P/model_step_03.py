import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, tau, vmax, rho, branch = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + branch * math.sqrt(max(u, 0.0))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(x, z):
        return -a*x + b*z + drive, (0.52*x-z)/tau
    for _ in range(n):
        k1, l1 = deriv(x, z)
        k2, l2 = deriv(x + dt*k1/2, z + dt*l1/2)
        k3, l3 = deriv(x + dt*k2/2, z + dt*l2/2)
        k4, l4 = deriv(x + dt*k3, z + dt*l3)
        x += dt*(k1 + 2*k2 + 2*k3 + k4)/6
        z += dt*(l1 + 2*l2 + 2*l3 + l4)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    bounds = ([0.0, 0.0, 0.03, 0.0, 0.69, -10.0], [10.0, 10.0, 10.0, 20.0, 0.72, 10.0])
    best = None
    for p0 in ([0.7, 0.5, 0.5, 3.0, 0.70, 0.2], [1.5, 1.0, 1.5, 5.0, 0.70, -0.3], [0.3, 0.2, 0.15, 2.0, 0.70, 0.5]):
        result = least_squares(residual, p0, bounds=bounds, max_nfev=1200, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_forecast(row, _params))