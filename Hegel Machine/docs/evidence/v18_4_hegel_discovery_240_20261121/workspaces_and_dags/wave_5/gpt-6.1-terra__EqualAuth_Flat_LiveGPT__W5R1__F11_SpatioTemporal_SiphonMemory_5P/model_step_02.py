import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x, z = float(row['x_now']), float(row['mem_conv'])
    h = float(row['H']) / 40.0
    def f(x, z):
        return -a*x + b*z + J, (0.52*x - z)/tau
    for _ in range(40):
        k1, l1 = f(x, z)
        k2, l2 = f(x + h*k1/2, z + h*l1/2)
        k3, l3 = f(x + h*k2/2, z + h*l2/2)
        k4, l4 = f(x + h*k3, z + h*l3)
        x += h*(k1 + 2*k2 + 2*k3 + k4)/6
        z += h*(l1 + 2*l2 + 2*l3 + l4)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    best = None
    for tau0 in (0.15, 0.5, 2.0):
        result = least_squares(residual, [0.3, 0.5, 3.0, 0.5, tau0], bounds=([0, 0, 0, 0, 0.03], [10, 10, 30, 2, 20]), max_nfev=1500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_forecast(row, _params))