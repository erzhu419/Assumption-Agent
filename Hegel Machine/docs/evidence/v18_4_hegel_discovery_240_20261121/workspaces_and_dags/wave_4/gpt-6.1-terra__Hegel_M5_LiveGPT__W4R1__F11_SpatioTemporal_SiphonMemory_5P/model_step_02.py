import numpy as np
from scipy.optimize import least_squares

params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    h = float(row['H']) / 80.0
    def deriv(x, z):
        return -a*x + b*z + j, (0.52*x - z)/tau
    for _ in range(80):
        k1, l1 = deriv(x, z)
        k2, l2 = deriv(x + h*k1/2, z + h*l1/2)
        k3, l3 = deriv(x + h*k2/2, z + h*l2/2)
        k4, l4 = deriv(x + h*k3, z + h*l3)
        x += h*(k1 + 2*k2 + 2*k3 + k4)/6
        z += h*(l1 + 2*l2 + 2*l3 + l4)/6
    return x

def fit(train_rows):
    global params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    result = least_squares(residual, [0.8, 0.5, 6.5, 0.7, 0.5], bounds=([0, 0, 0, 0, 0.05], [10, 10, 30, 1, 10]), max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    params = result.x
    return params

def predict(row):
    return float(_forecast(row, params))