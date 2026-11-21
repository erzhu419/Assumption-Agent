import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(rows, p):
    a, b, vmax, rho, tau = p
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float) / 0.05
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float) / 300.0
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    def deriv(x, z):
        return -a*x + b*z + j, (0.52*x - z)/tau
    for _ in range(300):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*h*k1x, z + 0.5*h*k1z)
        k3x, k3z = deriv(x + 0.5*h*k2x, z + 0.5*h*k2z)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x = x + h*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z = z + h*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _evaluate(rows, p) - y,
                           x0=[1.0, 1.0, 6.5, 0.70, 0.05],
                           bounds=([0.0, 0.0, 1.0, 0.05, 0.005],
                                   [10.0, 10.0, 15.0, 0.95, 2.0]),
                           max_nfev=500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x

def predict(row):
    return float(_evaluate([row], _params)[0])