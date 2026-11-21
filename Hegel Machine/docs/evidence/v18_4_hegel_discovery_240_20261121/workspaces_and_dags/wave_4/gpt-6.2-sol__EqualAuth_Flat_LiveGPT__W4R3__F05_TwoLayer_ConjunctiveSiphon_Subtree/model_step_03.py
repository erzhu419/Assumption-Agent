import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    g = s / (1.0 + s)
    root_u = np.sqrt(u)

    def residual(p):
        a, k, rho, c = p
        return a * (1.0 - rho * g) * u / (k + u) + c * root_u - y

    _params = least_squares(residual, [6.0, 0.06, 0.7, 2.0], bounds=([0.0, 1e-8, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row):
    a, k, rho, c = _params
    u, s = row['u_now'], row['s_grad']
    return float(a * (1.0 - rho * s / (1.0 + s)) * u / (k + u) + c * math.sqrt(u))
