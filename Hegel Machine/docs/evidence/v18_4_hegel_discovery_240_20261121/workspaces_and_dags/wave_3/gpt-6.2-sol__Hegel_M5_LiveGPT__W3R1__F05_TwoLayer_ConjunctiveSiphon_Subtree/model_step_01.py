import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    z = np.asarray([[r['u_now'], r['s_grad'], r['y']] for r in train_rows], dtype=float)
    u, s, y = z.T
    n = u / 0.05
    b = n / (1.2 + n)
    g = s / (1.0 + s)
    q = np.sqrt(n)

    def residual(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * g) * b + c * q - y

    _params = least_squares(residual, [6.5, 0.70, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x
    return _params

def predict(row) -> float:
    vmax, rho, c = _params
    n = row['u_now'] / 0.05
    s = row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (n / (1.2 + n)) + c * math.sqrt(n))
