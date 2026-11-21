import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _inputs(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    a, g, b = np.array([_inputs(r) for r in rows], dtype=float).T
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * g) * a + c * b - y
    _params = least_squares(residual, [6.5, 0.5, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x

def predict(row) -> float:
    a, g, b = _inputs(row)
    vmax, rho, c = _params
    return float(vmax * (1.0 - rho * g) * a + c * b)
