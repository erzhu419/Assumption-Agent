import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(row, p):
    vmax, rho, k, branch = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (k + u) + branch * math.sqrt(max(0.0, u))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_law(r, p) for r in rows]) - y
    _params = least_squares(residual, [8.0, 0.8, 1.2, -0.3], bounds=([0.0, 0.0, 1e-8, -np.inf], [np.inf, np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    return float(_law(row, _params))