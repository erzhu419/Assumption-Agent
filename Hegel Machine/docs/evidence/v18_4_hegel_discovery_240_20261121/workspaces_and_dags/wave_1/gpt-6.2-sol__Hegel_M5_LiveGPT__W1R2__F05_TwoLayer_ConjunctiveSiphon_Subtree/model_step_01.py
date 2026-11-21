import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    f, g, z = np.asarray([_features(r) for r in rows], dtype=float).T
    y = np.asarray([r['y'] for r in rows], dtype=float)

    def residual(p):
        v, rho, c = p
        return v * (1.0 - rho * g) * f + c * z - y

    _params = least_squares(residual, [6.5, 0.70, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]), gtol=1e-12).x

def predict(row):
    f, g, z = _features(row)
    v, rho, c = _params
    return float(v * (1.0 - rho * g) * f + c * z)
