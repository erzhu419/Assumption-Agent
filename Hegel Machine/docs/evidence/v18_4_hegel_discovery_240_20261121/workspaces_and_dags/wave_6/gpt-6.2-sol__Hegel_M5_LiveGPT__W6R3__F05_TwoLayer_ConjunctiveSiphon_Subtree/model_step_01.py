import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = row['u_now'] / 0.005
    s = row['s_grad']
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)


def fit(train_rows):
    global _params
    a, g, b = np.array([_features(r) for r in train_rows], dtype=float).T
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * g) * a + c * b - y
    _params = least_squares(residual, [6.5, 0.70, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x


def predict(row):
    vmax, rho, c = _params
    a, g, b = _features(row)
    return float(vmax * (1.0 - rho * g) * a + c * b)
