import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)


def fit(train_rows):
    global _params
    base, sink, local = np.array([_features(row) for row in train_rows], dtype=float).T
    y = np.array([row['y'] for row in train_rows], dtype=float)

    def residual(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * sink) * base + c_loc * local - y

    _params = least_squares(residual, [6.5, 0.70, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x


def predict(row) -> float:
    base, sink, local = _features(row)
    vmax, rho, c_loc = _params
    return float(vmax * (1.0 - rho * sink) * base + c_loc * local)
