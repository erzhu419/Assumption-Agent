import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    q, g, root = np.asarray([_features(row) for row in rows], dtype=float).T
    y = np.asarray([row['y'] for row in rows], dtype=float)

    def residual(p):
        vmax, rho, sink = p
        return vmax * q * (1.0 - rho * g) - sink * root - y

    _params = least_squares(residual, [5.0, 0.8, 0.2]).x


def predict(row) -> float:
    q, g, root = _features(row)
    vmax, rho, sink = _params
    return float(vmax * q * (1.0 - rho * g) - sink * root)
