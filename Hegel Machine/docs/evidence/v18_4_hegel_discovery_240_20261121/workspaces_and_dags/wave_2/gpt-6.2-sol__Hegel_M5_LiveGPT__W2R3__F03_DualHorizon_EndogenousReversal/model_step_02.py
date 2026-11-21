import math
import numpy as np
from scipy.optimize import least_squares

_p = None

def fit(train_rows):
    global _p
    h, u, y = np.asarray([[r['H'], r['u_now'], r['y']] for r in train_rows], dtype=float).T

    def residual(p):
        a, b, fast, gap = p
        slow = fast + gap
        return u * (-a * np.expm1(-h / fast) + b * np.expm1(-h / slow)) - y

    _p = least_squares(residual, [1.5, 2.3, 0.4, 8.0], bounds=([0, 0, 1e-6, 1e-6], np.inf), x_scale='jac').x


def predict(row) -> float:
    a, b, fast, gap = _p
    h = row['H']
    return float(row['u_now'] * (-a * math.expm1(-h / fast) + b * math.expm1(-h / (fast + gap))))