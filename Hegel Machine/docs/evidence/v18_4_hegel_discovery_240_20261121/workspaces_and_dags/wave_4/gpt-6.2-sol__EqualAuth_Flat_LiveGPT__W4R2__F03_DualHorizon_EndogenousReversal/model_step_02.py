import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, h, y = np.array([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T

    def residual(p):
        a, b, tf, gap = p
        ts = tf + gap
        return u * (-a * np.expm1(-h / tf) + b * np.expm1(-h / ts)) - y

    _params = least_squares(residual, [1.5, 2.0, 0.5, 10.0], bounds=(1e-8, np.inf)).x
    return _params

def predict(row) -> float:
    a, b, tf, gap = _params
    h = row['H']
    return float(row['u_now'] * (-a * math.expm1(-h / tf) + b * math.expm1(-h / (tf + gap))))