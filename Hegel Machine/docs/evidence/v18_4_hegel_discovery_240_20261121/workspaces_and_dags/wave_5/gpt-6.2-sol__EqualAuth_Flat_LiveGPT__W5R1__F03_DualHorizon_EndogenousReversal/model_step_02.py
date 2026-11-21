import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.array([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(q):
        a, b, tf, gap = np.exp(q)
        ts = tf + gap
        return u * (-a * np.expm1(-h / tf) + b * np.expm1(-h / ts)) - y

    result = least_squares(residual, np.log([1.0, 2.0, 0.5, 10.0]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    a, b, tf, gap = np.exp(result.x)
    _params = (a, b, tf, tf + gap)

def predict(row) -> float:
    a, b, tf, ts = _params
    h = row['H']
    return float(row['u_now'] * (-a * math.expm1(-h / tf) + b * math.expm1(-h / ts)))
