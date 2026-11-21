import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.array([[r['H'], r['u_now'], r['y']] for r in train_rows], dtype=float).T

    def residual(p):
        a, b, fast, gap = np.exp(p)
        slow = fast + gap
        return u * (-a * np.expm1(-h / fast) + b * np.expm1(-h / slow)) - y

    result = least_squares(residual, np.log([1.4, 2.4, 0.4, 12.0]))
    a, b, fast, gap = map(float, np.exp(result.x))
    _params = (a, b, fast, fast + gap)

def predict(row) -> float:
    a, b, fast, slow = _params
    h = float(row['H'])
    return float(row['u_now']) * (-a * math.expm1(-h / fast) + b * math.expm1(-h / slow))