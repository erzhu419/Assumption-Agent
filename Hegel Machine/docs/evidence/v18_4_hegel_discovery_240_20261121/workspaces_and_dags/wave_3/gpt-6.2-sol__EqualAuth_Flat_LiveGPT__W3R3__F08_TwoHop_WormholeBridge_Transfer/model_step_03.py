import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _transfer(u, s, p):
    a, k, b, m, c = p
    return -a * math.expm1(-k * u) + b * u / ((m + u) * (1.0 + c * s))

def fit(train_rows):
    global _params
    data = [(float(r['u_now']), float(r['s_grad']), float(r['y'])) for r in train_rows]
    def residuals(p):
        return np.array([_transfer(u, s, p) - y for u, s, y in data])
    _params = least_squares(residuals, [1.0, 1.0, 5.0, 1.0, 1.0], bounds=(1e-8, np.inf)).x

def predict(row) -> float:
    return float(_transfer(float(row['u_now']), float(row['s_grad']), _params))
