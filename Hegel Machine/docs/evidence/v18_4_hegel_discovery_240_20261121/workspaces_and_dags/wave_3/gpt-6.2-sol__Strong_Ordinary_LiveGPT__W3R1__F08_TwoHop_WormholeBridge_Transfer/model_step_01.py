import math
import numpy as np
from scipy.optimize import least_squares

# Transfer the midpoint of the archived sink response after normalizing at s_grad=1.
_q = 0.5 * (0.69 / (1.0 - 0.69 / 2.0) + 0.72 / (1.0 - 0.72 / 2.0))
_p = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u / (1.2 + u) * (1.0 - _q * (s / (1.0 + s) - 0.5)), math.sqrt(u)

def fit(train_rows):
    global _p
    X = np.asarray([_features(row) for row in train_rows], dtype=float)
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    _p = least_squares(lambda p: X @ p - y, [5.0, 0.3]).x

def predict(row) -> float:
    a, b = _features(row)
    return float(_p[0] * a + _p[1] * b)
