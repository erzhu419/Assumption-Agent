import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, h, y = np.asarray([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T

    def residual(p):
        a_fast, a_slow, t_fast, gap = p
        fast = -np.expm1(-h / t_fast)
        slow = -np.expm1(-h / (t_fast + gap))
        return u * (a_fast * fast - a_slow * slow) - y

    result = least_squares(residual, [1.5, 2.5, 0.4, 10.0], bounds=([0.0, 0.0, 1e-8, 1e-8], np.inf))
    _params = tuple(result.x)

def predict(row) -> float:
    a_fast, a_slow, t_fast, gap = _params
    h = float(row['H'])
    fast = -math.expm1(-h / t_fast)
    slow = -math.expm1(-h / (t_fast + gap))
    return float(row['u_now'] * (a_fast * fast - a_slow * slow))
