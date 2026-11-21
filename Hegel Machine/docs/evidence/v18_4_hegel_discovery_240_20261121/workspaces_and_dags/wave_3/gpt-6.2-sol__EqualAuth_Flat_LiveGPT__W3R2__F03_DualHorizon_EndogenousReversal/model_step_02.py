import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(p):
        a_fast, a_slow, t_fast, gap = np.exp(p)
        t_slow = t_fast + gap
        return u * (-a_fast * np.expm1(-h / t_fast) + a_slow * np.expm1(-h / t_slow)) - y

    result = least_squares(residual, np.log([1.0, 2.0, 0.5, 10.0]))
    a_fast, a_slow, t_fast, gap = np.exp(result.x)
    global _params
    _params = (a_fast, a_slow, t_fast, t_fast + gap)

def predict(row) -> float:
    a_fast, a_slow, t_fast, t_slow = _params
    h = row['H']
    return float(row['u_now'] * (-a_fast * math.expm1(-h / t_fast) + a_slow * math.expm1(-h / t_slow)))