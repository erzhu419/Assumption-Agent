import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(log_params):
        a_fast, a_slow, t_fast, gap = np.exp(log_params)
        t_slow = t_fast + gap
        return u * (-a_fast * np.expm1(-h / t_fast) + a_slow * np.expm1(-h / t_slow)) - y

    result = least_squares(residual, np.log([1.5, 2.5, 0.5, 10.0]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = np.exp(result.x)
    return _params

def predict(row) -> float:
    a_fast, a_slow, t_fast, gap = _params
    h = row['H']
    return float(row['u_now'] * (-a_fast * math.expm1(-h / t_fast) + a_slow * math.expm1(-h / (t_fast + gap))))