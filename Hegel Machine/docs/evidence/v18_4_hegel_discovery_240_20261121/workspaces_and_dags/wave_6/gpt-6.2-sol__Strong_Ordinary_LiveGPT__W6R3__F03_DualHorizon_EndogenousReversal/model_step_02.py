import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(log_params):
        a_fast, a_slow, t_fast, gap = np.exp(log_params)
        t_slow = t_fast + gap
        return u * (-a_fast * np.expm1(-h / t_fast) + a_slow * np.expm1(-h / t_slow)) - y

    result = least_squares(residual, np.log([1.5, 2.5, 0.4, 10.0]), bounds=(-12.0, 12.0))
    a_fast, a_slow, t_fast, gap = np.exp(result.x)
    _params = (a_fast, a_slow, t_fast, t_fast + gap)

def predict(row):
    a_fast, a_slow, t_fast, t_slow = _params
    h = row['H']
    return float(row['u_now'] * (-a_fast * math.expm1(-h / t_fast) + a_slow * math.expm1(-h / t_slow)))