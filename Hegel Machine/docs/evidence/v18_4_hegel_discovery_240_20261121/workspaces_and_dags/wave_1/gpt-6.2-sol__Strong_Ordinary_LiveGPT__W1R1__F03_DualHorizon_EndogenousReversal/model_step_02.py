import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        a, b, log_fast, log_gap = p
        fast = math.exp(log_fast)
        slow = fast + math.exp(log_gap)
        gain = -a * np.expm1(-h / fast) + b * np.expm1(-h / slow)
        return u * gain - y

    result = least_squares(residual, [1.6, 2.7, math.log(0.4), math.log(13.0)], bounds=([0.0, 0.0, -20.0, -20.0], [np.inf, np.inf, 20.0, 20.0]))
    a, b, log_fast, log_gap = result.x
    fast = math.exp(log_fast)
    _params = (a, b, fast, fast + math.exp(log_gap))

def predict(row) -> float:
    a, b, fast, slow = _params
    h = row['H']
    return float(row['u_now'] * (-a * math.expm1(-h / fast) + b * math.expm1(-h / slow)))