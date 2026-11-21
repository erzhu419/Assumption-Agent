import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, h, y = np.array([[r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float).T

    def residual(p):
        a_fast, a_slow, log_fast, log_gap = p
        t_fast = math.exp(log_fast)
        t_slow = t_fast + math.exp(log_gap)
        gain = -a_fast * np.expm1(-h / t_fast) + a_slow * np.expm1(-h / t_slow)
        return u * gain - y

    a_fast, a_slow, log_fast, log_gap = least_squares(
        residual, [2.0, 3.0, math.log(0.5), math.log(9.5)],
        bounds=([0.0, 0.0, -np.inf, -np.inf], [np.inf] * 4),
    ).x
    t_fast = math.exp(log_fast)
    _params = (a_fast, a_slow, t_fast, t_fast + math.exp(log_gap))

def predict(row) -> float:
    a_fast, a_slow, t_fast, t_slow = _params
    h = float(row['H'])
    return float(row['u_now'] * (-a_fast * math.expm1(-h / t_fast) + a_slow * math.expm1(-h / t_slow)))
