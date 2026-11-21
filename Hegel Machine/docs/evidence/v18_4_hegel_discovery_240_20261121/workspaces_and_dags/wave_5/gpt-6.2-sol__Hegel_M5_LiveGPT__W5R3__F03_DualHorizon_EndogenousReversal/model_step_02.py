import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float)
    u, h, y = data.T

    def residual(log_params):
        a_fast, t_fast, a_slow, gap = np.exp(log_params)
        t_slow = t_fast + gap
        fast = -np.expm1(-h / t_fast)
        slow = -np.expm1(-h / t_slow)
        return u * (a_fast * fast - a_slow * slow) - y

    result = least_squares(residual, np.log([1.3, 0.3, 2.6, 13.0]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    a_fast, t_fast, a_slow, gap = map(float, np.exp(result.x))
    _params = (a_fast, t_fast, a_slow, t_fast + gap)
    return _params

def predict(row):
    a_fast, t_fast, a_slow, t_slow = _params
    h = float(row['H'])
    fast = -math.expm1(-h / t_fast)
    slow = -math.expm1(-h / t_slow)
    return float(row['u_now']) * (a_fast * fast - a_slow * slow)
