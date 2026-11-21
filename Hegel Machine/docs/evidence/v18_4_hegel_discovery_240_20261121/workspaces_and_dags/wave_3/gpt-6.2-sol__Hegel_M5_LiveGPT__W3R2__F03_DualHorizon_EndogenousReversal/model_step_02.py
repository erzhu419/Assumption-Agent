import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(log_params):
        af, tf, a_s, gap = np.exp(log_params)
        ts = tf + gap
        return u * (-af * np.expm1(-h / tf) + a_s * np.expm1(-h / ts)) - y

    result = least_squares(residual, np.log([1.6, 0.4, 2.7, 11.6]))
    af, tf, a_s, gap = np.exp(result.x)
    _params = (af, tf, a_s, tf + gap)

def predict(row) -> float:
    af, tf, a_s, ts = _params
    h = row['H']
    return float(row['u_now'] * (-af * math.expm1(-h / tf) + a_s * math.expm1(-h / ts)))