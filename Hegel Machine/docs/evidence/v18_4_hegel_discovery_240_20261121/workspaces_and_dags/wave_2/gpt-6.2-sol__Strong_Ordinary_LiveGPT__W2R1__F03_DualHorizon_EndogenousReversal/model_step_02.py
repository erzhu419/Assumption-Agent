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
        af, a_s, tf, gap = p
        ts = tf + gap
        return u * (-af * np.expm1(-h / tf) + a_s * np.expm1(-h / ts)) - y

    result = least_squares(residual, [1.5, 2.5, 0.5, 10.0], bounds=([0.0, 0.0, 1e-6, 1e-6], np.inf))
    _params = result.x

def predict(row) -> float:
    af, a_s, tf, gap = _params
    h = row['H']
    return float(row['u_now'] * (-af * math.expm1(-h / tf) + a_s * math.expm1(-h / (tf + gap))))
