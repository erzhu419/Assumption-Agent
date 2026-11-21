import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    h = np.array([r['H'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(log_params):
        af, a_s, tf, gap = np.exp(log_params)
        ts = tf + gap
        return u * (-af * np.expm1(-h / tf) + a_s * np.expm1(-h / ts)) - y

    start = np.log([1.5, 2.5, 0.4, 10.0])
    result = least_squares(residual, start, bounds=(-12.0, 12.0), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    af, a_s, tf, gap = np.exp(result.x)
    _params = tuple(map(float, (af, a_s, tf, tf + gap)))
    return _params

def predict(row) -> float:
    af, a_s, tf, ts = _params
    h = float(row['H'])
    return float(row['u_now']) * (-af * math.expm1(-h / tf) + a_s * math.expm1(-h / ts))
