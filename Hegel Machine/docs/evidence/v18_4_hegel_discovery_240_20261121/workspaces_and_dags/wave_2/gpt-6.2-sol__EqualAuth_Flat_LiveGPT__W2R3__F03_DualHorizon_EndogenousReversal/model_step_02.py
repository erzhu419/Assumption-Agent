import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float)
    u, h, y = data.T

    def residual(p):
        af, tf, a_s, gap = p
        ts = tf + gap
        return u * (-af * np.expm1(-h / tf) + a_s * np.expm1(-h / ts)) - y

    result = least_squares(residual, [1.5, 0.5, 2.5, 12.0],
                           bounds=([0, 1e-6, 0, 1e-6], np.inf),
                           x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = tuple(result.x)
    return _params

def predict(row) -> float:
    af, tf, a_s, gap = _params
    h = float(row['H'])
    return float(row['u_now'] * (-af * math.expm1(-h / tf) + a_s * math.expm1(-h / (tf + gap))))
