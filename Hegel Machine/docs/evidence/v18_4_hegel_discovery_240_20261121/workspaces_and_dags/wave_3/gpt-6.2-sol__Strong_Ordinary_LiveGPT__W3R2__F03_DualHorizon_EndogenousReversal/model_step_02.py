import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, h, y = np.asarray([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T

    def residual(p):
        af, a_s, tf, gap = p
        ts = tf + gap
        return u * (-af * np.expm1(-h / tf) + a_s * np.expm1(-h / ts)) - y

    _params = least_squares(residual, [1.5, 2.5, 0.4, 11.0], bounds=(1e-8, np.inf), x_scale='jac').x

def predict(row):
    af, a_s, tf, gap = _params
    h = float(row['H'])
    return float(row['u_now'] * (-af * math.expm1(-h / tf) + a_s * math.expm1(-h / (tf + gap))))
