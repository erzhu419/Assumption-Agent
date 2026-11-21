import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def unpack(q):
        af, a_s, tf, gap = np.exp(q)
        return af, a_s, tf, tf + gap

    def residual(q):
        af, a_s, tf, ts = unpack(q)
        return u * (-af * np.expm1(-h / tf) + a_s * np.expm1(-h / ts)) - y

    result = least_squares(residual, np.log([1.5, 2.5, 0.5, 12.0]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = unpack(result.x)


def predict(row):
    af, a_s, tf, ts = _params
    h = float(row['H'])
    return float(row['u_now']) * (-af * math.expm1(-h / tf) + a_s * math.expm1(-h / ts))
