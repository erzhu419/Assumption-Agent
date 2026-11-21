import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _unpack(q):
    af, tf, a_slow, gap = np.exp(q)
    return af, tf, a_slow, tf + gap

def fit(train_rows):
    global _params
    u, h, y = np.asarray([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T

    def residual(q):
        af, tf, a_slow, ts = _unpack(q)
        return u * (-af * np.expm1(-h / tf) + a_slow * np.expm1(-h / ts)) - y

    initial = np.log([1.6, 0.5, 2.7, 12.0])
    result = least_squares(residual, initial, bounds=(-14.0, 14.0), x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = tuple(float(p) for p in _unpack(result.x))
    return _params

def predict(row) -> float:
    af, tf, a_slow, ts = _params
    h = float(row['H'])
    return float(row['u_now']) * (-af * math.expm1(-h / tf) + a_slow * math.expm1(-h / ts))