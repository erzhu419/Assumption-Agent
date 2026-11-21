import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, h):
    af, tf, a_s, gap = np.exp(p)
    ts = tf + gap
    return u * (-af * np.expm1(-h / tf) + a_s * np.expm1(-h / ts))

def fit(train_rows):
    global _params
    u, h, y = np.asarray([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(lambda p: _response(p, u, h) - y,
                            np.log([2.0, 0.6, 2.7, 10.0])).x

def predict(row) -> float:
    return float(_response(_params, row['u_now'], row['H']))
