import math
import numpy as np
from scipy.optimize import least_squares

def predict(row, coeffs=None):
    if coeffs is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    a1, a2 = coeffs
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return float(a1 * math.log1p(u1) + a2 * (u2 / (1.0 + u2)))

def fit(train_rows):
    X = np.asarray([[math.log1p(float(r['u_1'])), float(r['u_2']) / (1.0 + float(r['u_2']))] for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda c: X @ c - y, x0=np.zeros(2))
    predict.__defaults__ = (tuple(float(v) for v in result.x),)
    return predict
