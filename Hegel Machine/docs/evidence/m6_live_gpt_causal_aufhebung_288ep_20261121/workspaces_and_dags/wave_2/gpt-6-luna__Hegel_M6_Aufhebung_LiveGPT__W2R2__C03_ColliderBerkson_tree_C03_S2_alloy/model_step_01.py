import math
import numpy as np
from scipy.optimize import least_squares

_COEFS = None

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return math.log1p(u1), u2 / (1.0 + u2)

def fit(train_rows):
    global _COEFS
    X = np.asarray([_features(r) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda a: X @ a - y, x0=np.zeros(2))
    _COEFS = result.x
    return _COEFS

def predict(row) -> float:
    if _COEFS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    f1, f2 = _features(row)
    return float(_COEFS[0] * f1 + _COEFS[1] * f2)