import math
import numpy as np
from scipy.optimize import least_squares

_COEF = None

def fit(train_rows):
    global _COEF
    X = np.array([[math.log1p(float(r['u_1'])), float(r['u_2']) / (1.0 + float(r['u_2']))] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda c: X @ c - y, x0=np.zeros(2))
    _COEF = result.x
    return _COEF.copy()

def predict(row) -> float:
    if _COEF is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_COEF[0] * math.log1p(float(row['u_1'])) + _COEF[1] * float(row['u_2']) / (1.0 + float(row['u_2'])))