import math
import numpy as np
from scipy.optimize import least_squares

_COEF = None

def fit(train_rows):
    global _COEF
    X = np.asarray([[math.log1p(float(r['u_1'])), float(r['u_2']), 1.0] for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda beta: X @ beta - y, x0=np.zeros(3))
    _COEF = result.x
    return lambda row: predict(row)

def predict(row) -> float:
    if _COEF is None:
        raise RuntimeError('Call fit(train_rows) first.')
    return float(_COEF[0] * math.log1p(float(row['u_1'])) + _COEF[1] * float(row['u_2']) + _COEF[2])