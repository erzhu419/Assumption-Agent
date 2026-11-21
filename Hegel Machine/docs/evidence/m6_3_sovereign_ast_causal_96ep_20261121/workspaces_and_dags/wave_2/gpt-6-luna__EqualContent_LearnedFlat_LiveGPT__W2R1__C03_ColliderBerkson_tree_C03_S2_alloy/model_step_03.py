import math
import numpy as np
from scipy.optimize import least_squares
_PARAMS = (3.0, 4.0)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    X = np.array([[math.log1p(float(r['u_1'])), float(r['u_2']) / (1.0 + float(r['u_2']))] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    res = least_squares(lambda p: X @ p - y, x0=np.array([3.0, 4.0]))
    _PARAMS = tuple(float(v) for v in res.x)
    return predict

def predict(row):
    a1, a2 = _PARAMS
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return a1 * math.log1p(u1) + a2 * u2 / (1.0 + u2)