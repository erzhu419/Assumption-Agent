import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def pred(p):
        k, a = p
        decay = np.exp(-a * H)
        return k * u + (x - k * u) * decay

    result = least_squares(lambda p: pred(p) - y, x0=np.array([2.0, 0.8]), bounds=([0.0, 1e-8], [20.0, 20.0]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    k, a = _PARAMS
    decay = math.exp(-a * float(row['H']))
    u = float(row['u_now'])
    x = float(row['x_now'])
    return k * u + (x - k * u) * decay