import math
import numpy as np
from scipy.optimize import least_squares

def _pred(p, row):
    a, k = p
    h = float(row['H'])
    return k * float(row['u_now']) + (float(row['x_now']) - k * float(row['u_now'])) * math.exp(-a * h)

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_pred(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([0.2, 1.0]), bounds=([1e-8, -100.0], [100.0, 100.0]))
    return result.x

def predict(row):
    return float(_pred(fit(_TRAIN_ROWS), row))