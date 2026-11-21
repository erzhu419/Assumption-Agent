import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, r):
        E, eta = p
        z = E * r['H'] / eta
        return (r['u_now'] / E) * (-math.expm1(-z))
    def residual(p):
        return np.array([pred(p, r) for r in train_rows]) - y
    res = least_squares(residual, x0=np.array([1.0, 1.0]), bounds=([1e-8, 1e-8], [100.0, 1e6]))
    return tuple(res.x)

def predict(row):
    E, eta = fit(_TRAIN_ROWS)
    return (row['u_now'] / E) * (-math.expm1(-E * row['H'] / eta))