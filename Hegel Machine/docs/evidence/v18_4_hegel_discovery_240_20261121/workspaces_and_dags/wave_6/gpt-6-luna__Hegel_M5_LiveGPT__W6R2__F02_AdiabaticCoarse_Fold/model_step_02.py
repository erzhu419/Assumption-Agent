import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        E, eta = np.exp(p)
        return (u / E) * (-np.expm1(-E * H / eta))
    result = least_squares(lambda p: pred(p) - y, np.log([0.4, 100.0]))
    return np.exp(result.x)

def predict(row):
    E, eta = fit(_TRAIN_ROWS)
    return (row['u_now'] / E) * (-math.expm1(-E * row['H'] / eta))
