import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        E, eta = np.exp(p)
        return (u / E) * (1.0 - np.exp(-E * H / eta))
    result = least_squares(lambda p: pred(p) - y, np.log([0.48, 0.2]))
    E, eta = np.exp(result.x)
    return float(E), float(eta)

def predict(row):
    E, eta = fit(_TRAIN_ROWS)
    return float((row['u_now'] / E) * (1.0 - math.exp(-E * row['H'] / eta)))