import math
import numpy as np
from scipy.optimize import least_squares

_coefficients = None

def fit(train_rows):
    global _coefficients
    X = np.array([[math.log1p(float(r['u_1'])), float(r['u_2']) / (1.0 + float(r['u_2']))] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    _coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

def predict(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return float(_coefficients[0] * math.log1p(u1) + _coefficients[1] * u2 / (1.0 + u2))
