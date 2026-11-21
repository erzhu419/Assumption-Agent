import numpy as np
from scipy.optimize import least_squares

_coef = None

def fit(train_rows):
    global _coef
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    _coef = float(least_squares(lambda p: p[0] * u - y, [2.0]).x[0])

def predict(row):
    return float(_coef * float(row['u_now']))
