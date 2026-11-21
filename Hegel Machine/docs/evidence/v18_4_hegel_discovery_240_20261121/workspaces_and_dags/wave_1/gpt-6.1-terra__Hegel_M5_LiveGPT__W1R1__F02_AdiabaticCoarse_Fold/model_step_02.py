import numpy as np
from scipy.optimize import least_squares

_slope = None

def fit(train_rows):
    global _slope
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: p[0] * u - y, [2.0])
    _slope = float(result.x[0])

def predict(row):
    return float(_slope * row['u_now'])
