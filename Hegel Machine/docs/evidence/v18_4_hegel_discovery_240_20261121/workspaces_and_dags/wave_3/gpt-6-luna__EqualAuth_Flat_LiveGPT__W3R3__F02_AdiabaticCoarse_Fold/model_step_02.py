import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    res = least_squares(lambda p: p[0] * u - y, x0=np.array([1.0]))
    return {'k': float(res.x[0])}

def predict(row):
    return float(row['u_now'] * _MODEL['k'])