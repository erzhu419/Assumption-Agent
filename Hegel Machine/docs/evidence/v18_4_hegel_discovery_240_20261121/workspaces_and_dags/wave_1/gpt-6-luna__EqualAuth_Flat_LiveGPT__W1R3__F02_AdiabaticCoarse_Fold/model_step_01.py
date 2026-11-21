import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    X = np.array([[float(r['u_now']), 1.0] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    res = least_squares(lambda p: X @ p - y, x0=np.array([1.0, 0.0]))
    return {'params': res.x}

def predict(row):
    p = _MODEL['params']
    return float(p[0] * float(row['u_now']) + p[1])
