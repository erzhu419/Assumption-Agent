import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    res = least_squares(lambda p: p[0] * u + p[1] - y, x0=np.array([1.0, 0.0]))
    return res.x

def predict(row):
    p = fit(_train_rows)
    return float(p[0] * row['u_now'] + p[1])