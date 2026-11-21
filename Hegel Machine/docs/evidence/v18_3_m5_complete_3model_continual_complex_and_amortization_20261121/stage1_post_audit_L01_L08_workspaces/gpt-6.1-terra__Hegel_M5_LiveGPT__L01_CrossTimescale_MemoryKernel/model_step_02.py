import numpy as np
from scipy.optimize import least_squares

_coef = None

def fit(train_rows):
    global _coef
    X = np.array([[r['x_now'], r['mem_conv'], r['u_now']] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    _coef = least_squares(lambda p: X @ p - y, np.zeros(3)).x

def predict(row):
    return float(np.dot(_coef, [row['x_now'], row['mem_conv'], row['u_now']]))