import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * u + p[1] - y
    p0 = np.array([2.0, 0.0], dtype=float)
    p = least_squares(residual, p0).x
    return {'slope': float(p[0]), 'intercept': float(p[1])}

def predict(row):
    p = _FIT
    return float(p['slope'] * float(row['u_now']) + p['intercept'])