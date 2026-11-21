import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * u + p[1] - y
    result = least_squares(residual, x0=np.array([2.1, 0.0]))
    return result.x

def predict(row):
    p = fit(train_rows)
    return float(p[0] * row['u_now'] + p[1])