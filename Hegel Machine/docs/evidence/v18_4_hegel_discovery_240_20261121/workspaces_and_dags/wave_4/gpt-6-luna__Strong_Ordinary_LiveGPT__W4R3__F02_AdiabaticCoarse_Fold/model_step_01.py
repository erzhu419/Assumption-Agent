import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return p[0] + p[1] * u - y
    result = least_squares(residual, x0=np.array([0.0, 2.0]))
    return result.x

def predict(row):
    p = fit(train_rows)
    return float(p[0] + p[1] * float(row['u_now']))