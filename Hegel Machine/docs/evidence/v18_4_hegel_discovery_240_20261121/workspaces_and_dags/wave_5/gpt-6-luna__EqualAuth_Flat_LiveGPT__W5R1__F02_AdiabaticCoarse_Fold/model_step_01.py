import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return p[0] * u + p[1] - y
    result = least_squares(residual, np.array([2.2, 0.0]))
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(p[0] * float(row['u_now']) + p[1])