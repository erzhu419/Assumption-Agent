import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    global params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * u * (-np.expm1(-p[1] * h)) - y
    result = least_squares(residual, [1.23, 2.0], bounds=([0.0, 0.0], [np.inf, np.inf]))
    params = result.x

def predict(row):
    return float(params[0] * row['u_now'] * (-np.expm1(-params[1] * row['H'])))