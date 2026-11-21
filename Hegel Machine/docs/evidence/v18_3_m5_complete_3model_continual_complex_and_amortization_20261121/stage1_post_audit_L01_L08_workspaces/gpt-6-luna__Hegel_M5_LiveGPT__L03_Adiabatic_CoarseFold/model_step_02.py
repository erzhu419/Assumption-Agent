import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        c0, beta = p
        return c0 * u * (1.0 + beta * np.cbrt(H)) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0]))
    return result.x

def predict(row):
    c0, beta = fit([row])
    return float(c0 * row['u_now'] * (1.0 + beta * np.cbrt(row['H'])))