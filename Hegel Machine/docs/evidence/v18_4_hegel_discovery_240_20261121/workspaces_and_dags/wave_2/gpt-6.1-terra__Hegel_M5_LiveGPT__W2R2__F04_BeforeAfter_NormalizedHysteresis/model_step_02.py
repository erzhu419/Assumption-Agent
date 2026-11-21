import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    d = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        denominator = np.exp(p[0]) + np.exp(p[1]) * h + p[2] * h * h
        return x + d / denominator - y
    result = least_squares(residual, [np.log(2.1), np.log(3.0), 0.0], max_nfev=10000)
    _params = result.x

def predict(row):
    h = float(row['H'])
    denominator = np.exp(_params[0]) + np.exp(_params[1]) * h + _params[2] * h * h
    return float(row['x_now'] + (row['u_now'] - row['u_past']) / denominator)
