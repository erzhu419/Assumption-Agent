import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    d = np.array([(r['u_now'] - r['u_past']) / 0.02 for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * x + p[1] * d * np.exp(-h / np.exp(p[2])) - y
    result = least_squares(residual, [1.0, -2.0, np.log(0.5)])
    _params = result.x

def predict(row):
    a, gain, log_tau = _params
    return float(a * row['x_now'] + gain * ((row['u_now'] - row['u_past']) / 0.02) * np.exp(-row['H'] / np.exp(log_tau)))
