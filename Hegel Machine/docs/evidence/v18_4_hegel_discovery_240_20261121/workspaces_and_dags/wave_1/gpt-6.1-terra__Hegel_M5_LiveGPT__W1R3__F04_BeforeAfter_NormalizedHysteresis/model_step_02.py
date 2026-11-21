import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    d = np.array([(r['u_now'] - r['u_past']) / 0.004 for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        amplitude, rate, power = p
        return x + d * amplitude * np.power(1.0 + rate * h, -power) - y
    result = least_squares(residual, [2.2, 0.5, 1.5], bounds=([0.0, 1e-8, 1e-8], [100.0, 100.0, 100.0]), max_nfev=10000)
    _params = result.x

def predict(row):
    amplitude, rate, power = _params
    delta = (row['u_now'] - row['u_past']) / 0.004
    return float(row['x_now'] + delta * amplitude * (1.0 + rate * row['H']) ** (-power))