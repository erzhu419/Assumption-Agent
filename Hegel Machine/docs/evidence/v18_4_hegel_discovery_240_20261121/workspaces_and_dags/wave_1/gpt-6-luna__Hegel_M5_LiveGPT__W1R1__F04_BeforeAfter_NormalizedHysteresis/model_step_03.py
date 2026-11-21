import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def residual(p):
        gain, tau = p
        return np.array([
            r['y'] - (r['x_now'] + gain * (r['u_now'] - r['u_past']) * np.exp(-r['H'] / tau))
            for r in rows
        ])
    result = least_squares(residual, x0=np.array([2.7, 1.6]), bounds=([0.0, 1e-6], [100.0, 100.0]))
    return result.x

def predict(row):
    gain, tau = fit(_MODEL_TRAIN_ROWS)
    return float(row['x_now'] + gain * (row['u_now'] - row['u_past']) * np.exp(-row['H'] / tau))