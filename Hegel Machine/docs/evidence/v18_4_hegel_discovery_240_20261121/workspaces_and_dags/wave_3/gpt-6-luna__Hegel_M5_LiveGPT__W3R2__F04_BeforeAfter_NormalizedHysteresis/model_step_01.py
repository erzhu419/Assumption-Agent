import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    def predict_params(row, p):
        amp, tau = p
        return row['x_now'] + amp * (row['u_now'] - row['u_past']) * np.exp(-row['H'] / tau)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([3.0, 1.3]), bounds=([0.0, 1e-6], [np.inf, np.inf]))
    return result.x

def predict(row):
    amp, tau = fit(_FIT_ROWS)
    return float(row['x_now'] + amp * (row['u_now'] - row['u_past']) * np.exp(-row['H'] / tau))