import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    up = np.array([r['u_past'] for r in train_rows], dtype=float)
    un = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        gain, tau, k_eq = p
        pred = gain * (un - up) * np.exp(-H / tau) + k_eq * un
        return pred - y
    result = least_squares(residual, x0=np.array([2.0, 1.0, 0.5]), bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    gain, tau, k_eq = fit(train_rows)
    return float(gain * (row['u_now'] - row['u_past']) * np.exp(-row['H'] / tau) + k_eq * row['u_now'])