import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    scale = 0.004
    up = np.array([r['u_past'] / scale for r in train_rows], dtype=float)
    un = np.array([r['u_now'] / scale for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        gain, tau, k_eq = p
        return gain * (un - up) * np.exp(-H / tau) + k_eq * un - y
    result = least_squares(residual, x0=np.array([-1.0, 0.5, 0.4]), bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    gain, tau, k_eq = fit(_TRAIN_ROWS)
    scale = 0.004
    up = row['u_past'] / scale
    un = row['u_now'] / scale
    return float(gain * (un - up) * np.exp(-row['H'] / tau) + k_eq * un)
