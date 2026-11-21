import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    d = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        E, tau = p
        return E * d * np.exp(-h / tau) - y
    p0 = [500.0, 1.0]
    opt = least_squares(residual, p0, bounds=([0.0, 1e-6], [1e6, 1e3]))
    E, tau = map(float, opt.x)
    def predict(row):
        return float(E * (row['u_now'] - row['u_past']) * math.exp(-row['H'] / tau))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor')
