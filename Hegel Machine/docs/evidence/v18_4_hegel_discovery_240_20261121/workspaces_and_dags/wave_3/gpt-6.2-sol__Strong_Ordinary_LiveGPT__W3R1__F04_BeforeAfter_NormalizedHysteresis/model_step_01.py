import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.02

def fit(train_rows):
    global params
    u_now = np.array([r['u_now'] for r in train_rows], dtype=float) / TAU_SCALE
    step = u_now - np.array([r['u_past'] for r in train_rows], dtype=float) / TAU_SCALE
    horizon = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        gain, tau, k_eq = p
        return k_eq * u_now + gain * step * np.exp(-horizon / tau) - y

    params = least_squares(residual, [-1.0, 1.0, 1.0], bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, tau, k_eq = params
    u_now = row['u_now'] / TAU_SCALE
    step = (row['u_now'] - row['u_past']) / TAU_SCALE
    return float(k_eq * u_now + gain * step * math.exp(-row['H'] / tau))