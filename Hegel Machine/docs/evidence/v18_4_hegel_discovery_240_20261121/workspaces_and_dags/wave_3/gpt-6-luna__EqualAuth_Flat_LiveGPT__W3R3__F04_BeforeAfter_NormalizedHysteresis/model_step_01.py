import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    gain, tau_relax, k_eq = p
    scale = 0.02
    u_n = row['u_now'] / scale
    u_p = row['u_past'] / scale
    return gain * (u_n - u_p) * math.exp(-row['H'] / tau_relax) + k_eq * u_n

def fit(train_rows):
    def residual(p):
        return np.array([_value(r, p) - r['y'] for r in train_rows], dtype=float)
    result = least_squares(residual, x0=np.array([5.0, 0.5, 0.5]), bounds=([0.0, 1e-6, -np.inf], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    return float(_value(row, fit(train_rows)))