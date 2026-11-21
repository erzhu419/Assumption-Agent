import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    tau_scale = 0.02
    def predict_params(p, row):
        gain, tau_relax, k_eq = p
        u_n = row['u_now'] / tau_scale
        u_p = row['u_past'] / tau_scale
        return gain * (u_n - u_p) * math.exp(-row['H'] / tau_relax) + k_eq * u_n
    def residual(p):
        return np.array([predict_params(p, r) - r['y'] for r in train_rows])
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5]), bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    gain, tau_relax, k_eq = _fitted_params
    u_n = row['u_now'] / 0.02
    u_p = row['u_past'] / 0.02
    return float(gain * (u_n - u_p) * math.exp(-row['H'] / tau_relax) + k_eq * u_n)
