import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.004

def fit(train_rows):
    up = np.array([r['u_past'] / TAU_SCALE for r in train_rows], dtype=float)
    un = np.array([r['u_now'] / TAU_SCALE for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def predict_params(p):
        gain, tau_relax, k_eq = p
        return gain * (un - up) * np.exp(-h / tau_relax) + k_eq * un

    result = least_squares(
        lambda p: predict_params(p) - y,
        x0=np.array([-1.0, 0.5, 0.4]),
        bounds=([-100.0, 1e-6, -100.0], [100.0, 100.0, 100.0]),
        max_nfev=20000,
    )
    return tuple(float(v) for v in result.x)

def predict(row):
    gain, tau_relax, k_eq = fit(_FIT_ROWS)
    up = row['u_past'] / TAU_SCALE
    un = row['u_now'] / TAU_SCALE
    return float(gain * (un - up) * math.exp(-row['H'] / tau_relax) + k_eq * un)
