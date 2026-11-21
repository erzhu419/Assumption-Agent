import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    scale = 0.02
    u_now = np.array([float(r['u_now']) / scale for r in train_rows], dtype=float)
    u_past = np.array([float(r['u_past']) / scale for r in train_rows], dtype=float)
    H = np.array([float(r['H']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def pred(p):
        gain, tau_relax, k_eq = p
        return gain * (u_now - u_past) * np.exp(-H / tau_relax) + k_eq * u_now
    res = least_squares(lambda p: pred(p) - y, x0=np.array([-1.0, 0.5, 0.5]), bounds=([-100.0, 1e-5, -100.0], [100.0, 100.0, 100.0]))
    return tuple(float(v) for v in res.x)

def predict(row):
    # Parameters are fitted on the training set and cached by fit.
    gain, tau_relax, k_eq = _MODEL_PARAMS
    scale = 0.02
    u_n = float(row['u_now']) / scale
    u_p = float(row['u_past']) / scale
    H = float(row['H'])
    return float(gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n)
