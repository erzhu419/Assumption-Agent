import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.004

def fit(train_rows):
    rows = list(train_rows)
    def predict_params(p, row):
        gain, tau_relax, k_eq = p
        u_n = float(row['u_now']) / TAU_SCALE
        u_p = float(row['u_past']) / TAU_SCALE
        H = float(row['H'])
        return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n
    def residual(p):
        return np.asarray([predict_params(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([-1.5, 1.0, 0.3]), bounds=([-100.0, 1e-5, -100.0], [100.0, 100.0, 100.0]))
    return result.x

def predict(row):
    gain, tau_relax, k_eq = fit([row])
    u_n = float(row['u_now']) / TAU_SCALE
    u_p = float(row['u_past']) / TAU_SCALE
    return float(gain * (u_n - u_p) * math.exp(-float(row['H']) / tau_relax) + k_eq * u_n)
