import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    scale = 0.004
    def pred(p, r):
        gain, tau, k_eq = p
        un = r['u_now'] / scale
        up = r['u_past'] / scale
        return gain * (un - up) * math.exp(-r['H'] / tau) + k_eq * un
    def residual(p):
        return np.array([pred(p, r) - r['y'] for r in rows])
    result = least_squares(residual, x0=[1.0, 1.0, 0.1], bounds=([-100.0, 1e-6, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000)
    return result.x.tolist()

def predict(row):
    gain, tau, k_eq = fit([row])
    un = row['u_now'] / 0.004
    up = row['u_past'] / 0.004
    return float(gain * (un - up) * math.exp(-row['H'] / tau) + k_eq * un)