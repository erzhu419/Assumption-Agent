import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, r):
        gain, tau_relax, k_eq = p
        un = float(r['u_now'])
        up = float(r['u_past'])
        h = float(r['H'])
        return gain * (un - up) * np.exp(-h / tau_relax) + k_eq * un
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([2.0, 0.5, 0.6]), bounds=([0.0, 1e-6, -np.inf], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    gain, tau_relax, k_eq = fit([row])
    return float(gain * (float(row['u_now']) - float(row['u_past'])) * np.exp(-float(row['H']) / tau_relax) + k_eq * float(row['u_now']))