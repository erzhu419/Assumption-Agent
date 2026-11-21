import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    def predict_params(p, row):
        gain, tau_relax, k_eq = p
        h = float(row['H'])
        un = float(row['u_now'])
        up = float(row['u_past'])
        return gain * (un - up) * math.exp(-h / tau_relax) + k_eq * un
    def residuals(p):
        return [predict_params(p, r) - float(r['y']) for r in train_rows]
    result = least_squares(residuals, x0=np.array([1.0, 1.0, 0.0]), bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    gain, tau_relax, k_eq = fit(_FIT_ROWS)
    return gain * (float(row['u_now']) - float(row['u_past'])) * math.exp(-float(row['H']) / tau_relax) + k_eq * float(row['u_now'])