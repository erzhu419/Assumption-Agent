import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def predict_params(row, p):
        gain, tau_relax, k_eq = p
        u_n = float(row['u_now'])
        u_p = float(row['u_past'])
        H = float(row['H'])
        return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n
    def residuals(p):
        return np.array([predict_params(r, p) - float(r['y']) for r in rows])
    result = least_squares(residuals, x0=np.array([1.0, 1.0, 0.5]), bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    return tuple(float(v) for v in result.x)

def predict(row):
    # fit(train_rows) must be called first; the fitted parameter tuple is stored below.
    gain, tau_relax, k_eq = predict.params
    return float(gain * (float(row['u_now']) - float(row['u_past'])) * math.exp(-float(row['H']) / tau_relax) + k_eq * float(row['u_now']))
