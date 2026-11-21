import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    H = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(log_params):
        E, eta = np.exp(log_params)
        pred = (u / E) * (-np.expm1(-E * H / eta))
        return pred - y

    result = least_squares(residual, np.log([0.5, 1.0]))
    E, eta = np.exp(result.x)
    return {'E_kv': float(E), 'eta_kv': float(eta)}

def predict(row):
    E = float(_MODEL['E_kv'])
    eta = float(_MODEL['eta_kv'])
    return (float(row['u_now']) / E) * (1.0 - math.exp(-E * float(row['H']) / eta))
