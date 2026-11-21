import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    h = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def residual(log_params):
        E, eta = np.exp(log_params)
        z = E * h / eta
        pred = (u / E) * (-np.expm1(-z))
        return pred - y
    result = least_squares(residual, np.log([0.42, 1.0]), max_nfev=20000)
    E, eta = np.exp(result.x)
    return {'E_kv': float(E), 'eta_kv': float(eta)}

def predict(row):
    E = float(row['_fit']['E_kv'])
    eta = float(row['_fit']['eta_kv'])
    u = float(row['u_now'])
    h = float(row['H'])
    return (u / E) * (-math.expm1(-E * h / eta))