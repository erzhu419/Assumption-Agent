import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    k_ss, a_eff = p
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    return k_ss*u + (x-k_ss*u)*np.exp(-a_eff*H)

def fit(train_rows):
    def residual(p):
        return np.array([_pred(r,p)-float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([2.4, 1.0]), bounds=([0.0, 0.0],[np.inf, np.inf]))
    return result.x

def predict(row):
    return float(_pred(row, _fitted_parameters))