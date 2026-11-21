import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def predict_params(p, r):
        k_ss, a_eff = p
        e = math.exp(-a_eff * float(r['H']))
        x_ss = k_ss * float(r['u_now'])
        return x_ss + (float(r['x_now']) - x_ss) * e
    def residual(p):
        return np.array([predict_params(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.8, 0.1]), bounds=([0.0, 0.0], [20.0, 20.0]))
    return result.x

def predict(row):
    k_ss, a_eff = fit([row])
    e = math.exp(-a_eff * float(row['H']))
    x_ss = k_ss * float(row['u_now'])
    return float(x_ss + (float(row['x_now']) - x_ss) * e)