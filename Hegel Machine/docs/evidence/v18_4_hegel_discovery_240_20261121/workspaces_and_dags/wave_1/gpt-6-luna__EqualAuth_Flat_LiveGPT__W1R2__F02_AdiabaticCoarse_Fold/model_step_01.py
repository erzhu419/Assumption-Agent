import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    x = np.asarray([r['x_now'] for r in train_rows], dtype=float)
    H = np.asarray([r['H'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        k_ss, a_eff = p
        return k_ss*u + (x-k_ss*u)*np.exp(-a_eff*H) - y
    result = least_squares(residual, x0=np.array([1.25, 1.5]), bounds=([0.0, 0.0], [10.0, 20.0]))
    return {'k_ss': float(result.x[0]), 'a_eff': float(result.x[1])}

def predict(row):
    k_ss = float(row['_fit']['k_ss'])
    a_eff = float(row['_fit']['a_eff'])
    x_ss = k_ss * float(row['u_now'])
    return x_ss + (float(row['x_now']) - x_ss) * math.exp(-a_eff * float(row['H']))