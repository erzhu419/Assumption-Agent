import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, row):
        vmax, k_half, alpha_inh, c_leak = p
        un = float(row['u_now'])
        sg = float(row['s_grad'])
        return vmax * (un**2 / (k_half + un**2)) / (1.0 + alpha_inh * sg) - c_leak * un
    def residual(p):
        return np.array([pred(p, r) for r in train_rows]) - y
    res = least_squares(residual, x0=np.array([5.0, 2.0, 0.5, 0.1]), bounds=([0.0, 1e-8, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]))
    params = res.x.copy()
    def predict(row):
        return float(pred(params, row))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) and use its returned predictor.')