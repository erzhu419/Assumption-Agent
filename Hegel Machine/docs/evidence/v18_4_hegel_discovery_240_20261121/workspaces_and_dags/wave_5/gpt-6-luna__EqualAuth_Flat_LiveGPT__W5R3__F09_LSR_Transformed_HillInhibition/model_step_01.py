import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) / 0.1 for r in train_rows], dtype=float)
    s = np.array([float(r['s_grad']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)

    def predict_params(p, uu, ss):
        vmax, k_half, alpha_inh, c_leak = p
        return vmax * (uu ** 2 / (k_half + uu ** 2)) / (1.0 + alpha_inh * ss) - c_leak * uu

    def residuals(p):
        return predict_params(p, u, s) - y

    result = least_squares(residuals, x0=np.array([5.0, 1.0, 0.5, 0.1]), bounds=(np.zeros(4), np.full(4, np.inf)), max_nfev=20000)
    p = result.x
    def predict(row):
        uu = float(row['u_now']) / 0.1
        ss = float(row['s_grad'])
        return float(predict_params(p, uu, ss))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor')