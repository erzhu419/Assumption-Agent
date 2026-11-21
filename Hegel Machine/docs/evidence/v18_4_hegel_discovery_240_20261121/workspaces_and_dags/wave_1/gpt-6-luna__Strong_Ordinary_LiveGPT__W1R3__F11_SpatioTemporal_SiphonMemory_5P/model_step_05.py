import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p, ui, si):
        vmax, kh, alpha, leak = p
        return vmax * (ui**2 / (kh + ui**2)) / (1.0 + alpha * si) - leak * ui
    def residual(p):
        return pred(p, u, s) - y
    result = least_squares(residual, [3.0, 1.0, 0.5, 0.1], bounds=([0.0, 1e-8, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]), max_nfev=20000)
    fit.params_ = result.x
    return fit

def predict(row):
    vmax, kh, alpha, leak = fit.params_
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (u*u / (kh + u*u)) / (1.0 + alpha*s) - leak*u)