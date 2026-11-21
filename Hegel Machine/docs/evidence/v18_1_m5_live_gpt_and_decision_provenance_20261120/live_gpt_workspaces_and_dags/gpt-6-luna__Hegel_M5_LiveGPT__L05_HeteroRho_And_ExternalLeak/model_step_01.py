import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred_params(p, r):
        vmax, rho, branch, leak = p
        u = float(r['u_now'])
        s = float(r['s_grad'])
        x = float(r['x_now'])
        H = float(r['H'])
        u_n = u  # tau_scale = 1.0
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
        return base + branch * x - leak * u_n * H
    def residual(p):
        return np.asarray([pred_params(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([6.0, 0.70, 0.1, 0.0]), bounds=([1.0, 0.05, -2.0, -10.0], [15.0, 0.95, 2.0, 10.0]))
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(pred_params(p, row))