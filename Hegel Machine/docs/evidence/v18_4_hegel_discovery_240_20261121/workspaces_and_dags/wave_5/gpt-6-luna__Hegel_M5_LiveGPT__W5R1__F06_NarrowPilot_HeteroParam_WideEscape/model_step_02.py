import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred_params(p, r):
        vmax, rho, c_loc = p
        u = max(float(r['u_now']), 0.0)
        s = float(r['s_grad'])
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        return base + c_loc * np.sqrt(u)
    def residual(p):
        return np.array([pred_params(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 0.5, 0.1]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    params = result.x
    def predict(row):
        return float(pred_params(params, row))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor')