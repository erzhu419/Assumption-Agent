import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    vmax, rho, c_loc = p
    u_n = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * np.sqrt(max(0.0, u_n))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.14]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return result.x

def predict(row):
    raise RuntimeError('Call fit(train_rows) first; use the fitted parameters with _predict_params(row, params).')