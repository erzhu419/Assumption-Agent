import numpy as np
from scipy.optimize import least_squares

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.5, 0.1]), bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else _FITTED_PARAMS
    return float(_model(row, p))