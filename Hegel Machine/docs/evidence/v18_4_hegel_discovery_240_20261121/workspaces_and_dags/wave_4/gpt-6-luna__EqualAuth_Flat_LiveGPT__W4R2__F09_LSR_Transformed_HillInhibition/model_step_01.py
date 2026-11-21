import numpy as np
from scipy.optimize import least_squares

def _value(row, name, default=0.0):
    return float(row.get(name, default))

def _predict_params(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = _value(row, 'u_now')
    s = _value(row, 's_grad')
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residuals(p):
        return np.array([_predict_params(r, p) for r in rows]) - y
    result = least_squares(residuals, x0=np.array([4.0, 1.0, 0.5, 0.1]), bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]))
    return {'params': result.x}

def predict(row):
    return float(_predict_params(row, fit.params))