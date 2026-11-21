import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = max(1e-4, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 1.0, 0.5, 0.05]), bounds=([1e-8, 1e-8, 0.0, 0.0], [100.0, 100.0, 100.0, 10.0]), max_nfev=20000)
    return result.x

def predict(row):
    return float(_pred(row, _FIT_PARAMS))