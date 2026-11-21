import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    starts = [
        [4.0, 1.0, 0.3, 0.05],
        [8.0, 2.0, 1.0, 0.2],
        [3.0, 0.3, 0.05, 0.01],
        [10.0, 5.0, 2.0, 0.5],
    ]
    fits = [least_squares(residual, x0, bounds=([0, 1e-8, 0, 0], [100, 100, 100, 100]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda result: np.sum(result.fun**2))
    return best.x

def predict(row):
    return float(_value(row, fit_rows_params))