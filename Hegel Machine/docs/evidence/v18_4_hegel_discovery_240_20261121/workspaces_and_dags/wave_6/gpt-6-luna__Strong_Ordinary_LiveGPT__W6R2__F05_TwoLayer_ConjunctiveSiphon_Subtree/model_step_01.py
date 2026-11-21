import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    vmax, rho, c_loc = p
    u_n = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * np.sqrt(max(0.0, u_n))

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in train_rows]) - y
    starts = ([2.0, 0.5, 0.0], [2.0, 0.5, 0.5], [2.0, 0.5, -0.5])
    fits = [least_squares(residual, np.array(p, dtype=float), max_nfev=20000) for p in starts]
    best = min(fits, key=lambda result: np.sum(result.fun ** 2))
    return best.x

def predict(row):
    p = fit([row])
    return float(_predict_params(row, p))