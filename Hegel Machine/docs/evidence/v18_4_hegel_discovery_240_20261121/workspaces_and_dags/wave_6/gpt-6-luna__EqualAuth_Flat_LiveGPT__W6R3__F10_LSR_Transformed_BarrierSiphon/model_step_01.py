import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    tau = 1.0
    un = u / tau
    return un, s

def fit(train_rows):
    rows = list(train_rows)
    ys = np.array([float(r['y']) for r in rows], dtype=float)
    def pred(p, r):
        un, s = _features(r)
        j0, beta_bar, rho, c_sqrt = p
        return j0 * un * np.exp(-beta_bar / (0.5 + un)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * np.sqrt(max(un, 0.0))
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - ys
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5, 0.1]), max_nfev=20000)
    params = result.x
    def predict(row):
        return float(pred(params, row))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first to obtain a fitted predictor.')