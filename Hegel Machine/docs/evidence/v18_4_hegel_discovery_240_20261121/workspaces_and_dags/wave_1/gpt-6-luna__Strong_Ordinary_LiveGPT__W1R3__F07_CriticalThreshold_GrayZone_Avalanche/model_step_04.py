import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(float(row['u_now']), 1e-9)
    s = float(row['s_grad'])
    return u, s

def _predict_params(row, p):
    u, s = _features(row)
    vmax, rho, c_sqrt = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_sqrt * np.sqrt(u)

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([8.0, 0.7, -1.0]), max_nfev=20000)
    return result.x

def predict(row):
    return float(_predict_params(row, _FIT_PARAMS))