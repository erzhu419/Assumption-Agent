import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    a, k, rho, c, t = p
    u = max(1e-4, float(row['u_now']))
    s = float(row['s_grad'])
    baseline = a * (1.0 - rho * s / (1.0 + s)) * (u / (k + u))
    branch = c * max(0.0, s - t) ** 2
    return baseline + branch

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=[5.0, 1.2, 0.7, -1.0, 2.7], bounds=([0.01, 0.01, 0.0, -20.0, 0.0], [30.0, 20.0, 0.999, 20.0, 10.0]), max_nfev=20000)
    return result.x.tolist()

def predict(row):
    if not hasattr(predict, '_params'):
        raise RuntimeError('Call fit(train_rows) and set predict._params to its result before prediction')
    return float(_pred(row, predict._params))