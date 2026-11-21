import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, row):
        vmax, rho, escape = p
        u = float(row['u_now'])
        s = float(row['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + escape * max(0.0, 1.0 - s)
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([12.5, 0.7, -1.0]), bounds=([0.0, 0.0, -20.0], [100.0, 1.0, 20.0]))
    params = result.x
    def predict(row):
        return float(pred(params, row))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor.')