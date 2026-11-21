import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, r):
        vmax, rho = p
        un = float(r['u_now']) / 4.0
        s = float(r['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 0.7]), bounds=([0.0, 0.0], [np.inf, 1.0]))
    params = result.x
    def predict(row):
        return float(pred(params, row))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first to estimate parameters')