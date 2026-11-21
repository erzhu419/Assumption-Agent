import math
import numpy as np
from scipy.optimize import least_squares

def _features(rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    p = np.array([r['u_past'] for r in rows], dtype=float)
    return u, s, p

def fit(train_rows):
    u, s, p = _features(train_rows)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(theta):
        vmax, rho, km, branch = theta
        base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u)
        return base + branch * p / (1.2 + p)
    result = least_squares(lambda theta: pred(theta) - y,
                           x0=np.array([4.0, 0.70, 1.2, 0.2]),
                           bounds=([0.01, 0.0, 0.01, -5.0], [20.0, 0.99, 20.0, 5.0]),
                           max_nfev=20000)
    return result.x

def predict(row):
    theta = fit(train_rows)
    u = float(row['u_now'])
    s = float(row['s_grad'])
    p = float(row['u_past'])
    vmax, rho, km, branch = theta
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + branch * p / (1.2 + p))