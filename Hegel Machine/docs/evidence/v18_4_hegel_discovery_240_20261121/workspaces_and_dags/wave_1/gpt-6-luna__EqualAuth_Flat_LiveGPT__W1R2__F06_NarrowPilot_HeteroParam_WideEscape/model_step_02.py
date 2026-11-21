import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([max(float(r['u_now']), 1e-8) for r in train_rows])
    s = np.array([max(float(r['s_grad']), 1e-8) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, km, beta = p
        return vmax * u / (km + u) * np.power(s, beta)
    result = least_squares(lambda p: pred(p) - y, x0=np.array([5.0, 1.2, -0.3]), bounds=([1e-6, 1e-6, -3.0], [100.0, 100.0, 3.0]))
    return result.x

def predict(row):
    p = fit(train_rows)
    u = max(float(row['u_now']), 1e-8)
    s = max(float(row['s_grad']), 1e-8)
    vmax, km, beta = p
    return float(vmax * u / (km + u) * (s ** beta))