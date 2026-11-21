import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([max(1e-8, float(r['u_now'])) for r in train_rows])
    s = np.array([max(1e-8, float(r['s_grad'])) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, km, c, beta = p
        return vmax * u / (km + u) / (1.0 + c * s**beta)
    res = least_squares(lambda p: pred(p) - y, x0=[8.0, 1.0, 1.0, 0.6], bounds=([1e-6, 1e-6, 0.0, 0.0], [100.0, 100.0, 100.0, 5.0]), max_nfev=20000)
    return res.x

def predict(row):
    vmax, km, c, beta = fit(train_rows)
    u = max(1e-8, float(row['u_now']))
    s = max(1e-8, float(row['s_grad']))
    return float(vmax * u / (km + u) / (1.0 + c * s**beta))