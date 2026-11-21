import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] / 0.02 for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p):
        vmax, rho, leak = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) - leak * u
    res = least_squares(lambda p: pred(p) - y, x0=[8.0, 0.7, 0.0], bounds=([0.0, 0.0, -10.0], [30.0, 1.0, 10.0]), max_nfev=20000)
    return res.x

def predict(row):
    p = fit([row])
    u = row['u_now'] / 0.02
    s = row['s_grad']
    vmax, rho, leak = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) - leak * u)
