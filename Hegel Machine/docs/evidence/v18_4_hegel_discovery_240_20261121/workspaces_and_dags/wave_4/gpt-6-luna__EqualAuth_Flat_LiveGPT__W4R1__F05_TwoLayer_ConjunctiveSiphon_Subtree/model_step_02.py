import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([max(1e-6, r['u_now']) for r in train_rows], dtype=float)
    s = np.array([max(0.0, r['s_grad']) for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        vmax, rho, c, km = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * np.sqrt(u)
    result = least_squares(lambda p: pred(p) - y, x0=[5.0, 0.4, 0.1, 2.0], bounds=([0.0, 0.0, 0.0, 1e-6], [20.0, 1.0, 5.0, 50.0]), max_nfev=20000)
    return result.x

def predict(row):
    p = fit(_train_rows)
    u = max(1e-6, row['u_now'])
    s = max(0.0, row['s_grad'])
    vmax, rho, c, km = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * np.sqrt(u))