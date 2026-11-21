import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * np.sqrt(np.maximum(u, 0.0))
    result = least_squares(lambda p: pred(p) - y, x0=np.array([6.5, 0.7, 0.3]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return result.x

def predict(row):
    vmax, rho, c = fit(_FIT_ROWS)
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * np.sqrt(max(u, 0.0)))