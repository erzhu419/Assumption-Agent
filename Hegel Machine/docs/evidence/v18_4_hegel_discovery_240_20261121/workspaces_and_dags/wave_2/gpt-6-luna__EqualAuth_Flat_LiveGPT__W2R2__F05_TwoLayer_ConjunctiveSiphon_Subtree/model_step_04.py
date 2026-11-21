import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p, r):
        vmax, rho, c_sqrt = p
        u = max(1e-4, float(r['u_now']) / 4.0)
        s = max(0.0, float(r['s_grad']))
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_sqrt * np.sqrt(u)
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.7, 0.2]), bounds=([0.0, 0.0, 0.0], [20.0, 1.5, 5.0]))
    return result.x

def predict(row):
    p = fit(train_rows)
    u = max(1e-4, float(row['u_now']) / 4.0)
    s = max(0.0, float(row['s_grad']))
    vmax, rho, c_sqrt = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_sqrt * np.sqrt(u))