import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    q = u / (1.2 + u)
    root = np.sqrt(np.maximum(u, 0.0))
    def residual(p):
        vmax, rho, c = p
        pred = vmax * (1.0 - rho * s / (1.0 + s)) * q + c * root
        return pred - y
    result = least_squares(residual, x0=np.array([6.0, 0.5, 0.0]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return result.x

def predict(row):
    vmax, rho, c = fit([row])
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * np.sqrt(max(u, 0.0)))