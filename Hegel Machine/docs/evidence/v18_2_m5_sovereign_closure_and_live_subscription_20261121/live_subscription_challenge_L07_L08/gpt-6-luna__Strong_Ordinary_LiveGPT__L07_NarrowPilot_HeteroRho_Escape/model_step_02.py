import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        vmax, rho, leak = p
        pred = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) - leak * u
        return pred - y
    result = least_squares(residual, x0=np.array([12.0, 0.7, 0.1]), bounds=([0.0, 0.0, 0.0], [100.0, 2.0, 10.0]))
    return result.x

def predict(row):
    vmax, rho, leak = fit(_TRAIN_ROWS)
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) - leak * u)
