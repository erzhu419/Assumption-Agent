import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u_n = row['u_now'] / 0.05
    s = row['s_grad']
    base = (1.0 - 0.70 * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return u_n, base

def fit(train_rows):
    X = np.array([[_features(r)[1], np.sqrt(_features(r)[0])] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    # Fit vmax, rho, and local sqrt-branch coefficient jointly.
    def residual(p):
        vmax, rho, c_loc = p
        pred = []
        for r in train_rows:
            u_n = r['u_now'] / 0.05
            s = r['s_grad']
            pred.append(vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * np.sqrt(u_n))
        return np.asarray(pred) - y
    res = least_squares(residual, x0=np.array([6.5, 0.70, 0.15]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return res.x

def predict(row):
    vmax, rho, c_loc = fit(_TRAIN_ROWS)
    u_n = row['u_now'] / 0.05
    s = row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * np.sqrt(u_n))