import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    X = np.array([[1.0, r['u_now'], r['s_grad'], r['u_now']**2, r['u_now']*r['s_grad'], r['s_grad']**2] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return X @ p - y
    p0 = np.linalg.lstsq(X, y, rcond=None)[0]
    return least_squares(residual, p0).x

def predict(row):
    p = fit([row])
    return float(np.dot(p, [1.0, row['u_now'], row['s_grad'], row['u_now']**2, row['u_now']*row['s_grad'], row['s_grad']**2]))