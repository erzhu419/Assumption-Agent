import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * u + p[1] - y
    result = least_squares(residual, x0=np.array([1.0, 0.0]))
    return {'slope': float(result.x[0]), 'intercept': float(result.x[1])}

def predict(row):
    return fit_params['slope'] * float(row['u_now']) + fit_params['intercept']