import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    def features(r):
        den = float(r['u_past']) + float(r['u_now'])
        # The equal-input case has no hysteretic contribution.
        z = (float(r['u_past']) - float(r['u_now'])) / den if den else 0.0
        h = float(r['H'])
        return np.array([z, z*h, z*h*h], dtype=float)
    X = np.vstack([features(r) for r in rows])
    y = np.array([float(r['y']) - float(r['x_now']) for r in rows])
    def residual(p):
        return X @ p - y
    result = least_squares(residual, np.zeros(3), max_nfev=5000)
    return result.x

def predict(row):
    den = float(row['u_past']) + float(row['u_now'])
    z = (float(row['u_past']) - float(row['u_now'])) / den if den else 0.0
    h = float(row['H'])
    p = _fit_parameters
    return float(row['x_now']) + z * (p[0] + p[1]*h + p[2]*h*h)
