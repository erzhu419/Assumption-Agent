import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        amp, km = p
        return amp * u / (km + u) - y
    result = least_squares(residual, x0=np.array([4.4, 1.2]), bounds=([0.0, 1e-8], [np.inf, np.inf]))
    return {'amp': float(result.x[0]), 'km': float(result.x[1])}

def predict(row):
    p = fit([{'u_now': 0.5, 'y': 1.3077106}, {'u_now': 1.2, 'y': 2.2584047}, {'u_now': 1.9, 'y': 2.756583}, {'u_now': 2.7, 'y': 3.0851843}, {'u_now': 3.5, 'y': 3.2843683}])
    u = float(row['u_now'])
    return float(p['amp'] * u / (p['km'] + u))