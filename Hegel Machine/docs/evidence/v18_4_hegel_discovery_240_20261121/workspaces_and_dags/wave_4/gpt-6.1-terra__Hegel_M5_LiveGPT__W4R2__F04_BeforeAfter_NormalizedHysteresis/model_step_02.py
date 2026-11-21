import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = [r for r in train_rows if abs(r['u_past'] - r['u_now']) > 1e-12]
    h = np.array([r['H'] for r in rows], dtype=float)
    d = np.array([r['u_past'] - r['u_now'] for r in rows], dtype=float)
    z = np.array([r['y'] - r['x_now'] for r in rows], dtype=float)
    def residual(p):
        return d * np.exp(np.clip(p[0] + p[1]*h + p[2]*h**2 + p[3]*h**3, -50, 50)) - z
    initial = np.polyfit(h, np.log(np.maximum(np.abs(z / d), 1e-12)), 3)[::-1]
    _params = least_squares(residual, initial, max_nfev=5000).x

def predict(row):
    h = float(row['H'])
    p = _params
    response = np.exp(np.clip(p[0] + p[1]*h + p[2]*h**2 + p[3]*h**3, -50, 50))
    return float(row['x_now'] + (row['u_past'] - row['u_now']) * response)
