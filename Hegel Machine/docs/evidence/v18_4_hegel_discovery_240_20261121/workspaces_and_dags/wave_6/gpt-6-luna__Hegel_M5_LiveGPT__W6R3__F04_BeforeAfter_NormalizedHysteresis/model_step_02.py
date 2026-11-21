import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    def residual(p):
        E, tau = p
        pred = np.array([r['x_now'] + E * (r['u_now'] - r['u_past']) * np.exp(-r['H'] / tau) for r in rows])
        return pred - np.array([r['y'] for r in rows])
    result = least_squares(residual, x0=np.array([120.0, 1.0]), bounds=([0.0, 1e-6], [1e5, 1e5]))
    return {'E': float(result.x[0]), 'tau': float(result.x[1])}

def predict(row):
    p = fit([row])
    return float(row['x_now'] + p['E'] * (row['u_now'] - row['u_past']) * np.exp(-row['H'] / p['tau']))
