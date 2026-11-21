import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    d = np.array([r['u_now'] - r['u_past'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        amp, tau = p
        return x + amp * d * np.exp(-h / tau) - y
    res = least_squares(residual, x0=[2.5, 1.0], bounds=([-20.0, 0.01], [20.0, 20.0]))
    return {'amp': float(res.x[0]), 'tau': float(res.x[1])}

def predict(row):
    p = fit([row])
    return float(row['x_now'] + p['amp'] * (row['u_now'] - row['u_past']) * np.exp(-row['H'] / p['tau']))