import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    X = np.array([[float(r['u_now']) - float(r['u_past']), float(r['H']), float(r['x_now'])] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p, z):
        amp, tau = p
        du, H, x = z
        return x + amp * du * np.exp(-H / tau)
    def residual(p):
        return np.array([pred(p, z) for z in X]) - y
    result = least_squares(residual, x0=[140.0, 1.5], bounds=([-1e5, 1e-6], [1e5, 1e5]))
    amp, tau = result.x
    def predict(row):
        du = float(row['u_now']) - float(row['u_past'])
        H = float(row['H'])
        return float(row['x_now']) + amp * du * np.exp(-H / tau)
    return predict

def predict(row):
    return fit([])(row)
