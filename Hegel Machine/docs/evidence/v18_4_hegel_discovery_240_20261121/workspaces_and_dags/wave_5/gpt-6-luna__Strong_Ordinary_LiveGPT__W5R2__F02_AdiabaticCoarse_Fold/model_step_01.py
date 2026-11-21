import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        a2, a1 = p
        return a2 * u**2 + a1 * u - y
    p0 = np.array([0.0, 2.1])
    p = least_squares(residual, p0).x
    return {'a2': float(p[0]), 'a1': float(p[1])}

def predict(row):
    p = _FIT
    u = float(row['u_now'])
    return p['a2'] * u**2 + p['a1'] * u
