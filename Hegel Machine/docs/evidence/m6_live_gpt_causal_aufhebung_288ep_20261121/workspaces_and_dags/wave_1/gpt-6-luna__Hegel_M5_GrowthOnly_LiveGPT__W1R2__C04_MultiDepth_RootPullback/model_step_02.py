import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[float(r['u_outdoor']), float(r['u_indoor']), float(r['p_weather']), float(r['a_foot']), float(r['impact_idx'])] for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    E = X[:, 0] * X[:, 2] + X[:, 1]
    impact = X[:, 3] * X[:, 4]
    def residual(p):
        gain, k, loss = p
        pred = gain * E / (k + E) - loss * impact
        return pred - y
    result = least_squares(residual, x0=np.array([7.0, 1.0, 1.0]), bounds=([0.0, 1e-9, 0.0], [100.0, 100.0, 100.0]))
    return result.x

def predict(row):
    gain, k, loss = fit(_FIT_ROWS)
    E = float(row['u_outdoor']) * float(row['p_weather']) + float(row['u_indoor'])
    return float(gain * E / (k + E) - loss * float(row['a_foot']) * float(row['impact_idx']))