import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(logp):
        E, eta = np.exp(logp)
        pred = (u / E) * (-np.expm1(-E * H / eta))
        return pred - y
    result = least_squares(residual, np.log([0.45, 0.4]))
    return np.exp(result.x)

def predict(row):
    E, eta = fit(_active_rows)
    return (row['u_now'] / E) * (-np.expm1(-E * row['H'] / eta))