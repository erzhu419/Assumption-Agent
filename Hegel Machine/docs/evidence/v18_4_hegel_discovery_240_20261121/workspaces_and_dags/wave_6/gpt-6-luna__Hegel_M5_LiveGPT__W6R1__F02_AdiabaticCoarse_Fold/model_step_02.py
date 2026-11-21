import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] for r in rows], dtype=float)
    H = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(logp):
        E, eta = np.exp(logp)
        pred = (u / E) * (-np.expm1(-E * H / eta))
        return pred - y
    result = least_squares(residual, np.log([0.5, 1.0]))
    return tuple(np.exp(result.x))

def predict(row):
    E, eta = fit(train_rows)
    return (row['u_now'] / E) * (-np.expm1(-E * row['H'] / eta))