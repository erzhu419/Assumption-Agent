import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    if len(rows) < 3:
        raise ValueError('At least three rows are required to fit the three parameters.')
    y = np.array([float(r['y']) for r in rows])
    d = np.array([float(r['u_now']) - float(r['u_past']) for r in rows])
    h = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    def predict_params(p):
        gain, tau, k_eq = p
        return gain * d * np.exp(-h / tau) + k_eq * u
    def residual(p):
        return predict_params(p) - y
    p0 = np.array([1.0, 1.0, 0.5])
    result = least_squares(residual, p0, bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]))
    return {'gain': float(result.x[0]), 'tau_relax': float(result.x[1]), 'k_eq': float(result.x[2])}

def predict(row):
    p = fit(row['_train_rows'])
    u_now = float(row['u_now'])
    u_past = float(row['u_past'])
    h = float(row['H'])
    return float(p['gain'] * (u_now - u_past) * np.exp(-h / p['tau_relax']) + p['k_eq'] * u_now)