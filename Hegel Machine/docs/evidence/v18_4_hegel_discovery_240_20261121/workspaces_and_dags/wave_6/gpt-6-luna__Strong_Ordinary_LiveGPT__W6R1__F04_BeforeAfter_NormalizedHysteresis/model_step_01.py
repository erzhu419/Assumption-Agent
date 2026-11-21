import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    scale = 5.0
    def predict_params(row, p):
        gain, tau, k_eq = p[0], math.exp(p[1]), p[2]
        un = float(row['u_now']) / scale
        up = float(row['u_past']) / scale
        h = float(row['H'])
        return gain * (un - up) * math.exp(-h / tau) + k_eq * un
    def residual(p):
        return np.array([predict_params(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, np.array([1.0, math.log(5.0), 0.3]))
    gain, log_tau, k_eq = result.x
    return {'gain': float(gain), 'tau': float(math.exp(log_tau)), 'k_eq': float(k_eq)}

def predict(row):
    p = fit(train_rows)
    un = float(row['u_now']) / 5.0
    up = float(row['u_past']) / 5.0
    return float(p['gain'] * (un - up) * math.exp(-float(row['H']) / p['tau']) + p['k_eq'] * un)