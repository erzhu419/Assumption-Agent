import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    delta = np.array([float(r['u_now']) - float(r['u_past']) for r in train_rows])
    h = np.array([float(r['H']) for r in train_rows])
    x = np.array([float(r['x_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        amplitude, log_tau = p
        tau = math.exp(log_tau)
        return x + amplitude * delta * np.exp(-h / tau) - y
    result = least_squares(residual, np.array([2.4, math.log(1.8)]))
    return result.x

def predict(row):
    amplitude, log_tau = fit(train_rows)
    tau = math.exp(log_tau)
    return float(row['x_now']) + amplitude * (float(row['u_now']) - float(row['u_past'])) * math.exp(-float(row['H']) / tau)