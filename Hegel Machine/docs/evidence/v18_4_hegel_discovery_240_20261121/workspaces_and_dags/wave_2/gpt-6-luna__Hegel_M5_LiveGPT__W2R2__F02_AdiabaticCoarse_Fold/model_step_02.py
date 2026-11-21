import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        E, eta = np.exp(p)
        pred = (u / E) * (1.0 - np.exp(-E * H / eta))
        return pred - y
    result = least_squares(residual, np.log([0.5, 0.6]))
    E, eta = np.exp(result.x)
    def predict(row):
        return float((row['u_now'] / E) * (1.0 - math.exp(-E * row['H'] / eta)))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor')