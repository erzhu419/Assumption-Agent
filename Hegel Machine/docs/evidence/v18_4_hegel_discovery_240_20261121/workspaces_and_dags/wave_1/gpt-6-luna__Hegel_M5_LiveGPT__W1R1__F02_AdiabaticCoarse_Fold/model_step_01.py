import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    E, eta = np.exp(p)
    H = max(float(row['H']), 0.0)
    z = E * H / eta
    return float(row['u_now']) / E * (-math.expm1(-z))

def fit(train_rows):
    def residual(p):
        return np.array([_predict_params(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, np.log([0.48, 0.5]))
    p = result.x.copy()
    def predict(row):
        return _predict_params(row, p)
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor.')